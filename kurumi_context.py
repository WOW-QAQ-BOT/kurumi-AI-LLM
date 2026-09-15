# -*- coding: utf-8 -*-
"""统一的人物与会话上下文组装。

聊天与 Agent **共享同一份人物状态与同一套预算**,只调整"当前表达场景" —— 两边预算不一致,
同一人物在两种模式下"记得的事"就会不同。

组装规则:
- **整轮选取**:从最近一轮向前累积到预算上限,绝不从消息中间切断 —— 从中间切断会腰斩
  路径/文件名、整段丢掉约定,还可能从一个 assistant 回复的半句开始;
- **当前指令永不截断**:预算只作用于"以往对话",不作用于主人这一次的输入;
- **省略明说**:丢掉了更早的轮次就写明省略了多少轮,让模型知道上下文不完整;
- **说话人显式标注**:转录开头说明「主人」是用户、「狂三」是模型自己 —— 历史若被拼成
  一条 user 输入,模型就会把狂三的话当成主人说的。
"""
from dataclasses import dataclass

from persona import DEFAULT_PERSONA, PersonaProfile


@dataclass(frozen=True)
class ContextBudget:
    """上下文预算的单一事实来源(聊天与 Agent 共用)。"""

    history_max_chars: int = 16000      # 与 --history_max_chars 同源
    memory_max_chars: int = 6000        # 两模式共用(约 30 条 × 200 字符的量级)
    rag_top_k: int = 3
    agent_transcript_max_chars: int = 6000   # Agent 转录上限(整轮选取,不从消息中间截断)


@dataclass(frozen=True)
class Turn:
    role: str        # user / assistant
    content: str


@dataclass(frozen=True)
class ChatContext:
    system: str
    messages: tuple


@dataclass(frozen=True)
class AgentContext:
    """一次 Agent 运行所需的上下文。

    - `turns`:以往对话的**结构化轮次**,直接作为 Responses 的 input 项发送,
      保留 user/assistant 角色边界(不得把历史压成一条 user 输入);
    - `instruction`:主人当前指令,永不截断;
    - `instructions_suffix`:本次特有的补充材料(如知识库设定参考),追加到
      runner 的 instructions 之后 —— 静态人设/记忆仍由 instructions_provider 提供,
      两者互不覆盖;
    - `transcript`:人类可读转录,用于审计、日志与单字符串接口回退。
    """

    turns: tuple = ()
    instruction: str = ""
    instructions_suffix: str = ""
    omitted_turns: int = 0
    transcript: str = ""

    def input_items(self) -> tuple:
        """转成 Responses 的 input 列表:以往轮次 + 本次指令。"""
        items = [{"role": t.role, "content": t.content} for t in self.turns]
        items.append({"role": "user", "content": self.instruction})
        return tuple(items)


@dataclass(frozen=True)
class RunContext:
    """交给 AgentRunner 的一次运行上下文。

    只带"随本次运行变化"的东西:结构化历史、本次指令、本次的补充材料。
    静态指令(人设/记忆/工作目录)仍由 runner 构造时的 instructions 提供,
    避免两处互相覆盖。
    """

    input_items: tuple
    instructions_suffix: str = ""
    # 本次运行所属的会话(界面传进来,把 Agent 运行与它绑定)。缺省空串 ——
    # 调用方不传时行为不变(运行照旧落库,只是不带会话归属)。
    session_id: str = ""

    @classmethod
    def from_agent_context(cls, ctx: AgentContext, session_id: str = "") -> "RunContext":
        return cls(input_items=ctx.input_items(), instructions_suffix=ctx.instructions_suffix,
                   session_id=str(session_id or ""))


def _message_text(msg) -> str:
    if not isinstance(msg, dict):
        return ""
    return str(msg.get("content", "") or "")


def _to_turns(history) -> list:
    """把 history 归并成"轮次":连续的 user/assistant 各成一条,其余角色保留原文。

    角色标签统一由 persona 提供,不再各处硬编码。
    """
    turns = []
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        text = _message_text(msg)
        if not text.strip():
            continue
        if role not in ("user", "assistant"):
            role = "system"      # 工具/系统内容:保留但不冒充任何一方
        turns.append(Turn(role=role, content=text))
    return turns


def select_turns(history, max_chars: int):
    """从最近的轮次向前取,直到超预算;返回 (保留的轮次, 省略的轮数)。

    只按**整条消息**取舍,绝不截断消息内容 —— 半个路径比没有路径更危险。
    """
    turns = _to_turns(history)
    kept = []
    used = 0
    for turn in reversed(turns):
        cost = len(turn.content)
        if kept and used + cost > max_chars:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    return kept, len(turns) - len(kept)


def select_memories(memories, max_chars: int, kinds=None, include_fiction: bool = False) -> list:
    """按字符预算选取记忆内容(两种模式共用同一份检索结果)。

    选取交由 `memory_model.select_entries` 决定,因此会:
    - 跳过已失效(被替代/被忘记)的条目;
    - 默认跳过 `kind=fiction`(角色虚构经历不是主人的真实经历);
    - 置顶条目优先且必留;
    - 超长条目整条丢弃,不切一半。
    """
    import memory_model

    entries = []
    for item in (memories or []):
        if isinstance(item, dict):
            normalized = memory_model.normalize_entry(item)
        else:
            # 兼容纯字符串形式的记忆
            text = str(item or "").strip()
            normalized = memory_model.normalize_entry({"content": text}) if text else None
        if normalized:
            entries.append(normalized)
    picked = memory_model.select_entries(entries, max_chars,
                                         kinds=kinds, include_fiction=include_fiction)
    return [str(e.get("content", "")).strip() for e in picked]


class ContextBuilder:
    """按统一预算组装聊天与 Agent 的上下文。"""

    def __init__(self, persona: PersonaProfile = None, budget: ContextBudget = None):
        self.persona = persona or DEFAULT_PERSONA
        self.budget = budget or ContextBudget()

    # ---------- 共用片段 ----------
    def memory_block(self, memories) -> str:
        lines = select_memories(memories, self.budget.memory_max_chars)
        if not lines:
            return ""
        return ("\n【你与主人之间的重要回忆（请自然地融入对话，不要逐条复述）】\n"
                + "\n".join("- " + line for line in lines) + "\n")

    def rag_block(self, query) -> str:
        from knowledge import build_rag_context
        try:
            return build_rag_context(query, top_k=self.budget.rag_top_k)
        except Exception as e:
            print(f"[!] 知识库检索失败,本次不注入设定参考: {e}")
            return ""

    def transcript(self, history, max_chars: int = None):
        """以往对话的整轮转录 + 省略说明。返回 (文本, 省略轮数)。"""
        limit = self.budget.agent_transcript_max_chars if max_chars is None else max_chars
        kept, omitted = select_turns(history, limit)
        if not kept:
            return "", omitted
        labels = self.persona.role_labels()
        lines = []
        if omitted:
            lines.append(f"（更早的 {omitted} 轮对话因长度上限已省略）")
        for turn in kept:
            label = labels.get(turn.role, "（系统）")
            lines.append(f"{label}：{turn.content}")
        return "\n".join(lines), omitted

    # ---------- 两种场景 ----------
    def build_chat(self, history, memories, query) -> ChatContext:
        """普通聊天:保留消息角色(结构化 messages),人设/记忆/设定走 system。

        system 用**逐字原文**的人设(`system_prompt`),再追加**附加事实**:
        知识库设定、记忆、能力说明。人设原文不改 —— 本地 LoRA 是按它训练的。

        为什么要加"能力说明":不加时,主人说"读取桌面上的某个文件",
        聊天引擎会**声称自己没有这个能力**("我可没有窥探您桌面文件的能力呢"),
        而实际上任务模式完全能读 —— 模型在说假话,主人于是被误导。
        这段只陈述程序事实,不改人设语气。
        """
        system = (self.persona.system_prompt
                  + self.rag_block(query)
                  + self.memory_block(memories)
                  + self.capability_block())
        return ChatContext(system=system, messages=tuple(dict(m) for m in (history or [])))

    def capability_block(self, allowed_root=None) -> str:
        """当前程序**真实具备**的能力(只写事实,不写人设口吻)。

        `allowed_root` 拿不到时只说"允许根目录内的文件",不编造具体路径。
        默认从实例属性取(调用方可注入);显式传参优先。
        """
        root = str(allowed_root if allowed_root is not None
                   else getattr(self, "allowed_root", "") or "").strip()
        scope = f"{root} 内的文件" if root else "允许根目录内的文件"
        return (
            "\n【来自程序的系统通知 —— 以下不是角色设定，优先级高于上文的角色扮演】\n"
            f"- 你（这个程序）**确实具备**文件与系统操作能力：可以通过「任务模式」读写、"
            f"整理 {scope}，也能在主人批准后打开应用或网址。\n"
            "- 因此当主人要求你读写/整理文件、打开应用时，**绝对不要回答「我没有这个能力」"
            "「我读不了文件」之类的话** —— 那是错误信息，会让主人以为这件事做不到。\n"
            "- 正确做法：用一句人设语气回应，并表示主人点消息旁的「执行这个任务」按钮就能让你"
            "真的去做（每一步都有审批与审计）。\n"
            "- 主人只是在闲聊或问设定时，正常聊天即可，不必提任务模式。\n"
        )

    def agent_instructions(self, memories) -> str:
        """Agent 的 instructions(安全规则 + 执行规则 + 人设 + **预算内**记忆)。

        单独抽出来是因为它有两个消费者:一次运行的转录(`build_agent`),以及
        runner 每轮使用的 `instructions_provider`。两处必须用同一份预算,
        否则"聊天与 Agent 记得的事不同"这个毛病会从后门回来。
        """
        from agent.prompt import build_agent_instructions

        return build_agent_instructions(
            self.persona, select_memories(memories, self.budget.memory_max_chars))

    def build_agent(self, history, memories, query) -> AgentContext:
        """Agent:结构化轮次 + 本次指令 + 本次补充材料(知识库设定参考)。

        关键约束:历史按**整轮**选取,绝不 `[-3000:]` 从中切断;当前指令单独成段且永不截断;
        省略了多少轮会明说;历史以**结构化角色**发送而不是压成一条 user 输入。
        """
        kept, omitted = select_turns(history, self.budget.agent_transcript_max_chars)
        transcript, _ = self.transcript(history)
        return AgentContext(
            turns=tuple(kept),
            instruction=query,
            instructions_suffix=self.rag_block(query),
            omitted_turns=omitted,
            transcript=transcript,
        )

    def compose_request(self, agent_context: AgentContext) -> str:
        """把 AgentContext 拼成单字符串请求(回退路径)。

        正常路径直接传结构化的 `RunContext`,这里只用于
        调用方仍需要单字符串的场合(例如审计展示)。
        """
        parts = []
        if agent_context.transcript:
            parts.append(self.persona.transcript_header())
            parts.append(agent_context.transcript)
        parts.append("【主人当前指令】\n" + agent_context.instruction)
        return "\n".join(parts)
