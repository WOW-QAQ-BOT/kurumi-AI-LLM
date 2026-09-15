# -*- coding: utf-8 -*-
"""会话状态与不变量的唯一所有者。

"会话"逻辑集中在一处管理,下面两条不变量必须同时成立,任一引擎路径都不能例外:

1. **Agent 路径同样裁剪历史** —— 裁剪不能只发生在聊天发送时
   (`on_send` → `_trim_history`),否则只用 Agent 时历史会无限增长;
2. **Agent 回合同样计入轮数** —— `turn_count` 由本服务统一自增,
   否则"每 N 轮自动整理记忆"在使用 Agent 时永远不会触发。

这里把"追加/完成/回滚/裁剪/计数"收进一个纯 Python 类(不依赖 Qt,可离线单独验证),
UI 只做薄薄一层委托。
"""
from dataclasses import dataclass

# 引擎标记:仅用于审计与回放,**不进模型消息**(消息只保留 role/content)
ENGINE_LOCAL = "local"
ENGINE_API = "api"
ENGINE_AGENT = "agent"


@dataclass(frozen=True)
class TurnRecord:
    role: str
    content: str
    engine: str = ""

    def as_message(self) -> dict:
        """转成喂给模型的消息(只保留 role/content,不带内部字段)。"""
        return {"role": self.role, "content": self.content}


def trim_messages(messages, max_chars: int) -> list:
    """按字符预算保留最近的消息,并保证以 user 开头(角色交替)。

    只丢**整条**消息,绝不截断内容;单条超预算时仍保留它(半个路径比没有路径更危险)。
    """
    if not messages or max_chars <= 0:
        return list(messages or [])
    kept = []
    total = 0
    for msg in reversed(messages):
        cost = len(str(msg.get("content", "") or ""))
        if kept and total + cost > max_chars:
            break
        kept.append(msg)
        total += cost
    kept.reverse()
    while kept and kept[0].get("role") == "assistant":
        kept.pop(0)
    return kept


class ConversationService:
    """会话状态机:谁说了什么、用的是哪个引擎、失败时回滚到哪一步。"""

    def __init__(self, history_max_chars: int = 16000):
        self.history_max_chars = int(history_max_chars or 0)
        self._turns: list = []
        self.turn_count = 0        # 已完成的对话轮数(两种引擎都计)
        self._open_turn = False    # 是否有一轮"已发起但未结束"
        self._counted_pending = False   # 最近一次 complete_turn 是否还没被回滚

    # ---------- 读 ----------
    def snapshot(self) -> list:
        """给上下文组装的只读快照(普通 dict 列表,调用方可安全 deepcopy/切片)。"""
        return [t.as_message() for t in self._turns]

    def records(self) -> tuple:
        return tuple(self._turns)

    def engines(self) -> list:
        """各回合的引擎标记(审计/回放用)。"""
        return [t.engine for t in self._turns]

    def __len__(self) -> int:
        return len(self._turns)

    # ---------- 写 ----------
    def begin_turn(self, user_text: str, engine: str = "") -> None:
        """开始一轮:追加 user 消息并裁剪。

        **两种引擎都必须走这里** —— Agent 路径只 append 不裁剪会让历史无限增长。
        """
        self._turns.append(TurnRecord("user", str(user_text), engine))
        self._trim()
        self._open_turn = True
        self._counted_pending = False

    def complete_turn(self, assistant_text: str, engine: str = "") -> None:
        """完成一轮:追加 assistant、累加轮数、裁剪。"""
        self._turns.append(TurnRecord("assistant", str(assistant_text), engine))
        self.turn_count += 1
        self._open_turn = False
        # 记下"这一轮已计入":Agent 可能先收到 final_text 再被判失败,
        # 那时必须把轮数一起撤回去,否则失败的任务会虚增轮数并提前触发记忆整理。
        self._counted_pending = True
        self._trim()

    def fail_turn(self, assistant_appended: bool = False) -> None:
        """失败/取消后回滚本轮:移除孤立的 assistant 与未配对的 user。

        - 聊天失败:assistant 从未写入 → `assistant_appended=False`;
        - Agent 失败:`final_text` 可能已经写入过 assistant → `assistant_appended=True`。
        """
        if assistant_appended and self._turns and self._turns[-1].role == "assistant":
            self._turns.pop()
            if self._counted_pending:
                self.turn_count = max(0, self.turn_count - 1)   # 计数与内容一起撤销
        self._counted_pending = False
        while self._turns and self._turns[-1].role == "user":
            self._turns.pop()      # 未被回复的 user 会污染下一轮上下文
        self._open_turn = False

    def append_note(self, text: str, engine: str = "", context_text: str = "") -> None:
        """追加一条**事实记录**(不是模型回复)。

        用途:任务失败/取消但已产生落盘副作用时,把"实际做成了什么"留在上下文里,
        否则下一轮被问"文件生成了吗"时,模型与程序都无从知晓。

        - **不计入 turn_count**:它不是一轮对话,虚增计数会提前触发记忆整理;
        - **不改变 `_open_turn`/`_counted_pending`**:它发生在失败回滚之后,
          不能把已经关掉的那一轮重新算作未完成;
        - `context_text` **必须给**:`trim_messages` 会为保证角色交替而丢掉
          开头的 assistant 消息,单独一条 assistant 记录会被 `begin_turn`
          触发的裁剪直接删掉。因此这里配一条 user 侧标记,让记录以
          "user → assistant"的完整形态留在历史里。
        """
        text = str(text or "").strip()
        if not text:
            return
        context_text = str(context_text or "").strip()
        if context_text:
            self._turns.append(TurnRecord("user", context_text, engine))
        self._turns.append(TurnRecord("assistant", text, engine))
        self._trim()

    def clear(self) -> None:
        self._turns.clear()
        self._open_turn = False
        # 注意:turn_count 刻意不清零 —— "每 N 轮整理记忆"是按会话累计的节奏,
        # 清空聊天记录不代表记忆节奏重新开始。

    def replace_history(self, messages) -> None:
        """用外部列表整体替换(用于加载/恢复会话)。"""
        self._turns = [TurnRecord(str(m.get("role", "")), str(m.get("content", "") or ""))
                       for m in (messages or []) if isinstance(m, dict)]
        self._trim()

    # ---------- 内务 ----------
    def _trim(self) -> None:
        """按预算裁剪。

        `trim_messages` 保留的必然是本列表的一个**连续后缀**(它从尾部累积,
        再按需去掉开头的 assistant),所以直接按丢弃条数切片即可 ——
        不需要按内容重新配对,引擎标记自然跟着走。
        """
        if not self.history_max_chars or len(self._turns) <= 1:
            return
        slim = trim_messages([t.as_message() for t in self._turns], self.history_max_chars)
        dropped = len(self._turns) - len(slim)
        if dropped > 0:
            self._turns = self._turns[dropped:]

    def should_remember(self, every: int) -> bool:
        """是否该整理记忆:由本服务计数,因此 Agent 回合同样计入。"""
        return bool(every and every > 0 and self.turn_count
                    and self.turn_count % every == 0)

    def set_budget(self, history_max_chars: int) -> None:
        """更新字符预算并按新预算裁剪(参数变更时使用)。"""
        self.history_max_chars = int(history_max_chars or 0)
        self._trim()
