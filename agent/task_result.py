# -*- coding: utf-8 -*-
"""结构化任务结果:让 Agent 能**据实**说明"完成了什么"。

背景（用户点名的验收）:
- "文件写入成功后模拟断网,再问「文件生成了吗」,能够准确回答并指出路径" ——
  这要求产物路径是**结构化记录**，而不是让模型从自由文本里回忆；
- "任务中断后保留已发生的动作" —— 失败/取消时仍要能说清"哪些已经做了"。

因此这里不做任何模型调用，只把 runner 已经知道的事实整理成可核对的记录。
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskAction:
    """一次工具调用的可核对记录。"""

    tool: str
    code: str
    ok: bool = False
    targets: tuple = ()          # 涉及的路径/URL（写入类工具的真实产物）
    summary: str = ""

    def describe(self) -> str:
        mark = "成功" if self.ok else "未成功"
        target = ("：" + "、".join(self.targets)) if self.targets else ""
        return f"{self.tool} {mark}（{self.code}）{target}"


@dataclass(frozen=True)
class RunStats:
    """一次运行的统一统计（请求/重试/token/时间）。

    token 字段为 **-1 表示端点未给用量**(未知),而不是 0 —— 把"没有数据"显示成 0
    会让主人以为这次没有消耗。只有真正取到 usage 时才显示数字。
    """

    model_requests: int = 0
    model_retries: int = 0
    tool_calls: int = 0
    prompt_tokens: int = -1
    completion_tokens: int = -1
    active_seconds: float = 0.0
    active_limit_seconds: float = 0.0

    @property
    def usage_known(self) -> bool:
        """是否拿到了真实用量(任一侧有数据即算已知)。"""
        return int(self.prompt_tokens) >= 0 or int(self.completion_tokens) >= 0

    @property
    def total_tokens(self) -> int:
        return max(0, int(self.prompt_tokens)) + max(0, int(self.completion_tokens))

    @property
    def remaining_seconds(self) -> float:
        """活动预算还剩多少秒（不为负）。"""
        return max(0.0, float(self.active_limit_seconds) - float(self.active_seconds))

    def describe(self) -> str:
        tokens = f"token {self.total_tokens}" if self.usage_known else "token 未知（端点未返回用量）"
        return (f"模型请求 {self.model_requests} 次（重试 {self.model_retries}）、"
                f"工具调用 {self.tool_calls} 次、{tokens}、"
                f"活动 {self.active_seconds:.1f}s / 上限 {self.active_limit_seconds:.0f}s")


@dataclass(frozen=True)
class TaskResult:
    """一次任务的完整结论：状态 + 已完成动作 + 产物 + 来源 + 未完成事项 + 统计。"""

    state: str = ""
    code: str = ""
    message: str = ""
    actions: tuple = ()          # TaskAction
    artifacts: tuple = ()        # 产物路径（来自成功的写入类动作）
    sources: tuple = ()          # 来源 URL（联网搜索）
    unfinished: tuple = ()       # 未完成事项（人可读）
    stats: RunStats = field(default_factory=RunStats)

    @property
    def completed(self) -> bool:
        return str(self.state) == "completed"

    @property
    def succeeded_actions(self) -> tuple:
        return tuple(a for a in self.actions if a.ok)

    def facts_for_reply(self) -> list:
        """给模型看的事实清单（人设语气由模型自己把握，事实必须来自这里）。"""
        lines = []
        if self.artifacts:
            lines.append("实际生成/修改的文件：" + "、".join(self.artifacts))
        for action in self.actions:
            lines.append("工具记录：" + action.describe())
        if self.sources:
            lines.append("参考来源：" + "、".join(self.sources))
        if self.unfinished:
            lines.append("未完成：" + "、".join(self.unfinished))
        return lines

    def describe(self) -> str:
        head = f"状态 {self.state}（{self.code}）"
        if self.message:
            head += f"：{self.message}"
        parts = [head]
        if self.artifacts:
            parts.append("产物：" + "、".join(self.artifacts))
        if self.sources:
            parts.append("来源：" + "、".join(self.sources))
        parts.append(self.stats.describe())
        return "；".join(parts)


# 这些工具成功时，其目标路径就是"产物"
_ARTIFACT_TOOLS = ("file_write", "file_patch")
# 这些工具涉及来源 URL
_SOURCE_TOOLS = ("web_search", "web_fetch")


def build_task_result(state, code, message, actions=(), sources=(), unfinished=(),
                      stats=None) -> TaskResult:
    """从 runner 记录的事实组装 TaskResult（纯函数，便于离线单独验证）。

    产物只取**成功**的写入类动作的目标；失败动作会出现在 actions 里但不进 artifacts ——
    这样模型说"文件已生成"时，一定真的有对应记录。
    """
    actions = tuple(actions)
    artifacts = []
    for action in actions:
        if action.ok and action.tool in _ARTIFACT_TOOLS:
            for target in action.targets:
                if target and target not in artifacts:
                    artifacts.append(target)
    collected_sources = []
    for url in sources or ():
        if url and url not in collected_sources:
            collected_sources.append(url)
    for action in actions:
        if action.tool in _SOURCE_TOOLS:
            for target in action.targets:
                if target and target not in collected_sources:
                    collected_sources.append(target)
    return TaskResult(
        state=str(getattr(state, "value", state) or ""),
        code=str(code or ""),
        message=str(message or ""),
        actions=actions,
        artifacts=tuple(artifacts),
        sources=tuple(collected_sources),
        unfinished=tuple(unfinished or ()),
        stats=stats or RunStats(),
    )


# 这些工具成功即视为"已有落盘副作用":失败/取消后必须保留记录,
# 否则下一轮会以为文件没被创建,甚至重复创建。
_SIDE_EFFECT_TOOLS = ("file_write", "file_patch")

# 状态码 → 人类可读的中文标签(界面与日志共用,避免各处各写一套)
_STATE_LABELS = {
    "completed": "任务完成",
    "failed": "任务未完成",
    "cancelled": "任务已取消",
    "awaiting_approval": "等待主人批准",
}


def state_label(result) -> str:
    state = str(getattr(result, "state", "") or "")
    return _STATE_LABELS.get(state, state or "未知状态")


def has_side_effects(result) -> bool:
    """本次运行是否已经产生**落盘副作用**(成功的写入类动作)。

    这是"失败后要不要保留历史"的判据:纯读失败不该污染上下文,
    但一旦有文件被真正写入,历史里就必须留下痕迹。
    """
    if result is None:
        return False
    for action in getattr(result, "actions", ()) or ():
        if getattr(action, "ok", False) and getattr(action, "tool", "") in _SIDE_EFFECT_TOOLS:
            return True
    return False


def task_panel_text(result) -> str:
    """把 TaskResult 渲染成任务面板文本(纯函数,界面与日志共用)。

    这是**程序如实生成的事实**,与模型在气泡里说的话分开:人设语气由模型把握,
    但"哪个文件被写了、失败码是什么"不允许被措辞改写。
    result 为 None(旧调用方/未接线)时返回空串,界面据此不渲染面板。
    """
    if result is None:
        return ""
    lines = [f"【{state_label(result)}】{result.code}"]
    if result.message:
        lines.append(result.message.strip())
    if result.artifacts:
        lines.append("产物：" + "、".join(result.artifacts))
    if result.sources:
        lines.append("来源：" + "、".join(result.sources))
    done = [a.describe() for a in result.actions if a.ok]
    if done:
        lines.append("已完成：" + "；".join(done))
    failed = [a.describe() for a in result.actions if not a.ok]
    if failed:
        lines.append("未成功：" + "；".join(failed))
    if result.unfinished:
        lines.append("未完成：" + "、".join(result.unfinished))
    lines.append(result.stats.describe())
    return "\n".join(lines)


def history_note(result) -> str:
    """失败/取消且**已有副作用**时写进会话历史的事实记录。

    刻意不用人设语气:这条记录是给"下一轮"看的事实依据,不是替模型说话。
    """
    facts = result.facts_for_reply()
    head = f"[{state_label(result)}：{result.code}]"
    return head + ("\n" + "\n".join(facts) if facts else "")


# ==================== 序列化（落库 / 重启后恢复）====================
#
# 只做"纯数据 ↔ 纯 dict",不含 JSON、不含 IO —— 便于单独验证,也让存储层
# 只负责"把 dict 写成 JSON 字符串"这一件事。

def stats_to_dict(stats) -> dict:
    stats = stats or RunStats()
    return {
        "model_requests": int(stats.model_requests),
        "model_retries": int(stats.model_retries),
        "tool_calls": int(stats.tool_calls),
        "prompt_tokens": int(stats.prompt_tokens),
        "completion_tokens": int(stats.completion_tokens),
        "active_seconds": float(stats.active_seconds),
        "active_limit_seconds": float(stats.active_limit_seconds),
    }


def stats_from_dict(data) -> RunStats:
    """从落库的 dict 还原统计;**缺字段/类型不对时退回默认值**,绝不抛异常。

    落库数据可能来自老版本或被人手工改过,读取方不该因为一个坏字段而崩掉整个恢复流程
    (恢复失败会导致界面像"从没跑过任务"一样,比少一个统计数字严重得多)。
    """
    if not isinstance(data, dict):
        return RunStats()

    def number(key, default, cast):
        try:
            return cast(data.get(key, default))
        except (TypeError, ValueError):
            return default

    return RunStats(
        model_requests=number("model_requests", 0, int),
        model_retries=number("model_retries", 0, int),
        tool_calls=number("tool_calls", 0, int),
        prompt_tokens=number("prompt_tokens", -1, int),
        completion_tokens=number("completion_tokens", -1, int),
        active_seconds=number("active_seconds", 0.0, float),
        active_limit_seconds=number("active_limit_seconds", 0.0, float),
    )


def action_to_dict(action) -> dict:
    return {
        "tool": str(action.tool),
        "code": str(action.code),
        "ok": bool(action.ok),
        "targets": [str(t) for t in (action.targets or ())],
        "summary": str(action.summary or ""),
    }


def action_from_dict(data) -> "TaskAction | None":
    if not isinstance(data, dict):
        return None
    targets = data.get("targets")
    if not isinstance(targets, (list, tuple)):
        targets = ()
    return TaskAction(
        tool=str(data.get("tool", "") or ""),
        code=str(data.get("code", "") or ""),
        ok=bool(data.get("ok", False)),
        targets=tuple(str(t) for t in targets),
        summary=str(data.get("summary", "") or ""),
    )


def to_dict(result) -> dict:
    """TaskResult → 纯 dict(可直接 json.dumps)。"""
    return {
        "state": str(result.state or ""),
        "code": str(result.code or ""),
        "message": str(result.message or ""),
        "actions": [action_to_dict(a) for a in (result.actions or ())],
        "artifacts": [str(a) for a in (result.artifacts or ())],
        "sources": [str(s) for s in (result.sources or ())],
        "unfinished": [str(u) for u in (result.unfinished or ())],
        "stats": stats_to_dict(result.stats),
    }


def from_dict(data):
    """dict → TaskResult;结构不对时返回 None(调用方按"没有结构化结果"处理)。"""
    if not isinstance(data, dict):
        return None
    actions = tuple(a for a in (action_from_dict(x) for x in (data.get("actions") or [])) if a)

    def strings(key):
        value = data.get(key)
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(v) for v in value if str(v))

    return TaskResult(
        state=str(data.get("state", "") or ""),
        code=str(data.get("code", "") or ""),
        message=str(data.get("message", "") or ""),
        actions=actions,
        artifacts=strings("artifacts"),
        sources=strings("sources"),
        unfinished=strings("unfinished"),
        stats=stats_from_dict(data.get("stats")),
    )
