# -*- coding: utf-8 -*-
"""记忆的查看 / 纠正 / 遗忘 / 置顶命令(纯函数,便于离线单独验证)。

用户在输入框里直接敲命令,UI 拦截后**不发模型请求**、只改本地记忆并写盘。
图形面板留待 UI 拆分时再做;命令入口能立刻满足"主人可查看/纠正/遗忘/置顶",
而且完全可离线单独验证。

命令:
    /记忆                       列出条目与序号
    /忘记 <序号或关键词>          让匹配的记忆失效(不物理删除,可回溯)
    /纠正 <序号或关键词> <新内容>   改写内容,原内容留在 supersedes
    /置顶 <序号或关键词>          置顶(预算紧张时仍会注入)
    /取消置顶 <序号或关键词>       取消置顶
    /记忆帮助                    用法说明

设计要点:
- **认不出就返回 None**,让 UI 按普通消息处理 —— 绝不能让一条正常聊天被命令层吞掉;
- 所有改动都返回给人看的中文说明,UI 直接展示;
- 不在这里做写盘:由调用方走 `save_memories`(保留既有的防误清空/原子写入/损坏备份保护)。
"""
from dataclasses import dataclass

from kurumi import memory as km

PREFIX = "/"
HELP_TEXT = (
    "记忆命令:\n"
    "  /记忆 —— 查看已记住的内容(带序号)\n"
    "  /忘记 <序号或关键词> —— 让某条记忆失效\n"
    "  /纠正 <序号或关键词> <新内容> —— 改写某条记忆\n"
    "  /置顶 <序号或关键词> —— 置顶(预算紧张时仍会注入)\n"
    "  /取消置顶 <序号或关键词> —— 取消置顶"
)

_ALIASES = {
    "记忆": "list", "记忆帮助": "help", "记忆列表": "list",
    "忘记": "forget", "忘掉": "forget", "遗忘": "forget",
    "纠正": "correct", "更正": "correct", "修改记忆": "correct",
    "置顶": "pin", "取消置顶": "unpin",
}
# 命令名按长度倒序匹配,避免「取消置顶」被「置顶」抢先命中
_ORDERED_NAMES = sorted(_ALIASES, key=len, reverse=True)


@dataclass(frozen=True)
class Command:
    name: str
    args: str = ""

    @property
    def changes_memory(self) -> bool:
        return self.name in ("forget", "correct", "pin", "unpin")


# 这些命令接受参数,因此也允许「/忘掉咖啡」这种中文习惯的无空格写法;
# 无参数命令(列表/帮助)后面若还跟着别的东西,就**不算命令**,
# 免得把「/记忆x」这类输入误吞。
_NAMES_WITH_ARGS = ("forget", "correct", "pin", "unpin")


def parse_command(text: str):
    """解析命令;不是命令(或不是已知命令)时返回 None。

    返回 None 都按普通消息处理:不是 `/` 开头、或以 `/` 开头但命令名未知
    (例如主人只是想发一句"/etc/hosts 是什么"),因此不会吞掉正常输入。
    """
    raw = str(text or "").strip()
    if not raw.startswith(PREFIX):
        return None
    body = raw[len(PREFIX):].strip()
    if not body:
        return None
    for name in _ORDERED_NAMES:
        if not body.startswith(name):
            continue
        action = _ALIASES[name]
        rest = body[len(name):].strip()
        if not rest:
            return Command(action)
        if action in _NAMES_WITH_ARGS:
            return Command(action, rest)
        # 无参数命令后面还有内容 → 不是这条命令,继续试其它名字
    return None


def run_command(command: Command, memories):
    """执行命令,返回 (给主人看的回复, 记忆是否被改动)。"""
    if command.name == "help":
        return HELP_TEXT, False
    if command.name == "list":
        return km.format_memory_list(memories), False
    if command.name == "forget":
        return _run_forget(command, memories)
    if command.name == "correct":
        return _run_correct(command, memories)
    if command.name in ("pin", "unpin"):
        return _run_pin(command, memories)
    return HELP_TEXT, False


def _run_forget(command, memories):
    if not command.args:
        return "要忘记哪一条?先说「/记忆」看看序号,例如「/忘记 2」。", False
    forgotten, count = km.forget_memory(memories, command.args)
    if not count:
        return f"没有找到「{command.args}」对应的记忆。可以先「/记忆」看看有哪些。", False
    joined = "、".join(f"「{c}」" for c in forgotten)
    return f"好的,已经忘掉 {joined}。(条目仍留在文件里,可随时纠正回来)", True


def _run_correct(command, memories):
    target, new_content = _split_target(command.args)
    if not target or not new_content:
        return "用法:「/纠正 <序号或关键词> <新内容>」,例如「/纠正 1 主人只喝乌龙茶」。", False
    replaced, text = km.correct_memory(memories, target, new_content)
    if not replaced:
        return f"没有找到「{target}」对应的记忆。可以先「/记忆」看看有哪些。", False
    joined = "、".join(f"「{c}」" for c in replaced)
    return f"记下了:{joined} → 「{text}」。", True


def _run_pin(command, memories):
    pinned = command.name == "pin"
    if not command.args:
        verb = "置顶" if pinned else "取消置顶"
        return f"要给哪一条{verb}?先说「/记忆」看看序号。", False
    hit = km.set_pinned(memories, command.args, pinned)
    if not hit:
        return f"没有找到「{command.args}」对应的记忆。可以先「/记忆」看看有哪些。", False
    joined = "、".join(f"「{c}」" for c in hit)
    word = "置顶" if pinned else "取消置顶"
    return f"好的,已{word}:{joined}。", True


def _split_target(args: str):
    """把「<目标> <新内容>」拆开:目标是第一个空白前的词。"""
    text = str(args or "").strip()
    if not text:
        return "", ""
    for index, ch in enumerate(text):
        if ch.isspace():
            return text[:index].strip(), text[index:].strip()
    return text, ""


def handle_text(text, memories):
    """一站式入口:是记忆命令就执行,否则返回 None(调用方按普通消息处理)。

    返回 (回复文本, 是否改动了记忆) 或 None。
    """
    command = parse_command(text)
    if command is None:
        return None
    reply, changed = run_command(command, memories)
    return reply, changed
