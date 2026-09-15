# -*- coding: utf-8 -*-
"""系统工具:启动应用/打开网址(逐次审批)、只读系统信息。"""
import os
import webbrowser

from agent.policy import hash_arguments
from agent.tools.base import ApprovalGrant, PreparedToolCall, ToolDefinition, ToolRegistry
from agent.types import PolicyDecision, ToolCall, ToolResult

MAX_PROCESS_ENTRIES = 30


def _default_open(target: str) -> None:
    """默认打开方式:http(s) 走默认浏览器;其余走系统默认关联程序。"""
    if target.lower().startswith(("http://", "https://")):
        webbrowser.open(target)
        return
    if hasattr(os, "startfile"):
        os.startfile(target)
        return
    raise RuntimeError("当前平台不支持 os.startfile")


def build_system_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="open_item",
        description="经主人批准后,用默认程序打开本地文件/文件夹,或用默认浏览器打开 http/https 网址。",
        parameters={
            "type": "object",
            "properties": {
                "path_or_url": {"type": "string"},
            },
            "required": ["path_or_url"],
            "additionalProperties": False,
        },
        mutating=True,
    ))
    registry.register(ToolDefinition(
        name="system_info",
        description="读取只读系统信息。resource 取值:processes(进程列表,最多 30 条)/cpu(占用率)/memory(内存)/disk(允许根所在磁盘)。",
        parameters={
            "type": "object",
            "properties": {
                "resource": {"type": "string", "enum": ["processes", "cpu", "memory", "disk"]},
            },
            "required": ["resource"],
            "additionalProperties": False,
        },
        mutating=False,
    ))
    return registry


class SystemToolExecutor:
    """与文件工具执行器同契约:prepare(策略评估)/ execute(审批校验后执行)。

    执行器本身不做阻塞式采样:system_info 的 cpu 项使用 psutil 的非阻塞语义
    (interval=None),因为 runner 的活动时间预算按真实耗时计算,半秒的阻塞采样
    会凭空吃掉预算。若需要更准确的 CPU 占用,请在独立工作线程中周期调用本工具。
    """

    def __init__(self, policy, opener=None, psutil_mod=None):
        self.policy = policy
        self.registry = build_system_registry()
        self.opener = opener or _default_open
        self._psutil_mod = psutil_mod
        self._cpu_warmed = False

    def _ps(self):
        if self._psutil_mod is not None:
            return self._psutil_mod
        try:
            import psutil
        except ImportError as e:
            raise RuntimeError("未安装 psutil,无法读取系统信息(pip install psutil)") from e
        return psutil

    def _cpu_percent(self, ps) -> float:
        """非阻塞 CPU 占用采样。

        psutil 的 interval=None 返回"自上次调用以来"的平均占用:首次调用总是 0.0,
        因此这里先预热一次建立基线,再取一次真实值。用 interval=0.5 采样会让
        调用线程阻塞半秒,既拖慢 UI 又吃掉 runner 的活动时间预算。
        """
        if not self._cpu_warmed:
            ps.cpu_percent(interval=None)   # 预热基线,返回值丢弃
            self._cpu_warmed = True
        return float(ps.cpu_percent(interval=None))

    def prepare(self, call: ToolCall) -> PreparedToolCall:
        definition, args = self.registry.prepare(call.name, call.arguments)
        assessment = self.policy.assess(ToolCall(call.id, call.name, args))
        canonical = assessment.canonical_arguments or args
        return PreparedToolCall(
            call=call,
            definition=definition,
            canonical_arguments=canonical,
            arguments_hash=hash_arguments(canonical),
            policy=assessment,
        )

    def execute(self, prepared: PreparedToolCall, approval=None) -> ToolResult:
        assessment = prepared.policy
        if assessment.decision == PolicyDecision.DENY:
            return ToolResult.denied(prepared.call.id, assessment.reason)
        if assessment.decision == PolicyDecision.ASK:
            if approval is None:
                return ToolResult(prepared.call.id, False, "APPROVAL_REQUIRED", "需要用户批准后执行")
            if not isinstance(approval, ApprovalGrant):
                return ToolResult(prepared.call.id, False, "INVALID_APPROVAL", "审批对象无效")
            try:
                approval.validate(prepared)
            except PermissionError as e:
                return ToolResult(prepared.call.id, False, "APPROVAL_MISMATCH", str(e))
        error = self._execution_recheck(prepared)
        if error is not None:
            return error
        try:
            return self._dispatch(prepared)
        except OSError as e:
            return ToolResult(prepared.call.id, False, "IO_ERROR", str(e))
        except RuntimeError as e:
            return ToolResult(prepared.call.id, False, "UNAVAILABLE", str(e))
        except Exception as e:
            # 白名单外异常必须在这里收敛(与 files.py 一致):webbrowser.open 在无可用
            # 浏览器时抛 webbrowser.Error(既不是 OSError 也不是 RuntimeError),
            # 穿透到 runner 的兜底就会把**整个 run** 变成 INTERNAL_ERROR,
            # 而它其实只是"这一个工具调用失败了"。
            return ToolResult(prepared.call.id, False, "TOOL_ERROR",
                              f"{type(e).__name__}: {e}")

    def _execution_recheck(self, prepared: PreparedToolCall):
        """执行瞬间复核:路径/决策/指纹任一变化即中止,防止批准后被替换。"""
        try:
            reassessment = self.policy.assess(
                ToolCall(prepared.call.id, prepared.call.name,
                         dict(prepared.canonical_arguments)))
        except Exception as e:
            return ToolResult(prepared.call.id, False, "PATH_CHANGED", f"路径复查失败: {e}")
        if reassessment.decision == PolicyDecision.DENY:
            return ToolResult(prepared.call.id, False, "PATH_CHANGED", reassessment.reason)
        if hash_arguments(reassessment.canonical_arguments or {}) != prepared.arguments_hash:
            return ToolResult(prepared.call.id, False, "PATH_CHANGED",
                              "目标在执行前被替换,操作已取消")
        if prepared.file_fingerprint is not None:
            from pathlib import Path as _P

            from agent.policy import fingerprint as _fp
            path = _P(prepared.canonical_arguments.get("path_or_url", ""))
            if path.is_symlink():
                return ToolResult(prepared.call.id, False, "PATH_CHANGED",
                                  "目标被替换为符号链接,操作已取消")
            if _fp(path) != prepared.file_fingerprint:
                return ToolResult(prepared.call.id, False, "FINGERPRINT_CHANGED",
                                  "文件在审批后被修改,操作已取消")
        return None

    def _dispatch(self, prepared: PreparedToolCall) -> ToolResult:
        call_id = prepared.call.id
        args = prepared.canonical_arguments
        name = prepared.call.name
        if name == "open_item":
            target = args["path_or_url"]
            self.opener(target)
            return ToolResult(call_id, True, "OK", f"已打开 {target}")
        if name == "system_info":
            return self._system_info(call_id, args["resource"])
        return ToolResult.denied(call_id, f"未实现工具: {name}")

    def _system_info(self, call_id: str, resource: str) -> ToolResult:
        ps = self._ps()
        try:
            if resource == "processes":
                procs = []
                for p in ps.process_iter(["pid", "name", "memory_info"]):
                    try:
                        info = p.info
                        mem_obj = info.get("memory_info")
                        rss = int(getattr(mem_obj, "rss", 0) or 0)
                        procs.append((info.get("pid", 0), info.get("name", "?"), rss))
                    except (ps.NoSuchProcess, ps.AccessDenied):
                        continue
                procs.sort(key=lambda t: -t[2])
                lines = [
                    f"{pid:>8}  {name}  ({mem / 1048576:.1f} MB)"
                    for pid, name, mem in procs[:MAX_PROCESS_ENTRIES]
                ]
                return ToolResult(call_id, True, "OK", "\n".join(lines) or "（无进程信息）",
                                  {"count": len(procs)})
            if resource == "cpu":
                percent = self._cpu_percent(ps)
                return ToolResult(call_id, True, "OK",
                                  f"CPU 占用 {percent:.1f}%，逻辑核心 {ps.cpu_count()}",
                                  {"percent": percent, "cores": ps.cpu_count()})
            if resource == "memory":
                vm = ps.virtual_memory()
                return ToolResult(call_id, True, "OK",
                                  f"内存 {vm.percent:.1f}%（已用 {vm.used / 2**30:.1f} / 共 {vm.total / 2**30:.1f} GiB）",
                                  {"percent": vm.percent, "used": vm.used, "total": vm.total})
            if resource == "disk":
                root = str(self.policy.settings.allowed_root)
                du = ps.disk_usage(root)
                return ToolResult(call_id, True, "OK",
                                  f"磁盘 {root} 占用 {du.percent:.1f}%"
                                  f"（可用 {du.free / 2**30:.1f} / 共 {du.total / 2**30:.1f} GiB）",
                                  {"percent": du.percent, "free": du.free, "total": du.total})
        except Exception as e:
            return ToolResult(call_id, False, "SYSINFO_ERROR", str(e))
        return ToolResult(call_id, False, "INVALID_RESOURCE", f"未知 resource: {resource}")


def build_system_executor(policy, opener=None, psutil_mod=None) -> SystemToolExecutor:
    return SystemToolExecutor(policy, opener=opener, psutil_mod=psutil_mod)
