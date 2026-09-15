# -*- coding: utf-8 -*-
"""AgentController:UI 与 AgentRunner 之间的 Qt 桥。

GUI 线程只发出命令、渲染不可变事件;runner 在独立线程运行。
"""
import threading
import uuid

from PySide6.QtCore import QObject, Signal

from agent.model.deepseek import CapabilityItem, CapabilityReport
from agent.types import AgentEvent, ApprovalDecision, RunOutcome, RunState
from runtime_control import CancellationToken


def _unavailable_report(reason: str) -> CapabilityReport:
    """探测过程本身抛异常时的报告:一律标成未验证,不能据此断言"端点不支持"。"""
    return CapabilityReport(
        model_call=CapabilityItem("model_call", False, verified=False, evidence=reason),
        function_calls=CapabilityItem("function_calls", False, verified=False, evidence=reason),
        web_search=CapabilityItem("web_search", False, verified=False, evidence=reason),
        streaming=CapabilityItem("streaming", False, verified=False, evidence="未验证"),
        retryable=True,
    )


class AgentController(QObject):
    event_received = Signal(object)      # AgentEvent
    approval_requested = Signal(object)  # AgentEvent(kind=approval_requested)
    run_finished = Signal(object)        # RunOutcome
    # 能力报告**必须带上代次**:信号是跨线程排队的,"发出"与"被 UI 处理"之间
    # 主人完全可能已经做了选择 —— 只在控制器里"发送前核对代次"还不够:报告已经写进
    # `_capable`、信号已经排进事件循环时,UI 照样会把它当成最新结论。
    capability_reported = Signal(object, int)  # (CapabilityReport, 探测代次)

    def __init__(self, runner_factory, parent=None):
        super().__init__(parent)
        self._factory = runner_factory
        self._runner = None
        self._thread = None
        self._cancellation = None
        self._active_run_id = None
        self._capable = False
        self._capability_reason = ""
        self._capability_report = None
        # 探测的"代次"。迟到的旧探测绝不允许改写能力状态 —— 主人可能已经明确
        # 选了"用普通聊天",一条几秒后才回来的旧结论不能把它推翻。
        self._probe_lock = threading.Lock()
        self._probe_generation = 0

    # ---------- 能力探测 ----------
    def probe(self) -> None:
        """发起一次能力探测;写结论前核对代次,过期结果一律丢弃。

        为什么必须在**控制器**里丢,而不是让 UI 忽略回调:`_capable` 是控制器自己改的
        (见下),UI 就算不理会回调,控制器内部也已经把 Agent 当成可用了 ——
        之后任何一次 `start()` 都会照跑。主人选完普通聊天后,迟到的探测会把
        `agent_ready` 从 False 翻回 True。
        """
        with self._probe_lock:
            self._probe_generation += 1
            generation = self._probe_generation
        try:
            report = self._factory.probe_capabilities()
        except Exception as e:
            report = _unavailable_report(f"能力探测异常: {e}")
        with self._probe_lock:
            if generation != self._probe_generation:
                return                    # 期间被作废或已重试:这份结论不再采用
            self._capability_report = report
            self._capable = report.agent_usable
            self._capability_reason = report.reason
        # 带上代次:UI 收到时会再核对一次(见 discard_pending_probe 的说明)
        self.capability_reported.emit(report, generation)

    def discard_pending_probe(self) -> None:
        """作废探测结论(在飞的 + 已经写进状态但 UI 还没处理的)。

        调用时机:主人选择"用普通聊天"、兜底计时器放弃等待、Agent 被判定不可用。
        做两件事:

        1. 代次 +1 —— 还挂在网络上的探测回来时会被丢弃,已排队但没送达的信号也会被
           UI 按代次拒收(信号带 generation,见 `capability_reported`);
        2. **清掉尚未被接受的能力状态** —— 探测可能已经跑完并写好了 `_capable=True`、
           信号还在队列里;只加代次的话,控制器自己仍然认为 Agent 可用 ——
           `discard` 之后 `capable` 依旧是 True。
        """
        with self._probe_lock:
            self._probe_generation += 1
            self._capable = False
            self._capability_report = None
            self._capability_reason = ""

    @property
    def probe_generation(self) -> int:
        with self._probe_lock:
            return self._probe_generation

    @property
    def capable(self) -> bool:
        """Agent 是否可用:只看模型调用(本地工具不依赖联网搜索)。"""
        return self._capable

    @property
    def capability_report(self):
        return self._capability_report

    # ---------- 运行控制 ----------
    def start(self, request: str, context=None) -> None:
        """开始一次运行。

        `context`(可选)是结构化的本次上下文(RunContext):以往对话按角色发送、
        本次补充材料追加到 instructions。为 None 时退化为"单个请求字符串"的调用方式。
        """
        if not self._capable:
            return
        self._active_run_id = uuid.uuid4().hex
        self._cancellation = CancellationToken()
        try:
            self._runner = self._factory.create(event_sink=self._forward_event)
        except Exception as e:
            self.run_finished.emit(RunOutcome(state=RunState.FAILED, code="INTERNAL_ERROR", message=str(e)))
            return
        self._thread = threading.Thread(
            target=self._run_in_thread, args=(self._active_run_id, request, context), daemon=True
        )
        self._thread.start()

    def _run_in_thread(self, run_id: str, request: str, context=None) -> None:
        try:
            outcome = self._runner.run(request, run_id, self._cancellation, context=context)
        except Exception as e:
            import traceback
            tb_lines = traceback.format_exc().strip().splitlines()
            frame = next((ln.strip() for ln in reversed(tb_lines) if ln.strip().startswith('File "')), "")
            detail = f"{type(e).__name__}: {e}".strip()
            if frame:
                detail += f"（{frame}）"
            outcome = RunOutcome(state=RunState.FAILED, code="INTERNAL_ERROR", message=detail)
        if run_id == self._active_run_id:
            self.run_finished.emit(outcome)

    def approve_once(self, approval_id: str) -> None:
        if self._runner is not None:
            self._runner.resolve_approval(ApprovalDecision.allow_once(approval_id))

    def deny(self, approval_id: str) -> None:
        if self._runner is not None:
            self._runner.resolve_approval(ApprovalDecision.deny(approval_id))

    def cancel(self) -> None:
        if self._cancellation is not None:
            self._cancellation.cancel()
        if self._runner is not None:
            self._runner.cancel()

    def shutdown(self, timeout_ms: int = 2000) -> bool:
        self.cancel()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout_ms / 1000.0)
        return thread is None or not thread.is_alive()

    # ---------- 事件转发(按 run_id 丢弃迟到事件) ----------
    def _forward_event(self, event) -> None:
        run_id = None
        if isinstance(event, dict):
            run_id = event.get("run_id")
        else:
            run_id = getattr(event, "run_id", None)
        if run_id != self._active_run_id:
            return
        if isinstance(event, dict):
            event = AgentEvent(
                run_id=run_id,
                sequence=int(event.get("sequence", 0)),
                kind=str(event.get("kind", "")),
                message=str(event.get("message", "")),
                payload=event.get("payload") or {},
            )
        self.event_received.emit(event)
        if event.kind == "approval_requested":
            self.approval_requested.emit(event)
