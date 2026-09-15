# -*- coding: utf-8 -*-
"""有界 Agent 执行循环:模型↔工具往返、一次性审批、预算与取消。

同步实现,由 UI 层在独立线程中运行。等待审批使用 Condition 分片等待,
保证取消与超时能被及时观察到。
"""
import json
import threading
import time as _time

from agent.audit import mask_known_secrets, redact
from agent.model.deepseek import AgentCancelled
from agent.task_result import RunStats, TaskAction, build_task_result
from agent.tools.base import ApprovalGrant
from agent.types import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequest,
    PolicyDecision,
    RunOutcome,
    RunState,
    ToolCall,
    ToolResult,
)

_MAX_TOOL_RESULT_CHARS = 16000
_MAX_TOOL_METADATA_CHARS = 500

# 瞬态模型错误重试的固定退避(sleeper 可注入,因此调试时可以跳过真实等待)
_RETRY_DELAY_SECONDS = 0.1

# _wait_approval 的返回值哨兵:区分"被取消"(None)与"等待审批超时"
_APPROVAL_TIMEOUT = object()


class AgentRunner:
    def __init__(self, model, tools, store, settings, instructions="",
                 clock=_time.monotonic, event_sink=None, sleeper=_time.sleep):
        self.model = model
        self.tools = tools
        self.store = store
        self.settings = settings
        self.instructions = instructions
        self.clock = clock
        self.event_sink = event_sink or (lambda event: None)
        self.sleeper = sleeper
        self._condition = threading.Condition()
        self._decisions: dict[str, ApprovalDecision] = {}
        self._pending: dict[str, ApprovalRequest] = {}
        self._cancelled = threading.Event()
        self._seq = 0
        self._run_id = ""
        self._audit_failures = 0
        self._model_retries = 0
        self._usage_sink = None      # 由 run() 绑定,供重试路径累加用量

    # ---------- 对外接口 ----------
    def run(self, request: str, run_id: str, cancellation=None, context=None) -> RunOutcome:
        """执行一次任务。

        `request` 仍是要执行的任务文本(审计与旧调用方使用);`context` 可选,
        提供**结构化**的之前对话与本次补充材料:
        - `context.input_items`:直接作为 Responses 的 input 项(保留 user/assistant 角色),
          取代"把历史压成一条 user 输入"的做法;
        - `context.instructions_suffix`:本次特有的补充材料(如知识库设定参考),
          追加在构造时给定的 instructions 之后 —— 静态指令不被覆盖。
        """
        self._run_id = run_id
        self._seq = 0
        self._cancelled.clear()
        self._invalid_streak = 0
        self._audit_failures = 0
        self._model_retries = 0   # 重试预算按 run 计,防止 轮数×重试 放大付费请求
        self._model_requests = 0  # 统一统计用的请求计数
        # 把"这次运行属于哪一段会话"一并记下 —— 界面恢复"上次 Agent 结果"时按会话查,
        # 于是清空会话之后旧任务不会跨边界复活(按时间戳比先后在同秒/时钟回拨时并不可靠)。
        self._audit("start_run_id", run_id, request,
                    str(getattr(context, "session_id", "") or ""))
        if context is not None and context.input_items:
            input_items: list = [dict(item) for item in context.input_items]
            run_instructions = self._merge_instructions(context.instructions_suffix)
        else:
            input_items = [{"role": "user", "content": request}]
            run_instructions = self.instructions
        tool_count = 0
        rounds = 0
        last_sig = None
        active = 0.0
        outcome: RunOutcome | None = None
        # 记录"可核对的事实",供 TaskResult 组装
        actions: list = []
        self._actions = actions      # _record_action 追加到同一列表
        run_sources: list = []
        prompt_tokens = 0
        completion_tokens = 0
        usage_seen = False     # 端点是否真的给过用量:没给就不显示 0(见 finish)

        def record_usage(turn) -> None:
            """累加一次模型调用的用量。拿不到就保持"未知",绝不猜测。"""
            nonlocal prompt_tokens, completion_tokens, usage_seen
            prompt = getattr(turn, "prompt_tokens", None)
            completion = getattr(turn, "completion_tokens", None)
            if prompt is None and completion is None:
                return
            usage_seen = True
            if prompt is not None:
                prompt_tokens += int(prompt)
            if completion is not None:
                completion_tokens += int(completion)

        # 重试路径(_retry_model)也要能累加用量,故挂在实例上;每次 run 重新绑定,
        # 避免上一轮 run 的闭包残留(否则用量会跨任务串账)。
        self._usage_sink = record_usage

        def finish(state, code, message):
            """组装终态结果并落库。

            审计写库失败只追加一句提示,绝不抛出:否则调用方拿到的是异常而非
            RunOutcome,已经落盘的副作用(文件已被修改)会被误报成 INTERNAL_ERROR。
            """
            if not self._audit("set_run_state", run_id, state, code, message):
                message = f"{message}（审计写入失败 {self._audit_failures} 次,本次运行状态未落库）"
            # 附带可核对的结构化结果(失败/取消也带上**已完成**的动作,
            # 满足"任务中断后保留已发生的动作")。
            # 端点没给 usage 时用 -1 表示"未知",而不是 0 —— 把未知写成 0
            # 会让主人以为这次没有消耗。
            stats = RunStats(
                model_requests=self._model_requests,
                model_retries=self._model_retries,
                tool_calls=tool_count,
                prompt_tokens=prompt_tokens if usage_seen else -1,
                completion_tokens=completion_tokens if usage_seen else -1,
                active_seconds=round(active, 3),
                active_limit_seconds=float(self.settings.active_timeout_seconds or 0),
            )
            task = build_task_result(state, code, message, actions=actions,
                                     sources=run_sources, stats=stats)
            # 结构化结果落库,供**重启后**回答"上次到底做成了什么"。
            # 走 _audit 以沿用"审计失败不影响任务"的既有约定。
            self._audit("set_task_result", run_id, task)
            return RunOutcome(state=state, code=code, message=message, task_result=task)

        try:
            while rounds < self.settings.max_model_rounds:
                if self._is_cancelled(cancellation):
                    outcome = finish(RunState.CANCELLED, "CANCELLED", "任务已取消")
                    break
                rounds += 1
                # 用**剩余活动时间**约束这一次请求的超时。
                #
                # 注意不能"剩余不足就不发起":请求超时是**上界**而非预期耗时,真实请求往往
                # 几百毫秒就返回。用"剩余 < 超时"作门槛会把正常配置直接判死
                # (active_timeout 很小时,第一次调用前就会被拒绝)。
                # 正确做法是把这一次请求的超时压到剩余时间内,让它有机会按时返回;
                # 真的超过剩余时间,后面的活动时限检查照样会按 ACTIVE_TIMEOUT 收尾。
                if self.settings.active_timeout_seconds:
                    remaining_budget = float(self.settings.active_timeout_seconds) - active
                    if remaining_budget <= 0:
                        outcome = finish(RunState.FAILED, "ACTIVE_TIMEOUT",
                                         "活动时间预算已用尽，未再发起模型请求")
                        break
                    if hasattr(self.model, "timeout"):
                        self.model.timeout = max(1.0, min(self._request_timeout_seconds(),
                                                         remaining_budget))
                self._model_requests += 1
                start = self.clock()
                try:
                    turn = self.model.complete(
                        input_items, self.tools_registry_definitions(), self._sink, cancellation,
                        instructions=run_instructions,
                    )
                except AgentCancelled:
                    # 取消与模型调用的竞态:模型端在发出请求前发现已取消会抛这个。
                    # 若不单独接住,会被下面的兜底当成 INTERNAL_ERROR —— 用户动作是
                    # "取消",报告却是"内部错误",UI 还会走失败回滚逻辑。
                    active += self.clock() - start
                    outcome = finish(RunState.CANCELLED, "CANCELLED", "任务已取消")
                    break
                record_usage(turn)          # 用量随每次真实调用累加
                active += self.clock() - start
                if active > self.settings.active_timeout_seconds:
                    outcome = finish(RunState.FAILED, "ACTIVE_TIMEOUT", "活动执行超过时限")
                    break

                # 协议错误处理(重试无需回传本回合原始项)
                for pe in turn.protocol_errors:
                    if pe.code == "MODEL_TRANSIENT":
                        start = self.clock()
                        transient = self._retry_model(
                            input_items, cancellation, instructions=run_instructions, attempts=2)
                        active += self.clock() - start
                        if transient is not None:
                            # 这一发的用量已由 _retry_model 通过 _usage_sink 计入,
                            # 这里**不能**再记一次(会双重计费)。
                            turn = transient
                            # 重试后的新回合必须重新检查 MODEL_FATAL:若在此直接 break,
                            # 新回合的 MODEL_FATAL 不会被检查 → 落到下面"模型未产生内容"(EMPTY_TURN),
                            # 真实原因(如 invalid_request: bad schema)被吞掉。
                            fatal = next((p for p in turn.protocol_errors
                                          if p.code == "MODEL_FATAL"), None)
                            if fatal is not None:
                                outcome = finish(RunState.FAILED, "MODEL_FATAL", fatal.message)
                            break
                        if self._is_cancelled(cancellation):
                            # 取消发生在两次重试之间:应报 CANCELLED,而不是伪装成模型失败
                            outcome = finish(RunState.CANCELLED, "CANCELLED", "任务已取消")
                            break
                        outcome = finish(RunState.FAILED, "MODEL_FATAL", "模型请求多次失败（重试预算已耗尽）")
                        break
                    if pe.code == "MODEL_FATAL":
                        outcome = finish(RunState.FAILED, "MODEL_FATAL", pe.message)
                        break
                if outcome is not None:
                    break
                if active > self.settings.active_timeout_seconds:
                    outcome = finish(RunState.FAILED, "ACTIVE_TIMEOUT", "活动执行超过时限")
                    break

                # 内置网页搜索:计预算,无本地执行;来源写入审计与事件
                for w in turn.web_actions:
                    if tool_count + 1 > self.settings.max_tool_calls:
                        outcome = finish(RunState.FAILED, "TOOL_LIMIT", "工具调用次数超限")
                        break
                    tool_count += 1
                    self._emit("web_action", "正在执行联网搜索", {"action": "web_search"})
                    for url in (w.get("urls") or []) if isinstance(w, dict) else []:
                        self._emit("source", url, {"url": url, "title": ""})
                        self._audit("record_source", run_id, {"url": url, "title": ""})
                        self._collect_source(run_sources, url)
                # 最终文本 annotation 中的引用来源(url_citation)
                for c in turn.citations:
                    if not isinstance(c, dict):
                        continue
                    url = c.get("url")
                    if url:
                        self._emit("source", url, {"url": url, "title": c.get("title", "")})
                        self._audit("record_source", run_id, {"url": url, "title": c.get("title", "")})
                        self._collect_source(run_sources, url)
                if outcome is not None:
                    break

                # 无函数调用 → 最终答复(含网页动作的回合也可能同时给出最终文本)
                if not turn.function_calls:
                    if turn.text.strip():
                        self._emit("final_text", turn.text, {"text": turn.text})
                        outcome = finish(RunState.COMPLETED, "OK", turn.text)
                        break
                    if not turn.web_actions:
                        # 推理吃光输出预算时模型端给出 OUTPUT_TRUNCATED:把它带回结果,
                        # 否则主人只看到"模型未产生内容",无从知道该调大 max_output_tokens
                        detail = next((pe for pe in turn.protocol_errors
                                       if pe.code == "OUTPUT_TRUNCATED"), None)
                        message = "模型未产生内容"
                        if detail is not None:
                            message = f"模型未产生内容（{detail.code}: {detail.message}）"
                        outcome = finish(RunState.FAILED, "EMPTY_TURN", message)
                        break
                    # 仅网页动作:必须先回传本回合原始输出项(web_search_call/reasoning 等),
                    # 否则检索结果不进上下文,模型会重复检索并烧掉轮次预算
                    input_items.extend(turn.output_items)
                    continue   # 仅网页动作:进入下一轮

                # 执行本轮所有函数调用,收集结果
                if turn.text.strip():
                    # 人设说明文字(如"狂三准备打开哔哩哔哩")与工具调用同回合出现:先转发文字,再走工具
                    self._emit("persona_text", turn.text, {"text": turn.text})
                outputs: dict = {}
                for call in turn.function_calls:
                    if tool_count + 1 > self.settings.max_tool_calls:
                        outcome = finish(RunState.FAILED, "TOOL_LIMIT", "工具调用次数超限")
                        break
                    tool_count += 1
                    if self._is_cancelled(cancellation):
                        outcome = finish(RunState.CANCELLED, "CANCELLED", "任务已取消")
                        break

                    result, stop, sig, elapsed = self._handle_call(call, run_id, cancellation, last_sig)
                    active += elapsed   # 审批等待不计入活动时间
                    if active > self.settings.active_timeout_seconds:
                        outcome = finish(RunState.FAILED, "ACTIVE_TIMEOUT", "活动执行超过时限")
                        break
                    if sig is not None:
                        last_sig = sig
                    if result is not None:
                        outputs[call.id] = {
                            "type": "function_call_output",
                            "call_id": call.id,
                            "output": self._format_tool_output(result),
                        }
                    if stop:
                        outcome = finish(stop[0], stop[1], stop[2])
                        break
                if outcome is not None:
                    break

                # 按原始顺序回传:function_call 原始项后紧跟其输出(Responses API 要求成对)
                for raw in turn.output_items:
                    input_items.append(raw)
                    if isinstance(raw, dict):
                        rid = raw.get("call_id") or raw.get("id")
                        rtype = raw.get("type")
                    else:
                        rid = getattr(raw, "call_id", None) or getattr(raw, "id", None)
                        rtype = getattr(raw, "type", None)
                    if rtype == "function_call" and rid in outputs:
                        input_items.append(outputs.pop(rid))
                for out in outputs.values():   # 容错:未配对到原始项的输出放最后
                    input_items.append(out)
        except Exception as e:
            import traceback
            tb_lines = traceback.format_exc().strip().splitlines()
            frame = next((ln.strip() for ln in reversed(tb_lines) if ln.strip().startswith('File "')), "")
            detail = f"{type(e).__name__}: {e}".strip()
            if frame:
                detail += f"（{frame}）"
            outcome = finish(RunState.FAILED, "INTERNAL_ERROR", detail)
        if outcome is None:
            outcome = finish(RunState.FAILED, "MODEL_ROUND_LIMIT", "模型往返轮次超限")
        return outcome

    def resolve_approval(self, decision: ApprovalDecision) -> None:
        """记录主人的审批决定。

        即使该 approval_id 还没进入 _pending(审批卡刚弹出、runner 尚未走到等待点),
        也必须收下——若把它静默丢掉,任务会永久停在等待里。
        """
        with self._condition:
            self._decisions[decision.approval_id] = decision
            while len(self._decisions) > 64:   # 未被消费的过期决定:有界淘汰
                self._decisions.pop(next(iter(self._decisions)))
            self._condition.notify_all()

    def cancel(self) -> None:
        """取消本次运行。

        除了置令牌,**还要通知模型端** —— 否则在途的那次 HTTP 请求
        会一直等到自己的超时(最长 45 秒),主人的「停止」要等十几秒才有反应。
        `DeepSeekAgentClient.cancel()` 会把在途请求的超时压小让它尽快结束,
        并把由此产生的失败改判成取消(而不是模型故障)。

        模型端没有 cancel(如注入的假客户端)时静默跳过:取消令牌仍是有效兜底。
        """
        self._cancelled.set()
        cancel = getattr(self.model, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception as e:
                print(f"[agent] 通知模型端取消失败(已忽略,令牌仍生效): {e}")
        with self._condition:
            self._condition.notify_all()

    # ---------- 内部 ----------
    def _request_timeout_seconds(self) -> float:
        """模型端单次请求的超时(读不到就用一个保守默认值)。"""
        value = getattr(self.model, "timeout", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 45.0
        return value if value > 0 else 45.0
    def _merge_instructions(self, suffix: str) -> str:
        """把本次补充材料追加到静态 instructions 之后(不覆盖静态部分)。"""
        if not suffix or not suffix.strip():
            return self.instructions
        if not self.instructions:
            return suffix.strip()
        return self.instructions.rstrip() + "\n" + suffix.strip()

    def _is_cancelled(self, token) -> bool:
        if self._cancelled.is_set():
            return True
        return bool(token is not None and getattr(token, "cancelled", False))

    def _sink(self, event):
        """把模型端的事件转成 AgentEvent。

        **文本可能出现在 `message` 或 `text` 两个位置**:适配器的增量事件
        (`agent_text_delta` / `reasoning_delta`)把文本放在 `text` 里,旧式事件放 `message`。
        只取 `message` 会让增量的文本变成空字符串 —— 界面气泡永远不增长。
        """
        if isinstance(event, AgentEvent):
            self._emit(event.kind, event.message, event.payload)
        elif isinstance(event, dict):
            message = event.get("message")
            if not message:
                message = event.get("text") or ""
            payload = event.get("payload")
            if payload is None and event.get("text") is not None:
                payload = {"text": event.get("text")}
            self._emit(event.get("kind", "event"), message, payload or {})

    def _emit(self, kind, message, payload=None):
        """构造并转发事件。

        event_sink 属于 UI 层:其异常绝不能让工具调用链断在半途(回调抛错时
        已经落盘的副作用会被误判成内部错误),因此这里只降级、不改返回值。
        """
        self._seq += 1
        event = AgentEvent(run_id=self._run_id, sequence=self._seq, kind=kind,
                           message=message, payload=payload or {})
        try:
            self.event_sink(event)
        except Exception:
            pass
        return event

    def _audit(self, method: str, *args, **kwargs) -> bool:
        """审计落库统一入口:所有 store 调用都经此转发。

        返回值表示是否写入成功。任何异常都被吞掉(父目录只读、database is locked、
        磁盘写满等),首次失败打印一行提示,绝不抛出——审计是旁路,不能成为
        任务失败的原因,更不能把已执行完成的工具调用报成 INTERNAL_ERROR。
        """
        try:
            getattr(self.store, method)(*args, **kwargs)
            return True
        except Exception as e:
            self._audit_failures += 1
            if self._audit_failures == 1:
                try:
                    print(f"[agent] 审计写入失败(已忽略,任务继续): {method}: {type(e).__name__}: {e}")
                except Exception:
                    pass
            return False

    def _max_model_retries(self) -> int:
        """瞬态错误重试预算:与 max_model_rounds 相乘不会放大付费请求数。"""
        try:
            value = int(getattr(self.settings, "max_model_retries", 4))
        except (TypeError, ValueError):
            return 4
        return max(0, value)

    def tools_registry_definitions(self):
        registry = getattr(self.tools, "registry", None)
        if registry is not None:
            return registry.definitions()
        return []

    def _format_tool_output(self, result: ToolResult) -> str:
        """把工具结果格式化成回填给模型的文本。

        files.py 会把分段信号放在 metadata(如 truncated/returned_lines/start_line),
        只回填 content[:16000] 而不带任何标记时,模型会误以为内容完整并据此覆盖大文件。

        **回填前按值遮盖已知秘密**:这条文本会进入模型输入(审计库那条路径有
        `redact`,但管不到这里)。刻意只做"值遮盖"而**不套用整套 `redact()` 模式规则**:
        后者会把主人明确要求读取的文件正文里的 `KEY=value` 也改花 —— 而读取这类
        文件正当地要求拿到原文。
        """
        content = mask_known_secrets(result.content or "")
        text = content
        if len(content) > _MAX_TOOL_RESULT_CHARS:
            text = content[:_MAX_TOOL_RESULT_CHARS] + (
                f"\n…[结果已截断:原文 {len(content)} 字符,仅回填前 {_MAX_TOOL_RESULT_CHARS} 字符;"
                "如需后续内容请用 start_line 继续分段读取]")
        if result.metadata:
            try:
                meta = json.dumps(result.metadata, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                meta = str(result.metadata)
            text += f"\n[metadata] {mask_known_secrets(meta)[:_MAX_TOOL_METADATA_CHARS]}"
        return text

    def _retry_model(self, input_items, cancellation, instructions, attempts):
        """瞬态错误重试。

        每次额外请求都要从 max_model_retries 预算里扣除(预算按 run 计,不随轮次重置),
        耗尽即返回 None,由调用方按 MODEL_FATAL 结束:否则最坏情况是
        max_model_rounds × attempts 次付费请求。退避通过可注入的 sleeper 执行
        (默认真的睡,调试时可以传一个不睡的替身)。
        """
        for _ in range(attempts):
            if self._is_cancelled(cancellation):
                return None
            if self._model_retries >= self._max_model_retries():
                return None   # 重试预算耗尽
            self._model_retries += 1
            self._model_requests += 1   # 重试也是真实付费请求,必须计入统一统计
            self.sleeper(_RETRY_DELAY_SECONDS)
            turn = self.model.complete(
                input_items, self.tools_registry_definitions(), self._sink, cancellation,
                instructions=instructions,
            )
            # 重试请求同样消耗用量:即使这一发仍然失败,它也已经计费了
            if self._usage_sink is not None:
                self._usage_sink(turn)
            if not any(pe.code == "MODEL_TRANSIENT" for pe in turn.protocol_errors):
                return turn
        return None   # 全部尝试仍为瞬态错误

    def _handle_call(self, call: ToolCall, run_id: str, cancellation, last_sig=None):
        """返回 (result, stop, sig, elapsed)。

        - result: ToolResult 或 None(取消)
        - stop: (RunState, code, message) 或 None
        - sig: 本次调用的 (工具名, 规范参数哈希),校验失败时为 None
        - elapsed: 实际活动耗时(不含等待用户审批的时间)
        重复调用检测在 prepare 成功之后、execute 之前执行,防止第二次副作用。
        """
        t0 = self.clock()
        try:
            prepared = self.tools.prepare(call)
        except KeyError:
            elapsed = self.clock() - t0
            result = ToolResult(call.id, False, "UNKNOWN_TOOL", f"未注册工具: {call.name}")
            ev = self._emit("tool_result", f"工具 {call.name} 不存在", {
                "tool_name": call.name, "code": result.code, "summary": result.content})
            self._audit("record_tool_event", run_id, ev)
            self._record_attempt(call, result.code)
            return result, None, None, elapsed
        except ValueError as e:
            elapsed = self.clock() - t0
            self._invalid_streak = getattr(self, "_invalid_streak", 0) + 1
            result = ToolResult(call.id, False, "INVALID_ARGUMENTS", str(e))
            ev = self._emit("tool_result", "工具参数无效", {
                "tool_name": call.name, "code": result.code, "summary": result.content})
            self._audit("record_tool_event", run_id, ev)
            self._record_attempt(call, result.code)
            if self._invalid_streak >= 2:
                return result, (RunState.FAILED, "INVALID_ARGUMENTS", "连续两次工具参数无效,已停止"), None, elapsed
            return result, None, None, elapsed
        elapsed_prep = self.clock() - t0
        self._invalid_streak = 0

        # 重复调用检测:prepare 已成功(参数合法),执行前拦截
        sig = (call.name, prepared.arguments_hash)
        if last_sig is not None and sig == last_sig:
            result = ToolResult(call.id, False, "REPEATED_CALL", "检测到重复调用,已停止")
            self._emit("tool_result", "重复调用已拦截", {
                "tool_name": call.name, "code": result.code, "summary": result.content})
            self._record_action(call, prepared, result)
            return result, (RunState.FAILED, "REPEATED_CALL", "检测到重复调用,已停止"), None, elapsed_prep

        decision = prepared.policy.decision
        if decision == PolicyDecision.DENY:
            result = ToolResult.denied(call.id, prepared.policy.reason)
            ev = self._emit("tool_result", f"已拒绝 {call.name}", {
                "tool_name": call.name, "code": result.code, "summary": result.content})
            self._audit("record_tool_event", run_id, ev)
            self._record_action(call, prepared, result)
            return result, None, sig, elapsed_prep

        if decision == PolicyDecision.ASK:
            approval = ApprovalRequest(
                approval_id=f"{run_id}-{call.id}",
                run_id=run_id,
                tool_call_id=call.id,
                tool_name=call.name,
                canonical_arguments=prepared.canonical_arguments,
                arguments_hash=prepared.arguments_hash,
                risk_summary=prepared.policy.risk_summary,
                data_leaves_device=prepared.policy.data_leaves_device,
                file_fingerprint=prepared.file_fingerprint,
            )
            # 先注册 pending 再发事件:否则"事件已送达、主人立刻点允许"的那一票会被丢弃
            self._register_pending(approval.approval_id)
            try:
                self._audit("record_approval", run_id, approval)
                # 等待期间的真实状态先落库:UI 重启/进程被杀后能看出任务卡在审批而非"运行中";
                # 放在审批事件之前,UI 收到卡片时查询数据库一定能看到 awaiting_approval
                self._audit("set_run_state", run_id, RunState.AWAITING_APPROVAL,
                            "AWAITING_APPROVAL", f"等待主人批准 {approval.tool_name}")
                self._emit("approval_requested", "需要主人批准以下操作", {
                    "approval_id": approval.approval_id,
                    "tool_name": approval.tool_name,
                    "arguments": approval.canonical_arguments,
                    "risk_summary": approval.risk_summary,
                    "data_leaves_device": approval.data_leaves_device,
                })
                decision_obj = self._wait_approval(approval.approval_id, cancellation)   # 审批等待不计时
                # 审批结束恢复运行态(非终态不写 finished_at,见 AgentStore.set_run_state)
                self._audit("set_run_state", run_id, RunState.RUNNING, "", "")
            finally:
                self._unregister_pending(approval.approval_id)
            if decision_obj is None:
                return None, (RunState.CANCELLED, "CANCELLED", "任务已取消"), None, elapsed_prep
            if decision_obj is _APPROVAL_TIMEOUT:
                return None, (RunState.FAILED, "APPROVAL_TIMEOUT", "等待主人审批超时,任务已结束"), None, elapsed_prep
            self._audit("record_approval_decision", approval.approval_id, decision_obj.allowed)
            if not decision_obj.allowed:
                result = ToolResult.user_denied(call.id)
                self._emit("tool_result", "主人拒绝了该操作", {
                    "tool_name": call.name, "code": result.code, "summary": result.content})
                self._record_action(call, prepared, result)
                return result, None, sig, elapsed_prep
            grant = ApprovalGrant.from_prepared(run_id, prepared)
        else:
            grant = None

        t1 = self.clock()
        result = self.tools.execute(prepared, grant)
        elapsed = elapsed_prep + (self.clock() - t1)
        # 事件与任务摘要都要**先脱敏**再落:工具结果原文若不在这两条路径上遮盖
        # (模型回填与数据库那条路径已遮),已登记的秘密会顺着
        # 事件 → UI → TaskResult 一路显示出来。
        summary = redact(result.content)[:500]
        if "敏感" in (prepared.policy.risk_summary or ""):
            # 敏感文件操作:只记录状态与路径摘要,不保存内容
            summary = f"[敏感文件操作,内容不记录] {result.code}"
        ev = self._emit("tool_result", f"{call.name} 执行完成", {
            "tool_name": call.name, "code": result.code, "summary": summary,
            "arguments": prepared.canonical_arguments,
            # 截断事实与真实耗时都进事件/审计:否则事后无法判断模型看到的是否完整
            "truncated": len(result.content or "") > _MAX_TOOL_RESULT_CHARS,
            "duration_ms": int(elapsed * 1000),
        })
        self._audit("record_tool_event", run_id, ev)
        self._record_action(call, prepared, result)
        return result, None, sig, elapsed

    def _collect_source(self, run_sources: list, url) -> None:
        """把来源 URL 收进"本次运行的事实清单"(TaskResult.sources)。

        只写审计库与事件时,`run_sources` 会始终为空 → 交付结果里"参考来源"永远不显示,
        界面/回复拿不到来源。这里与审计写入并列,保证两者一致。
        """
        if isinstance(url, str) and url.strip() and url not in run_sources:
            run_sources.append(url.strip())

    def _record_attempt(self, call: ToolCall, code: str) -> None:
        """记录一次**没走到执行**的工具尝试(未知工具/参数校验失败)。

        没有这条记录时,失败面板只能说"工具调用 N 次"却列不出做了什么:
        会出现"面板显示 1 次调用、动作清单却是空的" ——
        主人无法从交付说明里看出模型尝试过什么。

        只记录程序已确认的事实:工具名、结果码;参数在 `canonical_arguments` 不可用时
        不做推测(因此 targets 为空,不会被当成产物)。
        """
        self._actions.append(TaskAction(
            tool=str(call.name), code=str(code), ok=False, targets=(), summary=""))

    def _record_action(self, call, prepared, result) -> None:
        """把这次工具调用记进"可核对的事实"(供 TaskResult 使用)。

        只记录程序已知的事实(工具名/结果码/目标路径),不做任何推测 ——
        模型据此说明结果时,"文件已生成"背后一定有一条成功记录。
        """
        arguments = prepared.canonical_arguments or {}
        targets = []
        for key in ("path", "destination", "source", "path_or_url", "url"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                targets.append(value.strip())
        self._actions.append(TaskAction(
            tool=str(call.name),
            code=str(result.code),
            ok=bool(result.ok),
            targets=tuple(targets),
            # 任务摘要会进 TaskResult → UI 面板与审计库,同样先脱敏
            summary=redact(str(result.content or ""))[:200],
        ))

    def _register_pending(self, approval_id: str) -> None:
        with self._condition:
            self._pending[approval_id] = approval_id

    def _unregister_pending(self, approval_id: str) -> None:
        with self._condition:
            self._pending.pop(approval_id, None)

    def _wait_approval(self, approval_id: str, cancellation):
        """等待审批决定,返回 ApprovalDecision / None(取消) / _APPROVAL_TIMEOUT(超时)。

        超时用真实墙钟计算:审批等待不计入 active 活动时间,因此也不该受注入 clock 影响;
        没有这个上限时,任何一次丢失的决定都会让任务永久挂起。
        """
        limit = getattr(self.settings, "approval_timeout_seconds", 0) or 0
        deadline = (_time.monotonic() + limit) if limit > 0 else None
        while True:
            if self._is_cancelled(cancellation):
                return None
            with self._condition:
                decision = self._decisions.pop(approval_id, None)
                if decision is not None:
                    return decision
                self._condition.wait(0.2)
            if deadline is not None and _time.monotonic() >= deadline:
                return _APPROVAL_TIMEOUT
