# -*- coding: utf-8 -*-
"""DeepSeek Responses API 适配器:能力探测、事件归一化、SDK 对象不出本模块。"""
import json
import time
from dataclasses import dataclass

from agent.audit import redact
from agent.types import ModelTurn, ProtocolError, ToolCall


@dataclass(frozen=True)
class CapabilityItem:
    """单项能力。

    `ok` 是结论,`verified` 是"这个结论是否有实际证据":
    - verified=True + ok=True  → 已由真实动作/来源确认可用;
    - verified=True + ok=False  → 已确认不可用(例如端点接受了参数但没有任何搜索动作);
    - verified=False           → 只是推断或探测本身失败,**不能当作"不支持"**。
    """

    name: str
    ok: bool
    verified: bool = True
    evidence: str = ""


@dataclass(frozen=True)
class CapabilityReport:
    """分项能力报告。

    一个布尔值代表不了"Agent 全部能力":探测只看"带 web_search 的请求是否报错"——
    官方文档明确写着内置 `web_search` 会被忽略,所以"请求成功"既不能证明"搜索可用",
    也不能证明"函数调用可用"。四类能力必须分开记录。
    """

    model_call: CapabilityItem
    function_calls: CapabilityItem
    web_search: CapabilityItem
    streaming: CapabilityItem
    retryable: bool = False      # 探测因瞬时故障失败:可稍后重试,不要永久禁用 Agent

    @property
    def agent_usable(self) -> bool:
        """Agent 能不能用:只取决于模型调用是否可用(本地工具不依赖联网搜索)。"""
        return self.model_call.ok

    @property
    def web_search_usable(self) -> bool:
        return self.web_search.ok

    # 兼容旧调用方(旧字段语义 = "Agent 是否可用")。新代码请用分项字段。
    @property
    def supported(self) -> bool:
        return self.agent_usable

    @property
    def reason(self) -> str:
        """不可用/降级的说明:模型调用失败优先,其次是联网搜索的结论。"""
        if not self.model_call.ok:
            return self.model_call.evidence or "模型调用不可用"
        if not self.web_search.ok:
            return self.web_search.evidence
        return ""

    def items(self) -> tuple:
        return (self.model_call, self.function_calls, self.web_search, self.streaming)

    def status_text(self) -> str:
        """供 UI 展示:联网未确认时**绝不说"联网就绪"**。

        措辞要点:这里的"联网搜索"指的是**端点内置的** `web_search`,与本应用自己的
        `web_search` 工具(走 ddgs)是两件事。主人明确要求**不要再在状态栏解释**这件事,
        所以这里只给结论,不附带"这不影响自带工具"之类的说明。
        """
        if not self.agent_usable:
            return f"Agent 不可用:{self.reason}"
        if self.web_search_usable:
            return "Agent 就绪(DeepSeek Responses + 内置联网搜索已确认)"
        caveat = "未确认" if not self.web_search.verified else "不可用"
        return f"Agent 就绪(本地工具可用)· **端点内置**联网搜索{caveat}:{self.web_search.evidence}"


class AgentCancelled(RuntimeError):
    """协作式取消信号。"""


# 取消在途请求时把 HTTP 超时压到该值:让读等待尽快结束,而不是等到配置的 45 秒。
#
# 为什么是 0.25 而不是 1.0:真实端点上压到 1.0 时"从点停止到真正结束"仍要
# ~8.9 秒(httpx 的 socket 轮询不会立刻感知超时变化,存在额外延迟)。
# 0.25 秒足以让回环上的正常响应完成,又明显缩短等待。
_CANCEL_GRACE_SECONDS = 0.25


def _get(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _usage_counts(response):
    """从响应里取 (输入 token, 输出 token);取不到就是 (None, None)。

    **绝不猜测**:端点没给 usage 时统计必须显示"未知",而不是 0 ——
    把"没有数据"写成 0 会让主人以为这次没花钱。
    """
    usage = _get(response, "usage")
    if usage is None:
        return None, None

    def pick(*names):
        for name in names:
            value = _get(usage, name)
            if isinstance(value, bool) or value is None:
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                return number
        return None

    return pick("input_tokens", "prompt_tokens"), pick("output_tokens", "completion_tokens")


def _plain_item(item):
    """把 SDK 响应项转成可序列化的普通 dict——SDK 对象不出本模块。"""
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            out = dump(exclude_unset=True)
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    out = {}
    for key in ("type", "id", "name", "status", "role", "action"):
        if hasattr(item, key):
            out[key] = getattr(item, key)
    return out


def _extract_urls(item, limit=5):
    """从 web_search_call 项中尽力提取来源 URL(供审计与 UI 展示)。"""
    urls = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "url" and isinstance(v, str) and v.startswith(("http://", "https://")):
                    if v not in urls:
                        urls.append(v)
                else:
                    walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)

    try:
        walk(_plain_item(item))
    except Exception:
        pass
    return urls[:limit]


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    text = str(exc).lower()
    return any(h in text for h in ("timeout", "timed out", "rate limit", "429", "5xx", "connection"))


# 搜索探测输入:必须要求"新鲜信息 + 来源",否则模型可能凭记忆直接回答,导致误判
_SEARCH_PROBE_INPUT = "请联网搜索今天的最新新闻,并给出至少一个来源链接。"
_PROBE_MAX_OUTPUT_TOKENS = 64


def _search_evidence(items) -> list:
    """从响应项里找"确实搜过"的证据:web_search_call 项或 url_citation 来源标注。

    只看"请求是否报错"是不够的:端点可以接受 `tools=[{"type":"web_search"}]`
    却什么都不做(官方文档明确说明内置 web_search 可能被忽略)。判定"搜索可用"
    必须观察到真实动作或来源。
    """
    found = []
    for item in items or []:
        kind = _get(item, "type")
        if kind == "web_search_call":
            found.append("web_search_call")
        action = _get(item, "action")
        if isinstance(action, dict) and str(action.get("type") or "") == "search":
            found.append("action=search")
        if kind == "message":
            for part in (_get(item, "content") or []):
                for ann in (_get(part, "annotations") or []):
                    if _get(ann, "type") == "url_citation":
                        found.append("url_citation")
    return found


class DeepSeekAgentClient:
    """封装 Responses API。所有输入输出归一化为 agent 契约;SDK 对象不逃逸。"""

    def __init__(self, model: str, api=None, api_key: str = "", base_url: str = "https://api.deepseek.com",
                 timeout: float = 45.0, max_output_tokens: int = 8192, stream: bool = False):
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        # 流式反馈。**默认关**:既保证"不传就是老行为",
        # 也让假 SDK(离线替身)不必实现流式;生产由 UI 按 api_config.json 显式开启。
        # 若开启后端点不支持,会自动退回非流式。
        self.stream = bool(stream)
        self._api = api
        # 重建底层客户端所需的信息(取消时可能把它关掉,见 cancel)
        self._api_key = api_key
        self._base_url = base_url
        if self._api is None:
            from openai import OpenAI
            self._api = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._capability_cache = None
        self._web_unavailable_notified = False
        # 真正的取消。见 cancel() / _raise_if_cancelled() 的说明。
        self._cancel_requested = False
        self._http_client = None

    def probe_capabilities(self) -> CapabilityReport:
        """分项探测:模型调用 / 函数调用 / 网页搜索 / 流式输出。

        两步,最多两次小请求:
        1. 不带 web_search 的最小请求 → 确认 Responses 端点与模型可用(决定 Agent 能否使用);
        2. 带 web_search 且要求"新鲜信息 + 来源"的请求 → **检查是否真的产生搜索动作或来源**。
           只有观察到动作/来源才算"联网可用";端点接受参数却什么都不做时,结论是"不可用"
           并给出该依据,绝不会显示成"联网就绪"。

        探测结果在进程内缓存;**瞬时故障不写缓存**(否则一次网络抖动会把 Agent
        永久钉成不可用,用户只能重启)。
        """
        if self._capability_cache is not None:
            return self._capability_cache

        model_call = self._probe_model_call()
        if not model_call.ok:
            report = CapabilityReport(
                model_call=model_call,
                function_calls=CapabilityItem(
                    "function_calls", False, verified=False,
                    evidence="模型调用不可用,函数调用未能验证"),
                web_search=CapabilityItem(
                    "web_search", False, verified=False,
                    evidence="模型调用不可用,联网搜索未能验证"),
                streaming=self._streaming_item(),
                retryable=not model_call.verified,
            )
            if not report.retryable:
                self._capability_cache = report
            return report

        web_search = self._probe_web_search()
        report = CapabilityReport(
            model_call=model_call,
            function_calls=CapabilityItem(
                "function_calls", True, verified=False,
                evidence="随模型调用一并假定可用(每次实际往返会再次校验:没有 function_call 项即视为本轮无工具调用)"),
            web_search=web_search,
            streaming=self._streaming_item(),
            retryable=not web_search.verified,
        )
        if not report.retryable:
            self._capability_cache = report
        return report

    def _streaming_item(self) -> CapabilityItem:
        """流式能力:如实反映**本客户端当前的配置**,而不是一句静态的"未验证"。

        适配器已经实现流式,`stream` 为真时它就是本次运行的实际行为;
        若端点拒绝过流式(会自动退回),`stream` 已被置为 False ——
        那时必须说"不可用"并给出真实原因,而不是继续显示"未验证"。

        **不额外发探测请求**:流式与否在第一次运行看到增量就确定了,为此再花一次付费请求
        没有意义(能力探测已经是两次请求)。
        """
        if not self.stream:
            return CapabilityItem(
                "streaming", False, verified=True,
                evidence="端点/SDK 不接受流式请求,已自动退回非流式(功能不受影响)")
        return CapabilityItem(
            "streaming", True, verified=False,
            evidence="按配置使用流式输出(以每次运行是否收到正文增量为准,不额外发探测请求)")

    def _probe_model_call(self) -> CapabilityItem:
        """最小非变更请求:确认 Responses 端点与模型可用。不带任何工具。"""
        try:
            self._api.responses.create(
                model=self.model,
                input="ping",
                max_output_tokens=_PROBE_MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            transient = _is_transient(e)
            return CapabilityItem(
                "model_call", False, verified=not transient,
                evidence=redact(str(e)) + ("（疑似瞬时故障,可稍后重试）" if transient else ""),
            )
        return CapabilityItem("model_call", True, verified=True, evidence="Responses 请求成功")

    def _probe_web_search(self) -> CapabilityItem:
        """验证联网搜索是否**真的可用**:必须观察到搜索动作或来源。

        三种结局区分清楚:
        - 请求报错且是瞬时的 → verified=False(未验证),不能断言"不支持";
        - 请求报错(非瞬时) → verified=True, ok=False,原因就是错误本身;
        - 请求成功但没有任何 web_search_call / url_citation → verified=True, ok=False,
          依据写明"端点接受了参数但没有产生任何搜索动作"——这正是官方文档描述的
          "内置 web_search 被忽略"的形态。
        """
        try:
            response = self._api.responses.create(
                model=self.model,
                input=_SEARCH_PROBE_INPUT,
                tools=[{"type": "web_search"}],
                max_output_tokens=_PROBE_MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            transient = _is_transient(e)
            return CapabilityItem(
                "web_search", False, verified=not transient,
                evidence=redact(str(e)) + ("（疑似瞬时故障,可稍后重试）" if transient else ""),
            )
        evidence = _search_evidence(_get(response, "output") or [])
        if evidence:
            return CapabilityItem(
                "web_search", True, verified=True,
                evidence=f"已由实际搜索动作/来源确认({'、'.join(sorted(set(evidence)))})",
            )
        return CapabilityItem(
            "web_search", False, verified=True,
            evidence="端点接受了 web_search 参数但没有产生任何搜索动作或来源;"
                     "内置 web_search 会被忽略,故按“不具备联网搜索能力”处理"
                     "(本地工具不受影响)",
        )

    def cancel(self) -> None:
        """请求取消**在途**的模型请求。

        取消不能只等当前这一轮请求自己结束(最长到配置的超时),否则主人点了
        「停止」仍要继续等十几秒。做法:
        - 置取消标志 → 请求返回后立刻抛 `AgentCancelled`,不被误判成模型故障、
          也不会因此把结果回填给模型;
        - 把底层 HTTP 客户端的 timeout 压到 `_CANCEL_GRACE_SECONDS` ——
          **在途**的读等待会在 1 秒左右以超时结束,而不是等到 45 秒。
          (`client.timeout` 运行时可改,请求上带的 `read` timeout 随之变化 ——
          这是压超时能生效的前提。)
        - 下一轮 `complete()` 会把超时恢复成配置值(见 complete 开头)。

        局限(诚实记录):HTTP 请求无法真正"中断"。这里用**关闭底层连接**达到近似效果
        (真实端点:从点停止到结束 ~1.4s;不取消则要等到 45 秒超时);
        但服务端可能仍在继续生成并按 token 计费 —— 要让它真正停下得靠流式 + 主动断流,
        属于流式反馈要解决的事。
        """
        self._cancel_requested = True
        self._shrink_timeout()
        self._abort_http_client()

    def _abort_http_client(self) -> None:
        """关闭底层 HTTP 客户端,立刻中断在途 socket。

        为什么需要:只把 timeout 压小,httpx 仍可能在已建立的响应流上继续等待 ——
        真实端点"从点停止到结束"要 ~7 秒。关闭连接才是立刻的。

        代价:关闭后 SDK **不会**自动重建,后续请求会直接失败。因此
        `complete()` 每轮开始都会调用 `_ensure_http_client()` 重建。
        """
        target = getattr(self._api, "_client", None)
        if target is None:
            return
        try:
            target.close()
        except Exception:
            pass

    def _ensure_http_client(self) -> None:
        """底层客户端被关闭过 → 重建一个(取消后的下一轮请求必须能正常工作)。

        重建时只还原超时:base_url/api_key 由 SDK 自身保存,新建的 httpx 客户端
        不需要它们。注入的 SDK 替身通常没有 `is_closed`,会在这里安全返回。
        """
        if self._api is None:
            return
        target = getattr(self._api, "_client", None)
        if target is None or not getattr(target, "is_closed", False):
            return
        try:
            import httpx
            self._api._client = httpx.Client(timeout=httpx.Timeout(self.timeout))
        except Exception:
            pass

    def reset_cancel(self) -> None:
        """清除取消标志,让本客户端可以服务下一次运行。

        与 `cancel()` 成对出现:取消是**运行级**语义,而 runner 的实例在多次
        `run()` 之间可能被复用(UI 的 factory 每次 create 新实例,但调试脚本等
        其他调用方可能复用),因此需要一个显式的"重新开始"入口。
        """
        self._cancel_requested = False
        self._restore_timeout()

    def _shrink_timeout(self) -> None:
        """把在途请求的超时压到宽限值(失败不影响取消语义)。"""
        for target in (getattr(self._api, "_client", None),
                       getattr(self._api, "timeout", None)):
            if target is None:
                continue
            try:
                if hasattr(target, "timeout"):
                    target.timeout = _CANCEL_GRACE_SECONDS
            except Exception:
                pass

    def _restore_timeout(self) -> None:
        """恢复本客户端配置的超时(每轮请求开始时调用)。"""
        target = getattr(self._api, "_client", None)
        if target is not None:
            try:
                import httpx
                target.timeout = httpx.Timeout(self.timeout)
            except Exception:
                pass

    def _raise_if_cancelled(self) -> None:
        """取消标志已置 → 抛 `AgentCancelled`(由 runner 转成 CANCELLED 终态)。"""
        if self._cancel_requested:
            raise AgentCancelled()

    def _call_api_streaming(self, request_kwargs, event_sink):
        """流式调用,返回 (最终响应, 已收正文增量, 已收思考增量)。

        设计要点:

        - **只让过程可见,不改结果形状**:最终响应仍然交回调用方,由**同一段解析逻辑**
          产出 ModelTurn(函数调用/web/引用/usage 因此完全一致,不会出现"流式与非流式
          行为不同"的分叉);
        - 正文增量发 `agent_text_delta`,思考增量发 `reasoning_delta`(界面可选择不上屏)。
          两者是两套事件名(`output_text.` / `reasoning_text.`),不会混淆;
        - 端点/SDK 不支持 `stream` 时:去掉该参数**退回非流式**,并把 `self.stream` 置 False,
          后续轮次不再白试(与 thinking 参数的回退策略一致)。
        """
        kwargs = dict(request_kwargs)
        kwargs["stream"] = True
        if "timeout" in kwargs:
            kwargs.pop("timeout")     # 超时已由底层客户端持有;此处传会让 SDK 报错
        try:
            stream = self._api.responses.create(**kwargs)
        except TypeError as e:
            if "stream" not in str(e):
                raise
            self.stream = False
            print("[!] 端点/SDK 不接受流式请求,本次起改用非流式(功能不受影响)")
            return self._call_api(request_kwargs), "", ""

        final = None
        text_parts, reasoning_parts = [], []
        # 停滞看门狗:流式下读超时**不再**终止请求(每个增量都是一次成功读取),
        # 因此需要自己盯着"多久没有新事件"。否则服务端卡住时 Agent 会一直挂着,
        # 而活动时限只在两次模型调用之间检查 —— 这是流式引入的新风险。
        #
        # 必须先取事件、处理完,再看门限:把检查放在循环开头会连"已经到达但还没处理"
        # 的那一个增量一起丢掉(放弃时正文会变成空)。
        stall_limit = max(30.0, float(self.timeout or 45.0) * 2)
        last_event_at = time.monotonic()
        iterator = iter(stream)
        try:
            while True:
                try:
                    event = next(iterator)
                except StopIteration:
                    break
                self._raise_if_cancelled()
                kind = str(_get(event, "type") or "")
                if kind == "response.output_text.delta":
                    delta = _get(event, "delta") or ""
                    if delta:
                        text_parts.append(str(delta))
                        self._emit_delta(event_sink, "agent_text_delta", delta)
                elif kind == "response.reasoning_text.delta":
                    delta = _get(event, "delta") or ""
                    if delta:
                        reasoning_parts.append(str(delta))
                        self._emit_delta(event_sink, "reasoning_delta", delta)
                elif kind in ("response.completed", "response.incomplete", "response.failed"):
                    # 这三个事件都带完整 response:交给调用方按既有逻辑解析
                    final = _get(event, "response")
                # 事件处理完再判停滞:超过门限说明服务端不再推进,放弃并把已收内容交回
                now = time.monotonic()
                if now - last_event_at > stall_limit:
                    print(f"[!] 流式响应停滞超过 {stall_limit:.0f} 秒,改用已收到的内容")
                    return None, "".join(text_parts), "".join(reasoning_parts)
                last_event_at = now
        except AgentCancelled:
            raise
        except Exception as e:
            # 事件流中途断了:把已收增量交回调用方兜底。
            # 若不在这里接住,异常会被 complete() 的通用分支变成"空回合",
            # 主人屏幕上已经出现的半截内容反而会消失。
            print(f"[!] 流式响应中断,改用已收到的内容: {type(e).__name__}: {e}")
            return None, "".join(text_parts), "".join(reasoning_parts)
        self._raise_if_cancelled()
        return final, "".join(text_parts), "".join(reasoning_parts)

    @staticmethod
    def _emit_delta(event_sink, kind, delta) -> None:
        """发一个增量事件;回调异常绝不影响模型调用链。

        文本同时放进 `text` 与 `message`:runner 的 `_sink` 按 `message` 构造 AgentEvent,
        界面读 `payload["text"]` —— 两个键都给,才不会在某一层被丢掉
        (只给 `text` 时界面拿到的 message 是空的,气泡永不增长)。
        """
        if event_sink is None:
            return
        try:
            event_sink({"kind": kind, "text": str(delta), "message": str(delta),
                        "payload": {"text": str(delta)}})
        except Exception:
            pass

    def _call_api(self, request_kwargs):
        """调用 Responses API;可选参数不被 SDK/端点接受时逐项剔除后重试。

        - store=False:显式关闭服务端留存(Responses 默认可能留存请求与响应);
        - truncation="auto":输出被上限截断时让服务端自动接着生成。
        老版本 SDK 会因未知关键字抛 TypeError,此时安全回退(功能不受影响),
        不会把请求直接打成 MODEL_FATAL。

        每次尝试前后都检查取消标志;并且**取消导致的请求失败会被改判成取消** ——
        否则它会被外层分类成 `MODEL_TRANSIENT`(读超时/连接中断),主人的动作是
        「停止」,报告却写成「模型请求多次失败」,还会白跑一次重试。
        """
        kwargs = dict(request_kwargs)
        while True:
            try:
                self._raise_if_cancelled()
                response = self._api.responses.create(**kwargs)
                self._raise_if_cancelled()
                return response
            except TypeError as e:
                message = str(e)
                dropped = None
                for key in ("truncation", "store"):
                    if key in kwargs and (key in message or "unexpected keyword" in message):
                        dropped = key
                        break
                if dropped is None:
                    raise
                kwargs.pop(dropped)
            except AgentCancelled:
                raise
            except Exception:
                # 请求失败时先看是不是"我们自己在取消":是 → 抛 AgentCancelled,
                # 让 runner 走 CANCELLED 终态,而不是把取消报成模型故障。
                self._raise_if_cancelled()
                raise

    def complete(self, input_items, tool_definitions, event_sink, cancellation, instructions: str = "") -> ModelTurn:
        # 每轮开始:恢复配置的超时;若上一轮取消时关掉了底层客户端,则重建它。
        #
        # **不清除取消标志**:取消是"本次运行"的语义(与 runner 的令牌一致),
        # 而 `complete()` 会被同一运行调用多轮 —— 若在这里重置标志,
        # 一次跨线程的取消会被下一次 complete 自己抹掉(取消后仍会返回结果)。
        # 需要重新开始时,由调用方显式 reset_cancel()。
        self._restore_timeout()
        self._ensure_http_client()
        if cancellation is not None and cancellation.cancelled:
            raise AgentCancelled()
        self._raise_if_cancelled()
        # 是否附带内置联网搜索:以"联网能力已被实际动作验证"为准,而不是"带参数请求没报错"。
        # 未探测过(capability is None)时保持默认附带(不擅自砍掉能力)。
        report = self._capability_cache
        web_usable = report is None or report.web_search_usable
        tools: list = []
        if web_usable:
            tools.append({"type": "web_search"})
        for d in tool_definitions or []:
            tools.append({
                "type": "function",
                "name": d.name,
                "description": d.description,
                "parameters": d.parameters,
            })
        request_kwargs = dict(
            model=self.model,
            input=input_items,
            tools=tools,
            timeout=self.timeout,
            max_output_tokens=self.max_output_tokens,
            store=False,          # 显式关闭服务端留存
            truncation="auto",    # 输出被截断时让服务端自动续写
        )
        if instructions:
            request_kwargs["instructions"] = instructions
        if not web_usable:
            # 不静默:明确告知本轮没有联网能力,并如实给出依据(已确认不支持 / 仅未验证)
            item = report.web_search
            state = "已确认不可用" if item.verified else "未通过验证"
            note = ProtocolError(
                "WEB_SEARCH_UNAVAILABLE",
                f"联网搜索{state},本轮已不附带 web_search 工具(本地工具仍可用);"
                f"依据: {item.evidence or '未知'}",
            )
            if event_sink is not None and not self._web_unavailable_notified:
                self._web_unavailable_notified = True
                try:
                    event_sink({"kind": "web_search_unavailable", "message": note.message,
                                "reason": item.evidence, "verified": item.verified})
                except Exception:
                    pass
        try:
            if self.stream:
                response, streamed_text, streamed_reasoning = self._call_api_streaming(
                    request_kwargs, event_sink)
                if response is None:
                    # 流式没有给完整响应:用它自己已经发过的增量凑一个(与既有"没有 message"
                    # 的处理不同,这里是在流被截断/端点只发增量时的兜底,必须明说来源)
                    if streamed_text:
                        return ModelTurn(text=streamed_text, prompt_tokens=None,
                                         completion_tokens=None)
                    if streamed_reasoning:
                        # 只产出了思考:与既有的"推理吃光预算"一致,报 OUTPUT_TRUNCATED
                        return ModelTurn(protocol_errors=(ProtocolError(
                            "OUTPUT_TRUNCATED",
                            f"流式响应只产出了思考内容({len(streamed_reasoning)} 字),没有正文;"
                            "通常是推理消耗了整个输出预算,请提高 "
                            "api_config.json 的 agent.max_output_tokens"),))
            else:
                response = self._call_api(request_kwargs)
                streamed_text = streamed_reasoning = ""
        except AgentCancelled:
            raise
        except Exception as e:
            code = "MODEL_TRANSIENT" if _is_transient(e) else "MODEL_FATAL"
            return ModelTurn(protocol_errors=(ProtocolError(code, redact(str(e))),))

        text_parts, calls, web, echoed, errors, citations = [], [], [], [], [], []
        if not web_usable:
            errors.append(note)
        for item in (response.output or []):
            kind = _get(item, "type")
            if kind == "message":
                for part in (_get(item, "content") or []):
                    ptype = _get(part, "type")
                    ptext = _get(part, "text")
                    if ptype in ("output_text", "text") and ptext:
                        text_parts.append(str(ptext))
                    # 引用来源常出现在 output_text.annotations(url_citation),与 web_search_call 互补
                    for ann in (_get(part, "annotations") or []):
                        if _get(ann, "type") == "url_citation" and _get(ann, "url"):
                            citations.append({
                                "url": str(_get(ann, "url")),
                                "title": str(_get(ann, "title") or ""),
                            })
            elif kind == "function_call":
                # 真实 Responses 对象同时含 id 与 call_id;配对必须用 call_id
                call_id = str(_get(item, "call_id") or _get(item, "id") or "")
                name = str(_get(item, "name") or "")
                try:
                    raw_args = _get(item, "arguments") or "{}"
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments 不是对象")
                except Exception:
                    errors.append(ProtocolError("INVALID_TOOL_JSON", "工具参数不是合法 JSON 对象"))
                    arguments = {}
                calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
                # 原始 function_call 项必须在下一轮原样回传(标准字段名为 call_id)
                echoed.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                })
            elif kind == "web_search_call":
                web.append({"type": "web_search_call", "urls": _extract_urls(item)})
                # 官方要求原样回传:保留 id/status/action 等全部字段
                echoed.append(_plain_item(item))
            else:
                # 不透明项(如 reasoning)仅保留在当前轮次,转为普通 dict,不持久化、不写日志
                echoed.append(_plain_item(item))
        if event_sink is not None:
            try:
                event_sink({
                    "kind": "model_turn",
                    "text": "\n".join(text_parts),
                    "function_calls": len(calls),
                    "web_actions": len(web),
                })
            except Exception:
                pass
        # 输出被上限截断(含推理 token)时给出明确提示,避免用户看到"说一半"的回复
        status = str(_get(response, "status") or "")
        if status == "incomplete":
            reason = _get(_get(response, "incomplete_details") or {}, "reason")
            detail = f"({reason})" if reason else ""
            if text_parts:
                text_parts.append(
                    f"\n【提示:回复因输出上限被截断{detail},可在 api_config.json 提高 agent.max_output_tokens】")
            else:
                # 推理吃光输出预算导致没有任何文本:不能静默落成 EMPTY_TURN
                errors.append(ProtocolError(
                    "OUTPUT_TRUNCATED",
                    f"模型输出被上限截断且没有任何文本内容{detail};"
                    "通常是推理消耗了整个输出预算,请提高 api_config.json 的 agent.max_output_tokens",
                ))
        prompt_tokens, completion_tokens = _usage_counts(response)
        if streamed_reasoning and not text_parts and not calls and not errors:
            # 流式专属兜底:整轮只产出了思考内容、没有任何正文/工具调用。
            # 真实端点会出现这种情形(思考吃光输出预算),必须明说原因,
            # 否则界面就是一个"什么都没有"的空回合。
            errors.append(ProtocolError(
                "OUTPUT_TRUNCATED",
                f"流式响应只产出了思考内容({len(streamed_reasoning)} 字),没有正文;"
                "通常是推理消耗了整个输出预算,请提高 api_config.json 的 agent.max_output_tokens",
            ))
        return ModelTurn(
            text="\n".join(text_parts),
            output_items=tuple(echoed),
            function_calls=tuple(calls),
            web_actions=tuple(web),
            citations=tuple(citations),
            protocol_errors=tuple(errors),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
