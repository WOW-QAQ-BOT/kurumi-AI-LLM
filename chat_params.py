# -*- coding: utf-8 -*-
"""DeepSeek 思考模式控制。

为什么单独成模块:`ui/workers.py` 里的 ApiChatWorker/ApiMemoryWorker 需要这几个函数,
而它们**不能**从 `ui` 包导入(会循环导入)。

状态(`_API_THINKING_SUPPORTED` / `_API_THINKING_DEGRADED`)只在本模块内读写;
`ui/__init__.py` 重新导出同名名字以保持既有访问方式可用。
"""
from openai import BadRequestError

# ==================== DeepSeek 思考模式控制 ====================
# 文档: https://api-docs.deepseek.com/guides/thinking_mode/
# - 思考模式**默认开启**(默认 effort=high),思考内容走 delta.reasoning_content;
# - max_tokens 是"思考 + 正文"共享预算 → 思考过长时正文可能为空(finish_reason=length),
#   界面就会渲染出空气泡,history 里还会塞进一条空 assistant;
# - 思考模式会**静默忽略** temperature/top_p/presence_penalty/frequency_penalty;
# - OpenAI SDK 必须用 extra_body 传:{"thinking": {"type": "disabled"|"enabled"}}。
# 本地引擎本来就关思考(enable_thinking=False),API 引擎也需要同样的控制 —— 由本模块补齐。
_API_THINKING_SUPPORTED = True
# _chat_create 首次发现"端点不接受 thinking 参数"时写入这里的告警文案;
# 由 worker 通过 degraded 信号取走并上屏(取走即清空,不会每轮重复刷屏)。
_API_THINKING_DEGRADED = None


def _take_thinking_degradation():
    """取出「本次请求新发生思考降级」的提示;取走即清空。"""
    global _API_THINKING_DEGRADED
    msg, _API_THINKING_DEGRADED = _API_THINKING_DEGRADED, None
    return msg


def _thinking_kwargs(mode="disabled"):
    """把 --api_thinking 转成请求参数;端点不认识时由 _chat_create 负责回退。"""
    mode = str(mode or "disabled").lower()
    if mode == "disabled":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if mode in ("low", "high", "max"):
        return {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": mode}
    return {}


def _unsupported_thinking_param(err):
    """只有明确提到这几个参数名的错误才算「端点不支持」,避免把真实 400 当参数问题。

    用 unknown/unrecognized 之类宽关键词会把 `unknown model` 这类真实错误也判成
    参数问题 → 多打一次付费请求,并永久关掉用户显式要求的 --api_thinking。
    """
    text = str(err).lower()
    return any(k in text for k in ("thinking", "reasoning_effort", "extra_body"))


def _chat_create(client, **kwargs):
    """调用 chat.completions.create;端点/SDK 不认识 thinking 参数时去掉重试一次。

    回退语义很重要:去掉参数后服务端会回到**默认(思考开启)**,即"空回复 + 白烧 token"
    的情况会回来。因此这里只回退一次并记录降级,由调用方
    (ApiChatWorker/ApiMemoryWorker)经 `degraded` 信号把这件事**上屏**,不静默吞掉。
    只对"参数不被支持"类错误回退,其余错误照常抛出(如上下文超限)。
    """
    global _API_THINKING_SUPPORTED, _API_THINKING_DEGRADED
    if "extra_body" in kwargs and not _API_THINKING_SUPPORTED:
        # 已经确认端点不接受该参数:后续请求直接不发,避免每轮都白撞一次 400
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in ("extra_body", "reasoning_effort")}
    if _API_THINKING_SUPPORTED and "extra_body" in kwargs:
        try:
            return client.chat.completions.create(**kwargs)
        except (TypeError, BadRequestError) as e:
            if not _unsupported_thinking_param(e):
                raise
            _API_THINKING_SUPPORTED = False
            _API_THINKING_DEGRADED = (
                "该端点不接受 thinking 参数,无法关闭思考模式:服务端将按默认(思考开启)处理,"
                "正文可能被思考占满预算而为空;可改用支持该参数的端点,或提高 --max_new_tokens。")
            print(f"[!] {_API_THINKING_DEGRADED} ({e})")
            kwargs = {k: v for k, v in kwargs.items()
                      if k not in ("extra_body", "reasoning_effort")}
    return client.chat.completions.create(**kwargs)
