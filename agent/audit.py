# -*- coding: utf-8 -*-
"""审计脱敏:递归遮盖敏感键与 Bearer 值;日志、异常、UI、数据库共用。"""
import re

_SENSITIVE_KEY_RE = re.compile(
    r"(authorization|api_key|apikey|token|cookie|password|secret|credential)", re.IGNORECASE
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_SK_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")
_KV_SECRET_RE = re.compile(
    r"(?i)\b([\w.-]*(?:password|passwd|secret|token|credential|api_?key|cookie)\w*)\s*[=:]\s*([^\s,;\"'<>]+)"
)

# 已知秘密的**逐字**遮盖表。
#
# 为什么需要:上面几条规则靠"上下文"识别(Bearer / sk- / KEY=VALUE),而主人自填的
# 秘密在异常文本或日志里往往是**裸的**,三条规则都命中不了 —— redact() 对这样的文本
# 原样返回。因此由持有该秘密的模块把值登记进来,这里按值遮盖。
#
# 只登记长度 ≥ _MIN_REGISTERED_SECRET 的值:太短的值(如 "test")做全局替换会把
# 无关文本改花。这条下限本身就是"不误伤"的守卫。
#
# web_search 不需要密钥(ddgs 不需要 API key),因此当前没有调用方;
# 机制保留给将来任何"确实持有秘密值"的模块 —— 删掉它会让下一个密钥又退回"裸的"。
_MIN_REGISTERED_SECRET = 8
_KNOWN_SECRETS: list[str] = []


def register_secret(secret) -> None:
    """登记一个已知秘密值,供 `redact()` 逐字遮盖。长度不足则忽略(避免误伤)。"""
    value = str(secret or "")
    if len(value) < _MIN_REGISTERED_SECRET:
        return
    if value not in _KNOWN_SECRETS:
        # 长值优先替换,避免短值先把长值切碎后长值再也匹配不上
        _KNOWN_SECRETS.append(value)
        _KNOWN_SECRETS.sort(key=len, reverse=True)


def mask_known_secrets(value: str) -> str:
    """只按**登记过的秘密值**逐字遮盖,不套用模式规则。

    这是给"内容本身必须保持原样、但绝不能带出已知秘密"的路径用的 —— 典型是**回填给
    模型的工具结果**:主人明确要求读某个文件时,文件正文里的 `KEY=value` 是普通文本,
    不该被改花(套用 `redact()` 会把它遮成 `KEY=***`,等于污染了主人要看的内容),
    但 App 自己的密钥绝不能因此泄漏。
    """
    text = value if isinstance(value, str) else str(value)
    for secret in _KNOWN_SECRETS:
        if secret in text:
            text = text.replace(secret, "***")
    return text


def redact(value):
    """递归脱敏:敏感键整值替换为 ***,字符串中的 Bearer/sk-/KEY=VALUE 形式遮盖。"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, str) and _SENSITIVE_KEY_RE.search(k):
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = mask_known_secrets(value)
        value = _BEARER_RE.sub(r"\1***", value)
        value = _SK_KEY_RE.sub("sk-***", value)
        return _KV_SECRET_RE.sub(r"\1=***", value)
    return value


def summarize_text(text, max_chars=500) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"…（截断，原 {len(text)} 字）"
