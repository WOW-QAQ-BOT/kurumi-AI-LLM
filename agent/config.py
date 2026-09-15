# -*- coding: utf-8 -*-
"""Agent 配置:只保存非秘密配置;API Key 一律走 Windows 凭据管理器。"""
import dataclasses
import json
import os
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "api_config.json"


def default_allowed_root() -> Path:
    """允许根默认值:当前用户的桌面目录。

    硬编码的 ``D:\\Desktop``——在真实机器上几乎总是错的(中文 Windows 的
    桌面通常是 ``C:\\Users\\<名>\\Desktop``,或由 OneDrive 重定向),该目录多半不存在,
    于是所有读取都会退化成"根外 → 每次都要批准"。所以这里按当前用户解析,
    并且**始终返回绝对路径**;实在找不到就退回家目录。
    """
    home = Path.home()
    onedrive = os.environ.get("OneDrive") or os.environ.get("OneDriveConsumer")
    candidates = [home / "Desktop", home / "桌面"]
    if onedrive:
        candidates.insert(0, Path(onedrive) / "Desktop")
    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            continue
    return home


def _safe_repr(instance, fields) -> str:
    """按 dataclass 字段生成 repr,把 `repr=False` 的字段值替换为 `***`。

    直接给敏感字段写 `repr=False` 会把**字段名**一起藏掉:有人排查"密钥到底配上没有"
    时,从 repr 上看不出这个字段存在,反而会去别处找。这里的做法是:字段名照常出现,
    值只在确实设过时显示 `***` —— 既不泄漏,也不隐藏"它存在且已配置"这个事实。

    遍历的是 `dataclasses.fields()`,所以**以后新加的秘密字段只要标了 `repr=False`
    就自动受保护**,不需要记得回来改这个函数。
    """
    parts = []
    for f in fields:
        try:
            value = getattr(instance, f.name)
        except AttributeError:            # 极端情况下字段缺失也不该让日志崩掉
            continue
        if f.repr is False:
            shown = "***" if value not in ("", None) else repr(value)
            parts.append(f"{f.name}={shown}")
        else:
            parts.append(f"{f.name}={value!r}")
    return f"{type(instance).__name__}({', '.join(parts)})"


@dataclass(frozen=True)
class AgentSettings:
    """Agent 配置。

    **凡是秘密字段一律 `repr=False`**:dataclass 的默认 repr 会把字段值原样打出来,
    而这个对象会被插进日志、异常文本与调试输出里 —— `repr=False`(配合下面的
    `__repr__`)是"不主动泄漏"的最低保证:默认 repr 下 `AgentSettings(web_search_api_key='…')`
    里的密钥会被原样打进日志。同理,这里不放任何**新增**的明文秘密
    字段;新秘密要么走凭据管理器,要么至少标 `repr=False`。
    """
    enabled: bool = True
    allowed_root: Path = field(default_factory=default_allowed_root)
    max_tool_calls: int = 8
    max_model_rounds: int = 10
    max_model_retries: int = 4          # 瞬态模型错误的额外重试预算(按整次运行计,不随轮次重置)
    active_timeout_seconds: int = 120
    approval_timeout_seconds: int = 1800   # 等待审批的硬上限(不计入活动时间),防止无限挂起
    retention_days: int = 30
    max_output_tokens: int = 8192
    # 流式反馈。开启后 Agent 运行期间正文增量实时上屏;
    # 端点不支持时会自动退回非流式(见 DeepSeekAgentClient._call_api_streaming)。
    stream: bool = True
    extra_tools_enabled: bool = False   # open_item / system_info(默认关闭,需用户显式开启)
    # "网页读取"工具(web_fetch):默认**关闭**,开启后每次调用仍需逐次审批。
    # 默认关闭 = 不改变既有工具范围;开启方式见 api_config.example.json。
    web_tools_enabled: bool = False
    web_timeout_seconds: int = 10
    # 关键词搜索(web_search):**默认关闭**,开启后由主人决定用哪个后端 ——
    # 实现不替主人选服务商。
    #
    # 后端是 **ddgs 库**(聚合多个搜索引擎),不接受
    # "填一个搜索 API 的 URL":受限链路上只需一个代理即可自动在多个
    # 引擎间取舍;而逐个接 API 既要密钥、又各自绑死一家服务。因此
    # web_search_url / web_search_api_key / KURUMI_WEB_SEARCH_KEY 不再存在:
    # 配置里写了会被静默忽略(不是错误)。
    web_search_enabled: bool = False
    web_search_proxy: str = ""              # 例 http://127.0.0.1:10809;socks5:// 也支持;空=直连
    web_search_engine: str = "auto"         # auto 会在可用引擎间自动取舍
    web_search_timeout_seconds: int = 15    # 库内部为每个引擎设的超时(秒)
    web_search_max_results: int = 5         # 模型没给 count 时的默认条数(1..10)
    web_search_allow_private: bool = False  # 仅作用于 web_fetch 的私网开关
    # 允许根之外的变更一律 DENY,没有例外开关:要写到别处请改 allowed_root。
    # 注:allow_outside_root_mutations 不是可用开关,执行层一律拒绝越根变更。

    def __repr__(self):
        return _safe_repr(self, dataclasses.fields(self))

    def __post_init__(self):
        """**构造即清理**:代理解析不出凭据,不管它是从配置读来的还是代码里传的。

        只在 `load_config()` 里清理是不够的 —— 直接
        `AgentSettings(web_search_proxy="http://alice:pw@host")` 时 `repr` 仍然带密码,
        而解析器多半由代码构造(UI/调试脚本)。frozen dataclass 要用 object.__setattr__。
        """
        cleaned = _sanitize_proxy(self.web_search_proxy)
        if cleaned != self.web_search_proxy:
            object.__setattr__(self, "web_search_proxy", cleaned)


@dataclass(frozen=True, repr=False)
class ApiSettings:
    base_url: str = "https://api.deepseek.com"
    chat_model: str = ""
    agent_model: str = ""
    credential_id: str = "kurumi-deepseek"

    def __repr__(self):
        return (
            f"ApiSettings(base_url={self.base_url!r}, chat_model={self.chat_model!r}, "
            f"agent_model={self.agent_model!r}, credential_id={self.credential_id!r})"
        )


@dataclass(frozen=True)
class ConfigDocument:
    api: ApiSettings
    agent: AgentSettings
    has_legacy_secret: bool = False
    allowed_hosts: tuple = ()
    raw: dict = field(default_factory=dict, repr=False)


def _pos_int(value, default):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


# ddgs/primp 实际支持的代理协议;写错的协议(如 `htp://`)在这里就被挡下,
# 而不是等到发请求时给出一个难以理解的库异常。
_PROXY_SCHEMES = ("http", "https", "socks4", "socks4a", "socks5", "socks5h")

# 拒绝痕迹的前缀:见到它就说明"这个值已经被清理过了",不能再解析一遍。
_REJECTED_PREFIX = "[已拒绝的代理地址"


def _rejected_proxy(reason: str, scheme: str = "", host: str = "", port=None) -> str:
    """拼一个"给人看的拒绝痕迹":只带**脱敏后的**协议/主机/端口,绝不回显原串。

    拒绝痕迹里**绝不能出现原串**:把带凭据的地址截断后写进痕迹,会让"防泄漏"的函数
    自己泄漏密码 —— 痕迹会一路进日志、UI 与审计库。所以这里一律只接受已经拆好的部件:
    拿不到主机时就不回显任何地址。
    """
    where = f":{scheme}://{host}" if host else ""
    if host and port:
        where += f":{port}"
    return f"[已拒绝的代理地址{where}]({reason})"


def _sanitize_proxy(raw) -> str:
    """代理地址:只接受"协议://主机[:端口]",**拒绝**任何 userinfo。

    带认证的代理形如 `http://alice:s3cret@host:port`。它会被 repr、超时提示、
    工具结果一路带进模型回填 / 事件 / UI / TaskResult / SQLite 审计库。
    与其到处脱敏,不如在入口就拒绝:代理凭据应当走凭据管理器,不该出现在配置文件里。

    拒绝时**保留一个可诊断的残留值**(脱敏后的协议+主机),这样主人一看设置项就知道
    "这里有个被拒的地址",而不是被静默清空后百思不得其解。

    这个函数**必须不抛异常**:它跑在配置解析路径上,而 UI 把 `load_config()`
    的任何异常都当成"Agent 配置读取失败"并把整个 Agent 关掉 —— 也就是说,主人把端口
    写成 `abc`,代价会是"Agent 功能整个没了",而原因与 Agent 能力毫无关系(`parsed.port`
    在端口非法时抛异常:`_sanitize_proxy("http://alice:pw@host:abc")` 会抛 ValueError)。
    所以下面每一步都自带兜底,并且要**主动校验端口合法性**,
    不能让 `http://127.0.0.1:abc` / `:99999` 这种地址原样过关后被交给网络库。
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith(_REJECTED_PREFIX):
        # 已经是"拒绝痕迹"就原样保留。否则它会**被二次解析**:痕迹长这样
        # `[已拒绝的代理地址:http://127.0.0.1](端口非法:…)`,再跑一遍这个函数会得到
        # "缺少协议或主机名" —— 真正的原因被改写掉,主人看到的是另一句话
        # (`load_config` 与 `AgentSettings.__post_init__` 各清理一次,同一个值会两次经过这里)。
        return text
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return _rejected_proxy("格式无法解析")
    scheme = (parsed.scheme or "").lower()
    host = parsed.hostname or ""
    has_userinfo = bool(parsed.username or parsed.password)
    try:
        port = parsed.port                  # 端口非法时抛 ValueError
        port_bad = False
    except ValueError:
        port, port_bad = None, True
    if not scheme or not host:
        return _rejected_proxy("缺少协议或主机名")
    if has_userinfo:
        # 凭据优先报:这是安全原因,哪怕端口同时也写错了
        return _rejected_proxy(
            "代理地址里不能带用户名/密码:请把凭据交给 Windows 凭据管理器,"
            "配置里只写 协议://主机:端口", scheme, host, port)
    if scheme not in _PROXY_SCHEMES:
        return _rejected_proxy(f"不支持的协议 {scheme}:只支持 http/https/socks5/socks5h",
                               scheme, host, port)
    if port_bad:
        return _rejected_proxy("端口非法:只接受 1-65535 的数字,或省略端口", scheme, host)
    return text


def _as_bool(value, default: bool) -> bool:
    """严格布尔解析:只接受真布尔或显式 true/false 字符串,其余回退默认值。

    直接把字符串丢给 bool() 会让 "false" 变成 True,安全开关被静默反转(fail-open)。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
    return default


def load_config(path: Path) -> ConfigDocument:
    """读取配置文件;结构非法时安全降级,不抛异常也不扩大权限。"""
    data: dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded
    except (OSError, ValueError):
        data = {}
    has_legacy = bool(str(data.get("api_key") or "").strip())

    api = ApiSettings(
        base_url=str(data.get("base_url") or "https://api.deepseek.com"),
        chat_model=str(data.get("chat_model") or data.get("model") or "deepseek-chat"),
        agent_model=str(data.get("agent_model") or "deepseek-v4-flash"),
        credential_id=str(data.get("credential_id") or "kurumi-deepseek"),
    )

    agent_raw = data.get("agent") if isinstance(data.get("agent"), dict) else {}
    allowed_root = Path(str(agent_raw.get("allowed_root") or default_allowed_root()))
    if not allowed_root.is_absolute():
        # 拒绝相对允许根:回退到当前用户桌面(始终绝对)
        allowed_root = default_allowed_root()
    agent = AgentSettings(
        enabled=_as_bool(agent_raw.get("enabled"), True),
        allowed_root=allowed_root,
        max_tool_calls=_pos_int(agent_raw.get("max_tool_calls"), 8),
        max_model_rounds=_pos_int(agent_raw.get("max_model_rounds"), 10),
        max_model_retries=_pos_int(agent_raw.get("max_model_retries"), 4),
        active_timeout_seconds=_pos_int(agent_raw.get("active_timeout_seconds"), 120),
        approval_timeout_seconds=_pos_int(agent_raw.get("approval_timeout_seconds"), 1800),
        retention_days=_pos_int(agent_raw.get("retention_days"), 30),
        max_output_tokens=_pos_int(agent_raw.get("max_output_tokens"), 8192),
        stream=_as_bool(agent_raw.get("stream"), True),
        extra_tools_enabled=_as_bool(agent_raw.get("extra_tools_enabled"), False),
        web_tools_enabled=_as_bool(agent_raw.get("web_tools_enabled"), False),
        web_timeout_seconds=_pos_int(agent_raw.get("web_timeout_seconds"), 10),
        web_search_enabled=_as_bool(agent_raw.get("web_search_enabled"), False),
        # 代理地址允许环境变量覆盖(与密钥同一个思路:不必写进文件)
        web_search_proxy=_sanitize_proxy(os.environ.get("KURUMI_SEARCH_PROXY")
                                         or agent_raw.get("web_search_proxy") or ""),
        web_search_engine=str(agent_raw.get("web_search_engine") or "auto").strip() or "auto",
        web_search_timeout_seconds=_pos_int(agent_raw.get("web_search_timeout_seconds"), 15),
        web_search_max_results=min(10, _pos_int(agent_raw.get("web_search_max_results"), 5)),
        web_search_allow_private=_as_bool(agent_raw.get("web_search_allow_private"), False),
        # allow_outside_root_mutations 不是可用开关:执行层一律拒绝越根变更,
        # 配置里若还有这个键,会被静默忽略(不是错误)。
        # 同理 web_search_url / web_search_api_key:搜索后端现在是 ddgs 库,不吃自填 URL。
    )
    hosts = data.get("allowed_hosts")
    allowed_hosts = tuple(str(h) for h in hosts) if isinstance(hosts, list) else ()
    return ConfigDocument(api=api, agent=agent, has_legacy_secret=has_legacy, allowed_hosts=allowed_hosts, raw=data)
