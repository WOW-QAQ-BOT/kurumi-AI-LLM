# -*- coding: utf-8 -*-
"""路径规范化与权限策略:执行层强制,不依赖提示词。"""
import hashlib
import json
import os
import re
import stat
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.types import PolicyDecision, ToolCall

_SENSITIVE_NAMES = (".env", ".npmrc", ".pypirc", ".netrc", "known_hosts")
# 应用自身/常见凭据载体:明文 API Key 常被直接写进这些文件,
# 而它们恰好落在默认允许根(桌面)里 → 不列名单就是零审批可读。
# 注:普通 config.json 故意不放进"无条件敏感"名单——桌面上的 config.json 绝大多数是
# 无关应用的普通配置,一律标记会造成审批疲劳(主人会习惯性点"允许",反而更不安全);
# 它走下面的内容级探测:真含 api_key/password/secret 时依然会被拦成 ASK。
_SENSITIVE_APP_NAMES = ("api_config.json", "api_config.local.json", "secrets.json",
                        "credentials.json", ".git-credentials", ".dockercfg")
# 名字**模式**名单:只做精确相等匹配会漏得彻底(.env.local/.env.production/id_rsa.bak
# 这类副本名全部漏网,连内容探测都因为扩展名不在白名单而跳过 → 零审批读出明文 Key)。
# 因此这里按前缀/模式匹配,并对 .bak/.old/.txt 之类的副本后缀做二次判断。
_SENSITIVE_NAME_PATTERNS = (
    re.compile(r"^\.?env(\..+)?$"),                                  # .env / .env.local / .env.production / env.bak
    re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)(\..*)?$"),              # id_rsa / id_rsa.bak / id_ed25519.pub
    re.compile(r"^\.?(git|npm|pypi|net)rc(\..+)?$"),
    re.compile(r"^(known_hosts|authorized_keys)(\..+)?$"),
)
# 副本/备份后缀:命中名单的名字被加一层后缀后仍要判敏感(id_rsa.bak、.env.production.old)
_COPY_SUFFIXES = (".bak", ".old", ".orig", ".save", ".copy", ".backup", ".txt", ".swp", ".tmp")
_SENSITIVE_EXTENSIONS = (".pem", ".key", ".crt", ".cer", ".p12", ".pfx", ".kdbx", ".ovpn")
_SENSITIVE_HINTS = ("secret", "credential", "cookie", "token", "password", "api_key", "apikey", "auth",
                    "秘钥", "密钥", "密码", "口令", "凭证", "凭据", "令牌")
_SENSITIVE_PARENTS = (".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker")

# 内容级敏感探测:名字没命中时,读文件头部匹配密钥特征(仅在 file_read 的审批判定里使用)
#
# 窗口必须覆盖"单次 file_read 能返回的最大字节数"(tools/files.MAX_READ_BYTES=256 KiB):
# 窗口小于该值时,只要密钥排在窗口之后(例如一个前 4 KiB 全是填充的普通 notes.txt),
# 探测就返回 False、assess=ALLOW,密钥被**零审批**读出并送往云端。
# 这里不限定扩展名白名单(那会让 .env.local 这类文件连探测都不做),改为按二进制
# 特征跳过非文本文件。
_SNIFF_BYTES = 256 * 1024
_SECRET_CONTENT_RES = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),                        # OpenAI/DeepSeek 风格明文 Key
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),        # PEM 私钥
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"(?i)\b(api[_-]?key|apikey|secret|password|passwd|token|credential|"
               r"authorization|bearer|private[_-]?key)\b"),
    re.compile("密钥|密码|口令|令牌|凭据|凭证"),
)
_DEVICE_PREFIXES = ("\\\\.\\", "\\\\?\\", "//./", "//?/")
_WILDCARD_CHARS = ("*", "?")
# open_item 命中这些扩展名时会真的执行代码(os.startfile 走系统文件关联)
_EXECUTABLE_EXTENSIONS = (".exe", ".com", ".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe",
                          ".js", ".jse", ".wsf", ".wsh", ".msi", ".msp", ".scr", ".cpl",
                          ".hta", ".reg", ".lnk", ".url", ".jar")


def _name_variants(name: str):
    """名字及其"去掉副本后缀"的变体:让 id_rsa.bak / .env.production.old 也能被模式命中。"""
    yield name
    stem = name
    for _ in range(2):        # 允许两层后缀(.env.production.old)
        for suffix in _COPY_SUFFIXES:
            if stem.endswith(suffix) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
                yield stem
                break


def _is_sensitive_path(p: Path) -> bool:
    name = p.name.lower()
    if name in _SENSITIVE_NAMES or name in _SENSITIVE_APP_NAMES:
        return True
    if any(rx.match(variant) for variant in _name_variants(name)
           for rx in _SENSITIVE_NAME_PATTERNS):
        return True
    if any(name.endswith(ext) for ext in _SENSITIVE_EXTENSIONS):
        return True
    if any(h in name for h in _SENSITIVE_HINTS):
        return True
    parts = [seg.lower() for seg in p.parts]
    return any(seg in _SENSITIVE_PARENTS for seg in parts)


def _has_other_links(p: Path) -> bool:
    """文件是否还有别的硬链接(无法判断那些链接在不在允许根内)。"""
    try:
        st = os.stat(p)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and st.st_nlink > 1


# 为覆盖 start_line 而允许额外向前扫描的字节上限。超过它就承认"覆盖不到",
# 交给调用方按 fail-closed(需审批)处理,而不是无界扫描整个大文件。
_SNIFF_SCAN_BUDGET = 4 * 1024 * 1024


def _match_secret(head: bytes) -> bool:
    """在一段字节里匹配密钥特征;二进制(含 NUL)直接跳过。"""
    if not head:
        return False
    if b"\x00" in head:
        return False          # 二进制:不是文本密钥载体,匹配只会制造噪音
    try:
        text = head.decode("utf-8")
    except UnicodeError:
        text = head.decode("utf-8", errors="ignore")   # GBK 等编码:仍尝试匹配,匹配不上不误判
    return any(rx.search(text) for rx in _SECRET_CONTENT_RES)


def _segment_looks_sensitive(p: Path, start_line: int = 1):
    """探测 **file_read 实际会返回的那一段**是否含密钥。

    返回 True/False;返回 None 表示"没能在扫描预算内定位到该片段"(调用方按未确认处理)。

    约束:`file_read` 支持 `start_line`,可跳到文件任意位置再读 256 KiB,
    而内容探测只看文件头部 —— 于是"秘密放在第 N 行之后"就是一条零审批外发通道。
    这里按 `start_line` 定位到真实片段再匹配:文件不超过探测窗口时与头部探测等价,
    因此常规读取没有额外开销。

    行尾语义:LF 与 CRLF 下的行号与 file_read 一致;若文件只有裸 `\\r` 分行,
    二进制计数会把整份文件当成一行 → 超出预算 → 返回 None → 调用方按需审批(fail-closed)。
    """
    try:
        st = os.stat(p)
    except OSError:
        return True                     # 读不到就 fail-safe(按敏感处理)
    if st.st_size <= _SNIFF_BYTES:
        # 整个文件都在单次读取窗口内:头部探测即覆盖全部内容
        return _content_looks_sensitive(p)
    if start_line <= 1:
        # 从第一行开始读:返回片段就在文件头部
        return _content_looks_sensitive(p)
    try:
        with open(p, "rb") as f:
            consumed = 0
            line_no = 1
            # 逐行跳过到 start_line;行号语义与 tools/files.py 的 file_read 一致(1 基)
            while line_no < start_line:
                line = f.readline()
                if not line:
                    return False        # 越过了文件末尾:实际返回空片段,无内容可泄露
                consumed += len(line)
                if consumed > _SNIFF_SCAN_BUDGET:
                    return None         # 超出预算:不假装安全
                line_no += 1
            return _match_secret(f.read(_SNIFF_BYTES))
    except OSError:
        return True                     # 读取失败 fail-safe


def _content_looks_sensitive(p: Path) -> bool:
    """内容级敏感探测:名字没命中时,读文件头部判断是否含明文密钥/口令。

    只在 `PermissionPolicy._assess_read` 里调用(路径已确认在允许根内且是普通文件)。
    探测窗口 = `_SNIFF_BYTES`,必须覆盖单次 `file_read` 能返回的最大字节数,否则
    "密钥藏在窗口之后"就是一条零审批外发通道(取值依据见 _SNIFF_BYTES 注释)。
    非文本文件(头部含 NUL)直接跳过,避免对二进制做无意义匹配。读取失败按"敏感"
    处理(fail-safe):探测本身出错时宁可多要一次审批,也不能静默放行。
    """
    try:
        with open(p, "rb") as f:
            head = f.read(_SNIFF_BYTES)
    except OSError:
        return True
    return _match_secret(head)


def _start_line_of(call: ToolCall) -> int:
    """读取起始行(与 tools/files.py 的归一化语义一致:至少 1)。"""
    try:
        return max(1, int(call.arguments.get("start_line", 1)))
    except (TypeError, ValueError):
        return 1


def _is_within(candidate: str, root: str) -> bool:
    """大小写不敏感的包含判断;不使用字符串前缀。"""
    left = os.path.normcase(os.path.abspath(candidate))
    right = os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath([left, right]) == right
    except ValueError:
        return False


def hash_arguments(arguments: dict) -> str:
    canonical = json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_FINGERPRINT_FULL_MAX_BYTES = 8 * 1024 * 1024    # 超过此大小改算部分哈希,避免桌面上的大文件卡住 UI
_FINGERPRINT_PARTIAL_BYTES = 4 * 1024 * 1024     # 大文件：头部与尾部各取这么多字节


def _iter_head_and_tail(f, size: int, span: int = _FINGERPRINT_PARTIAL_BYTES):
    """依次产出文件头部 span 字节与尾部 span 字节(仅供大文件部分哈希使用)。"""
    f.seek(0)
    remaining = span
    while remaining > 0:
        chunk = f.read(min(65536, remaining))
        if not chunk:
            return
        remaining -= len(chunk)
        yield chunk
    tail_start = max(span, size - span)          # size > 2*span 时不会与头部重叠
    f.seek(tail_start)
    remaining = size - tail_start
    while remaining > 0:
        chunk = f.read(min(65536, remaining))
        if not chunk:
            return
        remaining -= len(chunk)
        yield chunk


def fingerprint(path: Path) -> dict | None:
    """现有普通文件的内容指纹(大小、mtime、sha256);新目标返回 None。

    小文件(< 8 MiB)行为不变:整文件 SHA-256。
    大文件取**头部 4 MiB + 尾部 4 MiB** 两段拼接后哈希,并标注 `partial` / `hashed_bytes`:
    审批前与执行前各算一次,几 GB 的视频若全量哈希会让 UI 冻结两次。

    为什么必须带上尾部:只哈希头部时,"审批后改写文件尾部 + 用 os.utime 恢复 mtime"
    会让指纹**完全相同**,从而骗过 `execution_recheck()` 的"文件是否被换掉"复查 ——
    改动落在头部窗口之外的字节上时,部分哈希根本看不见。
    加上尾部后,这类改动会被发现;真要大文件全量校验请显式比对整文件哈希。
    """
    try:
        st = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    partial = st.st_size > _FINGERPRINT_FULL_MAX_BYTES
    h = hashlib.sha256()
    hashed = 0
    try:
        with open(path, "rb") as f:
            if partial:
                for chunk in _iter_head_and_tail(f, st.st_size):
                    h.update(chunk)
                    hashed += len(chunk)
            else:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
                    hashed += len(chunk)
    except OSError:
        return None
    info = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h.hexdigest()}
    if partial:
        info["partial"] = True
        info["hashed_bytes"] = hashed
    return info


@dataclass(frozen=True)
class PathInfo:
    original: str
    canonical: Path
    exists: bool
    within_allowed_root: bool


@dataclass(frozen=True)
class PolicyAssessment:
    decision: PolicyDecision
    reason: str
    canonical_arguments: dict[str, Any]
    risk_summary: str
    data_leaves_device: bool
    file_fingerprint: dict | None = None


class PathResolver:
    """Windows 路径规范化:绝对化、解析符号链接/junction、拒绝危险路径形态。"""

    def __init__(self, allowed_root: Path):
        self.allowed_root = Path(allowed_root)

    def canonicalize(self, path: str, for_creation: bool = False) -> PathInfo:
        original = path or ""
        if "\x00" in original:
            raise ValueError("路径包含非法字符")
        stripped = original.strip()
        upper = stripped.upper()
        for prefix in _DEVICE_PREFIXES:
            if upper.startswith(prefix.upper()):
                raise ValueError("设备/UNC 前缀路径被拒绝")
        # 裸 UNC(\\host\share、//host/share)一律拒绝:否则会向任意主机发起 SMB 认证
        # (NTLM 外泄面),open_item 还能直接执行远端程序
        if stripped.startswith(("\\\\", "//")):
            raise ValueError("UNC/网络路径被拒绝")
        p = Path(stripped)
        # Win32 打开文件时会静默吃掉路径分量结尾的空格与点("x.txt." 实际写的是 "x.txt"),
        # 而审批卡展示的是 canonicalize 的结果 → 主人批准的名字与实际被写的文件可能不同(别名歧义)。
        # "." 与 ".." 是合法的相对导航分量,不在拒绝范围内;".env" 这类以点开头的名字也合法(只看结尾)。
        for part in p.parts:
            if part in (".", "..") or part.endswith(("\\", "/")):
                continue
            if part.rstrip(" .") != part:
                raise ValueError("路径分量以空格/点结尾,存在 Win32 别名歧义,已拒绝")
        if not p.is_absolute():
            p = self.allowed_root / p
        try:
            if for_creation:
                parent = (p.parent if p.parent != Path("") else self.allowed_root).resolve(strict=False)
                canonical = parent / p.name
                exists = False
            else:
                canonical = p.resolve(strict=False)
                exists = canonical.exists()
        except OSError as e:
            # 受保护/无权限的路径无法解析:按拒绝处理,绝不升级为内部错误。
            # `from None` 抑制链式回溯:原始 OSError 只是原因,对调用方有意义的是
            # "这个路径被拒绝"这一结论(否则日志里会出现令人误解的 OSError)。
            raise ValueError(f"路径解析失败(权限不足): {e}") from None
        s = str(canonical)
        if ":" in s[2:]:
            raise ValueError("NTFS 备用数据流路径被拒绝")
        if any(ch in p.name for ch in _WILDCARD_CHARS):
            raise ValueError("通配符路径被拒绝")
        return PathInfo(
            original=original,
            canonical=canonical,
            exists=exists,
            within_allowed_root=_is_within(str(canonical), str(self.allowed_root)),
        )


class PermissionPolicy:
    """按规格矩阵评估每次工具调用:ALLOW / ASK / DENY。"""

    _READ_META = ("file_list", "file_search")
    _READ_CONTENT = ("file_read",)
    _MUTATE = ("file_write", "file_patch", "file_move", "file_trash")
    _READ_SYS = ("system_info",)
    _OPEN = ("open_item",)
    _WEB = ("web_fetch",)        # 读取网页(逐次审批、拒私网)
    _WEB_SEARCH = ("web_search",)   # 关键词搜索(逐次审批;后端由主人配置)

    def __init__(self, settings):
        self.settings = settings
        self.resolver = PathResolver(settings.allowed_root)

    def assess(self, call: ToolCall) -> PolicyAssessment:
        """对外入口:任何内部意外异常都 fail-closed 成 DENY,绝不向外抛。

        策略层抛错会穿透到 runner 的兜底 `except Exception`,把整个 run 变成
        INTERNAL_ERROR(任务直接失败);而"评估失败"在安全语义上等价于"不确定 → 拒绝"。
        """
        try:
            return self._assess(call)
        except Exception as e:
            return PolicyAssessment(
                decision=PolicyDecision.DENY,
                reason=f"策略评估异常,已按最保守方式拒绝: {type(e).__name__}: {e}",
                canonical_arguments={},
                risk_summary="策略评估异常",
                data_leaves_device=False,
            )

    def _assess(self, call: ToolCall) -> PolicyAssessment:
        if call.name in self._READ_META:
            return self._assess_meta(call)
        if call.name in self._READ_CONTENT:
            return self._assess_read(call)
        if call.name in self._MUTATE:
            return self._assess_mutate(call)
        if call.name in self._READ_SYS:
            return self._assess_sysinfo(call)
        if call.name in self._OPEN:
            return self._assess_open(call)
        if call.name in self._WEB:
            return self._assess_web(call)
        if call.name in self._WEB_SEARCH:
            return self._assess_web_search(call)
        return PolicyAssessment(
            decision=PolicyDecision.DENY,
            reason=f"未注册工具: {call.name}",
            canonical_arguments={},
            risk_summary="未知工具",
            data_leaves_device=False,
        )

    def _assess_sysinfo(self, call: ToolCall) -> PolicyAssessment:
        canonical = dict(call.arguments)
        if str(call.arguments.get("resource") or "") == "processes":
            # 进程清单(名字+PID+内存)等于本机"已安装/正在运行的软件资产画像",
            # 上传云端就是一次零成本的软件指纹采集,必须逐次批准。
            return PolicyAssessment(
                decision=PolicyDecision.ASK,
                reason="读取进程清单需批准",
                canonical_arguments=canonical,
                risk_summary="进程清单会发送至 DeepSeek 云端(可推断本机软件资产画像),请确认后再批准",
                data_leaves_device=True,
            )
        return PolicyAssessment(
            decision=PolicyDecision.ALLOW,
            reason="只读系统信息",
            canonical_arguments=canonical,
            risk_summary="只读系统信息,结果将发送至 DeepSeek 云端",
            data_leaves_device=True,
        )

    def _assess_web_search(self, call: ToolCall) -> PolicyAssessment:
        """关键词搜索:一律逐次审批。

        与 web_fetch 的区别:这里外发的是**关键词**而非网页地址 —— 关键词本身可能包含
        主人的隐私(姓名、文件名、病名……),所以审批卡必须点明"关键词会发给搜索服务",
        而不是笼统地说"联网搜索"。
        """
        canonical = dict(call.arguments)
        query = str(canonical.get("query") or "").strip()
        return PolicyAssessment(
            decision=PolicyDecision.ASK,
            reason="联网搜索需批准",
            canonical_arguments=canonical,
            risk_summary=(f"将把关键词「{query[:60]}」发送给搜索服务,并把结果摘要读入本次任务;"
                          "结果内容不可信,可能包含针对模型的指令"),
            data_leaves_device=True,
        )

    def _assess_web(self, call: ToolCall) -> PolicyAssessment:
        """网页读取:一律逐次审批,并**如实**标注"目标地址会发到外部站点"。

        为什么必须 ASK:这是唯一一个"把主人给出的地址发出去、再把外部内容读回来"的工具 ——
        既是数据外发面(URL 本身可能含敏感参数),也是提示注入的入口
        (网页内容会进入模型上下文)。默认关闭由 `agent.web_tools_enabled` 控制,
        这里只负责"开着的时候也必须逐次批准"。
        """
        canonical = dict(call.arguments)
        url = str(canonical.get("url") or "").strip()
        host = ""
        try:
            host = urllib.parse.urlsplit(url).hostname or ""
        except ValueError:
            host = ""
        return PolicyAssessment(
            decision=PolicyDecision.ASK,
            reason="读取外部网页需批准",
            canonical_arguments=canonical,
            risk_summary=(f"将访问外部站点 {host or url[:60]}(该地址会被发送出去),"
                          "并把网页正文读入本次任务 —— 网页内容不可信,可能包含针对模型的指令"),
            data_leaves_device=True,
        )

    def _assess_open(self, call: ToolCall) -> PolicyAssessment:
        target = str(call.arguments.get("path_or_url") or "")
        canonical = dict(call.arguments)
        lowered = target.lower()
        if lowered.startswith(("http://", "https://")):
            return PolicyAssessment(
                decision=PolicyDecision.ASK,
                reason="打开网址需批准",
                canonical_arguments=canonical,
                risk_summary="用默认浏览器打开网址",
                data_leaves_device=False,
            )
        # 非 http(s) 协议一律拒绝(file://、ftp://、javascript: 等)
        if "://" in lowered or lowered.startswith(("file:", "ftp:", "javascript:", "data:", "vbscript:")):
            return self._deny_on_error(ValueError(f"不支持的协议: {target[:20]}"))
        # 其余按本地路径处理,拒绝危险形态(设备路径、通配符等)
        try:
            info = self.resolver.canonicalize(target)
        except ValueError as e:
            return self._deny_on_error(e)
        canonical["path_or_url"] = str(info.canonical)
        fp = fingerprint(info.canonical) if info.exists else None
        # 可执行/脚本类扩展名会真的运行代码,审批卡必须如实说明,不能让主人以为只是"打开文件"
        if info.canonical.suffix.lower() in _EXECUTABLE_EXTENSIONS:
            return PolicyAssessment(
                decision=PolicyDecision.ASK,
                reason=f"将以可执行方式运行 {info.canonical.suffix.lower()}",
                canonical_arguments=canonical,
                risk_summary="⚠️ 该操作会执行本机代码(可能取得控制权),请确认程序来源可信",
                data_leaves_device=False,
                file_fingerprint=fp,
            )
        return PolicyAssessment(
            decision=PolicyDecision.ASK,
            reason="启动应用/打开文件需批准",
            canonical_arguments=canonical,
            risk_summary="启动应用或打开文件/文件夹",
            data_leaves_device=False,
            file_fingerprint=fp,
        )

    def _deny_on_error(self, err: Exception) -> PolicyAssessment:
        return PolicyAssessment(
            decision=PolicyDecision.DENY,
            reason=f"路径校验失败: {err}",
            canonical_arguments={},
            risk_summary="危险或非法路径",
            data_leaves_device=False,
        )

    def _assess_meta(self, call: ToolCall) -> PolicyAssessment:
        try:
            info = self.resolver.canonicalize(call.arguments.get("path", ""))
        except ValueError as e:
            return self._deny_on_error(e)
        canonical = dict(call.arguments)
        canonical["path"] = str(info.canonical)
        # file_search 带 query 时会读取目录内文件正文做匹配(见 tools/files.py),
        # 因此必须与 file_read 同级看待:敏感目录/根外一律需要批准并标注数据外发。
        if call.name == "file_search" and str(call.arguments.get("query") or "").strip():
            if _is_sensitive_path(info.canonical):
                return PolicyAssessment(PolicyDecision.ASK, "疑似敏感目录下的内容检索", canonical,
                                        "内容检索会读取目录内文件正文(含疑似敏感文件),内容将发送至 DeepSeek 云端", True)
            if not info.within_allowed_root:
                return PolicyAssessment(PolicyDecision.ASK, "允许根目录之外的内容检索", canonical,
                                        "目录外内容检索,内容将发送至 DeepSeek 云端", True)
            return PolicyAssessment(PolicyDecision.ASK, "内容检索需批准(会读取文件正文)", canonical,
                                    "内容检索会读取目录内文件正文,内容将发送至 DeepSeek 云端", True)
        if info.within_allowed_root:
            return PolicyAssessment(PolicyDecision.ALLOW, "允许根内目录列举/搜索", canonical,
                                    "只读元数据", False)
        return PolicyAssessment(PolicyDecision.ASK, "允许根目录之外的列举/搜索", canonical,
                                "目录外访问", False)

    def _assess_read(self, call: ToolCall) -> PolicyAssessment:
        try:
            info = self.resolver.canonicalize(call.arguments.get("path", ""))
        except ValueError as e:
            return self._deny_on_error(e)
        canonical = dict(call.arguments)
        canonical["path"] = str(info.canonical)
        sensitive = _is_sensitive_path(info.canonical)
        # 名字没命中时再做内容级探测:只在"允许根内、真实存在的普通文件"上做,
        # 根外/不存在的目标本来就要求审批,不必多付一次读盘开销。
        # 关键:按 **实际会返回的片段** 探测(file_read 的 start_line 能跳到文件任意位置),
        # 覆盖不到时 fail-closed 交审批,绝不因为"头部干净"就放行整段读取。
        segment_unverified = False
        if not sensitive and info.within_allowed_root and info.canonical.is_file():
            verdict = _segment_looks_sensitive(info.canonical, _start_line_of(call))
            if verdict is None:
                segment_unverified = True
            else:
                sensitive = verdict
        fp = fingerprint(info.canonical) if info.exists else None
        if sensitive:
            return PolicyAssessment(PolicyDecision.ASK, "疑似敏感文件", canonical,
                                    "疑似敏感文件(密钥/凭据),内容将发送至 DeepSeek 云端", True, fp)
        if segment_unverified:
            return PolicyAssessment(
                PolicyDecision.ASK, "读取片段超出内容探测范围", canonical,
                f"文件较大且本次从第 {_start_line_of(call)} 行开始读取,"
                "该片段无法在安全预算内完成密钥检测,需主人确认;内容将发送至 DeepSeek 云端",
                True, fp)
        # 硬链接:路径文本与 resolve() 都看不出它指向根外(硬链接不是重解析点),
        # 于是根内一个指向根外文件的硬链接能零审批读到根外内容。
        # st_nlink>1 无法判断"另一个链接在哪",按不可信处理 → 逐次审批。
        if info.within_allowed_root and _has_other_links(info.canonical):
            return PolicyAssessment(PolicyDecision.ASK, "文件存在硬链接,归属不可判定", canonical,
                                    "该文件还有其它硬链接(可能指向允许根之外),内容将发送至 DeepSeek 云端",
                                    True, fp)
        if not info.within_allowed_root:
            return PolicyAssessment(PolicyDecision.ASK, "允许根目录之外的读取", canonical,
                                    "目录外读取,内容将发送至 DeepSeek 云端", True, fp)
        return PolicyAssessment(PolicyDecision.ALLOW, "允许根内普通文本读取", canonical,
                                "读取后内容将发送至 DeepSeek 云端", True, fp)

    def _assess_mutate(self, call: ToolCall) -> PolicyAssessment:
        canonical = dict(call.arguments)
        targets: list[Path] = []
        try:
            if call.name == "file_move":
                src = self.resolver.canonicalize(call.arguments.get("source", ""))
                dst = self.resolver.canonicalize(call.arguments.get("destination", ""), for_creation=True)
                canonical["source"] = str(src.canonical)
                canonical["destination"] = str(dst.canonical)
                fp = fingerprint(src.canonical) if src.exists else None
                if not src.exists:
                    raise ValueError("源文件不存在")
                if self._is_protected_target(dst.canonical):
                    raise ValueError("目标为受保护位置(盘根/允许根/通配符)")
                targets = [src.canonical, dst.canonical]
                reason, summary = "移动/重命名需批准", "移动/重命名文件"
            elif call.name == "file_trash":
                info = self.resolver.canonicalize(call.arguments.get("path", ""))
                canonical["path"] = str(info.canonical)
                fp = fingerprint(info.canonical) if info.exists else None
                if not info.exists:
                    raise ValueError("目标不存在")
                if self._is_protected_target(info.canonical):
                    raise ValueError("禁止回收允许根或磁盘根")
                targets = [info.canonical]
                reason, summary = "回收需批准", "将文件移入回收站"
            else:
                # file_write / file_patch
                info = self.resolver.canonicalize(call.arguments.get("path", ""), for_creation=True)
                canonical["path"] = str(info.canonical)
                fp = fingerprint(info.canonical) if info.canonical.exists() else None
                mode = call.arguments.get("mode", "")
                op = "新建文件" if mode == "create" else ("覆盖文件" if mode == "overwrite" else "修改文件")
                targets = [info.canonical]
                reason, summary = "文件修改需批准", op + ",内容将发送至 DeepSeek 云端"
        except ValueError as e:
            return self._deny_on_error(e)

        # 允许根之外的变更:一律拒绝,不提供任何"越根放行"开关。
        #
        # 越根放行开关在架构上必然失效:策略层即便放行(ASK→主人批准),执行器的
        # `_atomic_write` 仍以"父目录越出允许根"中止,文件根本写不出去 ——
        # 主人批准了却什么都没发生。与其提供一个永远失败的选项,不如明确不支持:
        # 要写到别处,请把 allowed_root 改对。
        outside = [t for t in targets if not _is_within(str(t), str(self.settings.allowed_root))]
        if outside:
            return PolicyAssessment(
                decision=PolicyDecision.DENY,
                reason=f"允许根之外禁止变更: {outside[0]}",
                canonical_arguments=canonical,
                risk_summary="越界变更已拒绝(如需允许,请把 api_config.json 的 "
                             "agent.allowed_root 指向正确目录)",
                data_leaves_device=False,
                file_fingerprint=fp,
            )
        return PolicyAssessment(PolicyDecision.ASK, reason, canonical, summary, True, fp)

    def _is_protected_target(self, target: Path) -> bool:
        t = os.path.normcase(str(target))
        root = os.path.normcase(str(self.settings.allowed_root))
        drive = os.path.splitdrive(t)[0] + os.sep
        if os.path.normcase(t.rstrip("\\/")) == os.path.normcase(root.rstrip("\\/")):
            return True
        if os.path.normcase(t.rstrip("\\/")) == os.path.normcase(drive.rstrip("\\/")):
            return True
        return any(ch in target.name for ch in _WILDCARD_CHARS)
