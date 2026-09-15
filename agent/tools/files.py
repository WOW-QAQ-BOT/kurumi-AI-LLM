# -*- coding: utf-8 -*-
"""有界文本文件工具:Schema 严格、读取分段、写入原子、修改一律走审批。"""
import fnmatch
import os
from pathlib import Path

from agent.policy import (
    PermissionPolicy,
    _content_looks_sensitive,
    _is_sensitive_path,
    _is_within,
    fingerprint,
    hash_arguments,
)
from agent.tools.base import ApprovalGrant, PreparedToolCall, ToolDefinition, ToolRegistry
from agent.types import PolicyDecision, ToolCall, ToolResult

MAX_LIST_ENTRIES = 200
MAX_SEARCH_RESULTS = 100
MAX_SEARCH_SCAN = 2000
MAX_SEARCH_CONTENT_BYTES = 16 * 1024 * 1024   # 单次内容检索允许读取的总字节预算
MAX_READ_LINES = 400
MAX_READ_BYTES = 256 * 1024
MAX_WRITE_BYTES = 1024 * 1024


def _default_trash(path: Path) -> None:
    try:
        from send2trash import send2trash
    except ImportError as e:
        raise RuntimeError("send2trash 未安装,无法安全回收") from e
    send2trash(str(path))


def build_file_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="file_list",
        description="列出目录内容(默认不递归,上限 200 项)。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": False},
                "max_entries": {"type": "integer", "default": 200},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        mutating=False,
    ))
    registry.register(ToolDefinition(
        name="file_search",
        description="在目录中按文件名(query)或通配符(glob)搜索,可选匹配文本内容;上限 100 条结果。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "query": {"type": "string", "default": ""},
                "glob": {"type": "string", "default": ""},
                "max_results": {"type": "integer", "default": 100},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        mutating=False,
    ))
    registry.register(ToolDefinition(
        name="file_read",
        description="分段读取文本文件;单次最多 400 行且不超过 256 KiB。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "max_lines": {"type": "integer", "default": 400},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        mutating=False,
    ))
    registry.register(ToolDefinition(
        name="file_write",
        description="新建或覆盖文本文件;mode 只能为 create 或 overwrite;单次最多 1 MiB。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["create", "overwrite"]},
            },
            "required": ["path", "content", "mode"],
            "additionalProperties": False,
        },
        mutating=True,
    ))
    registry.register(ToolDefinition(
        name="file_patch",
        description="用 replacement 替换 expected_text;expected_text 必须恰好出现一次;结果最多 1 MiB。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "expected_text": {"type": "string"},
                "replacement": {"type": "string"},
            },
            "required": ["path", "expected_text", "replacement"],
            "additionalProperties": False,
        },
        mutating=True,
    ))
    registry.register(ToolDefinition(
        name="file_move",
        description="移动或重命名单个文件;目标已存在时拒绝覆盖。",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["source", "destination"],
            "additionalProperties": False,
        },
        mutating=True,
    ))
    registry.register(ToolDefinition(
        name="file_trash",
        description="通过 Windows 回收站回收单个目标,不提供永久删除。",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        mutating=True,
    ))
    return registry


class ToolExecutor:
    """prepared-call 绑定 + 审批校验 + TOCTOU 复查 + 有界执行。"""

    def __init__(self, policy: PermissionPolicy, trash_adapter=None):
        self.policy = policy
        self.registry = build_file_registry()
        self.trash_adapter = trash_adapter or _default_trash

    # ---------- prepare ----------
    def prepare(self, call: ToolCall) -> PreparedToolCall:
        definition, args = self.registry.prepare(call.name, call.arguments)
        args = self._enforce_limits(call.name, args)
        assessment = self.policy.assess(ToolCall(call.id, call.name, args))
        canonical = assessment.canonical_arguments or args
        return PreparedToolCall(
            call=call,
            definition=definition,
            canonical_arguments=canonical,
            arguments_hash=hash_arguments(canonical),
            policy=assessment,
            file_fingerprint=assessment.file_fingerprint,
        )

    def _enforce_limits(self, name: str, args: dict) -> dict:
        if name == "file_list":
            args["max_entries"] = max(1, min(int(args.get("max_entries", 200)), MAX_LIST_ENTRIES))
        elif name == "file_search":
            args["max_results"] = max(1, min(int(args.get("max_results", 100)), MAX_SEARCH_RESULTS))
        elif name == "file_read":
            args["start_line"] = max(1, int(args.get("start_line", 1)))
            args["max_lines"] = max(1, min(int(args.get("max_lines", 400)), MAX_READ_LINES))
        elif name == "file_write":
            content = args.get("content", "")
            if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
                raise ValueError("内容超过 1 MiB 上限")
        elif name == "file_patch":
            if len(args.get("replacement", "").encode("utf-8")) > MAX_WRITE_BYTES:
                raise ValueError("替换结果超过 1 MiB 上限")
        return args

    # ---------- execute ----------
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
        # 执行瞬间复核(ALLOW 与已批准调用都执行):重新解析最终路径与父目录,
        # 防止 prepare→execute 之间目标被替换为符号链接等(TOCTOU)
        error = self.execution_recheck(prepared)
        if error is not None:
            return error
        return self._dispatch(prepared)

    def execution_recheck(self, prepared: PreparedToolCall):
        """重新走一遍策略评估;路径、归属、指纹任一变化立即中止,不落盘不读取。

        用「规范参数」重新评估(与 prepare 时的哈希基准一致),评估内部会再次
        resolve 最终路径与父目录,从而捕获 prepare→execute 之间的符号链接替换。
        """
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
                              "目标路径在执行前被替换,操作已取消")
        if prepared.file_fingerprint is not None:
            path = Path(prepared.canonical_arguments.get("path")
                        or prepared.canonical_arguments.get("source", ""))
            if fingerprint(path) != prepared.file_fingerprint:
                return ToolResult(prepared.call.id, False, "FINGERPRINT_CHANGED",
                                  "文件在审批后被修改,操作已取消")
        return None

    def revalidate(self, prepared: PreparedToolCall):
        """兼容旧接口:仅做指纹复查,路径复查统一走 execution_recheck。"""
        if prepared.file_fingerprint is None:
            return None
        path = Path(prepared.canonical_arguments.get("path")
                    or prepared.canonical_arguments.get("source", ""))
        if fingerprint(path) != prepared.file_fingerprint:
            return ToolResult(prepared.call.id, False, "FINGERPRINT_CHANGED",
                              "文件在审批后被修改,操作已取消")
        return None

    def _dispatch(self, prepared: PreparedToolCall) -> ToolResult:
        name = prepared.call.name
        args = prepared.canonical_arguments
        call_id = prepared.call.id
        try:
            if name == "file_list":
                return self._file_list(call_id, args)
            if name == "file_search":
                return self._file_search(call_id, args)
            if name == "file_read":
                return self._file_read(call_id, args)
            if name == "file_write":
                return self._file_write(call_id, args)
            if name == "file_patch":
                return self._file_patch(call_id, args)
            if name == "file_move":
                return self._file_move(call_id, args)
            if name == "file_trash":
                return self._file_trash(call_id, args)
            return ToolResult.denied(call_id, f"未实现工具: {name}")
        except PermissionError as e:
            # 必须排在 OSError 之前(PermissionError 是 OSError 子类):
            # 这是执行层的"安全中止"信号(父目录越根等),不是磁盘故障,
            # 贴成 IO_ERROR 会诱导模型"重试/换个路径再来一次"。
            return ToolResult(call_id, False, "PATH_CHANGED", f"安全中止: {e}")
        except UnicodeError as e:
            return ToolResult(call_id, False, "INVALID_CONTENT", str(e))
        except RuntimeError as e:
            # trash_adapter 不可用(send2trash 未安装等)
            return ToolResult(call_id, False, "TRASH_UNAVAILABLE", str(e))
        except ValueError as e:
            return ToolResult(call_id, False, "INVALID_CONTENT", str(e))
        except OSError as e:
            return ToolResult(call_id, False, "IO_ERROR", str(e))
        except Exception as e:
            # 兜底隔离:白名单外异常(webbrowser.Error、KeyError、第三方库异常…)若穿到
            # runner 的兜底,会让整个 run 变成 INTERNAL_ERROR;工具内部错误必须就地降级成
            # 一条可读的工具结果(TOOL_ERROR),让模型能换个做法而不是整轮失败。
            return ToolResult(call_id, False, "TOOL_ERROR", f"{type(e).__name__}: {e}")

    # ---------- 只读 ----------
    def _file_list(self, call_id, args):
        root = Path(args["path"])
        if not root.is_dir():
            return ToolResult(call_id, False, "NOT_A_DIRECTORY", "目标不是目录")
        lines = []
        skipped_links = 0
        allowed_root = str(self.policy.settings.allowed_root)

        def unsafe(p: Path) -> bool:
            """p 是否通过链接/junction 指到允许根之外。

            file_list 只做 iterdir + is_dir()(紧跟链接)时,根内一个指向根外的
            目录链接就能把根外的目录结构与文件名零审批列出来(实机复现);
            这里与 file_search 一样先解析再读取,做同样的事前校验。
            """
            if not p.is_symlink():
                return False
            try:
                resolved = p.resolve(strict=True)
            except OSError:
                return True
            return not _is_within(str(resolved), allowed_root)

        def walk(directory, depth):
            nonlocal skipped_links
            if depth > 4 or len(lines) >= args["max_entries"]:
                return
            for p in sorted(directory.iterdir(), key=lambda x: x.name.lower()):
                if len(lines) >= args["max_entries"]:
                    return
                if unsafe(p):
                    # 不输出名字、也不递归:名字本身就是根外信息
                    skipped_links += 1
                    continue
                rel = p.relative_to(root)
                lines.append(("D" if p.is_dir() else "F") + " " + str(rel))
                if args.get("recursive") and p.is_dir():
                    walk(p, depth + 1)

        walk(root, 1)
        metadata = {"returned_entries": len(lines), "path": str(root)}
        if skipped_links:
            # 显式告知被跳过的链接数,否则模型会把"不完整"当成"没有"
            metadata["skipped_outside_links"] = skipped_links
        return ToolResult(call_id, True, "OK", "\n".join(lines) or "(空目录)", metadata)

    def _file_search(self, call_id, args):
        root = Path(args["path"])
        query = args.get("query", "").strip()
        glob_pat = args.get("glob", "").strip()
        if not root.is_dir():
            return ToolResult(call_id, False, "NOT_A_DIRECTORY", "目标不是目录")
        if glob_pat and "\x00" in glob_pat:
            return ToolResult(call_id, False, "INVALID_GLOB", "非法通配符")
        matches = []
        scanned = 0
        content_bytes = 0
        decode_skipped = 0
        for p in root.rglob("*"):
            scanned += 1
            if scanned > MAX_SEARCH_SCAN or len(matches) >= args["max_results"]:
                break
            if not p.is_file():
                continue
            name_ok = True
            if query and query.lower() not in p.name.lower():
                name_ok = False
            if glob_pat and not fnmatch.fnmatch(p.name.lower(), glob_pat.lower()):
                name_ok = False
            # 先解析再读取:链接指向根外时不读取内容、不泄露根外大小
            try:
                resolved = p.resolve(strict=True)
            except OSError:
                resolved = None
            within = resolved is not None and _is_within(
                str(resolved), str(self.policy.settings.allowed_root))
            content_hit = False
            if (query and not name_ok and within
                    and content_bytes < MAX_SEARCH_CONTENT_BYTES):
                # 敏感文件永不参与内容匹配:否则内容检索会变成绕过审批的密钥存在性神谕。
                # 名字没命中的文件同样要过内容探测 —— 否则"名字普通的笔记里写着 api_key"
                # 仍会参与匹配(实机复现),与上面这条注释承诺的范围不符。
                if _is_sensitive_path(resolved) or _content_looks_sensitive(resolved):
                    continue
                try:
                    st = resolved.stat()
                except OSError:
                    st = None
                if st is not None and st.st_size <= MAX_READ_BYTES * 2:
                    try:
                        text = resolved.read_text(encoding="utf-8")   # 用已解析路径读取,缩小竞态窗口
                    except UnicodeError:
                        # 非 UTF-8(GBK 中文文件很常见)时静默当"无命中"会造成假阴性;
                        # 这里计入 decode_skipped 让模型/主人知道结果不完整
                        text = ""
                        decode_skipped += 1
                    except OSError:
                        text = ""
                    content_bytes += int(st.st_size)
                    content_hit = query in text
            if name_ok or content_hit:
                rel = p.relative_to(root)
                size_text = ""
                if within:
                    try:
                        size_text = f" ({resolved.stat().st_size} B)"
                    except OSError:
                        pass
                matches.append(f"{rel}{size_text}")
        return ToolResult(call_id, True, "OK",
                          "\n".join(matches) or "（未找到匹配项）",
                          {"returned_results": len(matches), "scanned": scanned,
                           "content_scanned_bytes": content_bytes,
                           "content_budget_exhausted": content_bytes >= MAX_SEARCH_CONTENT_BYTES,
                           # > 0 表示有文件因编码非 UTF-8 未能参与内容匹配(存在假阴性)
                           "decode_skipped": decode_skipped})

    def _file_read(self, call_id, args):
        path = Path(args["path"])
        if not path.is_file():
            return ToolResult(call_id, False, "NOT_A_FILE", "目标不是文件")
        start = args["start_line"]
        max_lines = args["max_lines"]
        collected = []
        total_bytes = 0
        truncated = False
        try:
            # utf-8-sig:带 BOM 的文件(BOM 是 Windows 记事本常见产物)首行不能带 \ufeff,
            # 否则针对首行的 file_patch 的 expected_text 几乎永远匹配不上
            with open(path, encoding="utf-8-sig") as f:
                for idx, line in enumerate(f, start=1):
                    if idx < start:
                        continue
                    if len(collected) >= max_lines or total_bytes >= MAX_READ_BYTES:
                        truncated = True
                        break
                    enc = line.encode("utf-8")
                    if total_bytes + len(enc) > MAX_READ_BYTES:
                        # 单行超限:截断到剩余预算
                        room = MAX_READ_BYTES - total_bytes
                        collected.append(enc[:room].decode("utf-8", errors="ignore"))
                        truncated = True
                        break
                    collected.append(line.rstrip("\n"))
                    total_bytes += len(enc)
        except UnicodeDecodeError:
            return ToolResult(call_id, False, "UNSUPPORTED_ENCODING", "仅支持 UTF-8 等文本编码")
        content = "\n".join(collected)
        if start > 1:
            truncated = True
        return ToolResult(call_id, True, "OK", content,
                          {"returned_lines": len(collected), "start_line": start,
                           "truncated": truncated})

    # ---------- 修改 ----------
    def _atomic_write(self, path: Path, text: str) -> None:
        """原子写入:同目录随机名 + O_EXCL 排他创建 + 全程父目录校验。

        防两类攻击:①预先占位随机临时名的符号链接;②执行过程中父目录被换成
        指向根外的目录链接。创建前后与替换前都重新解析父目录并校验归属。
        """
        import secrets
        parent = path.parent
        parent_expected = parent.resolve(strict=True)
        if not _is_within(str(parent_expected), str(self.policy.settings.allowed_root)):
            raise PermissionError("父目录越出允许根,已中止")
        last_err = None
        for _ in range(5):
            tmp = parent / (path.name + f".tmp-{secrets.token_hex(8)}")
            try:
                fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            ok = False
            try:
                # 创建后校验:临时文件必须是普通文件,且其真实父目录未被替换
                if tmp.is_symlink():
                    raise ValueError("临时文件被替换为符号链接,已中止")
                if tmp.resolve(strict=True).parent != parent_expected:
                    raise ValueError("父目录在执行中被替换,已中止")
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                    fd = None
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                # 替换前最后一刻再校验父目录
                if parent.is_symlink() or parent.resolve(strict=True) != parent_expected:
                    raise ValueError("父目录在执行中被替换,已中止")
                os.replace(tmp, path)
                ok = True
                return
            except BaseException as e:
                last_err = e
                raise
            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                if not ok:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        raise OSError("无法创建唯一临时文件") from last_err

    def _file_write(self, call_id, args):
        path = Path(args["path"])
        mode = args["mode"]
        if mode == "create" and path.exists():
            return ToolResult(call_id, False, "FILE_EXISTS", "目标已存在,create 模式拒绝覆盖")
        self._atomic_write(path, args["content"])
        return ToolResult(call_id, True, "OK", f"已写入 {path}", {"size": path.stat().st_size})

    def _file_patch(self, call_id, args):
        path = Path(args["path"])
        if not path.is_file():
            return ToolResult(call_id, False, "NOT_A_FILE", "目标不是文件")
        # 有界读取:按"字节"判上限(按字符数比会漏判:中文文件实际能放进 ~3 MiB);
        # 先读字节再解码,同时天然保留原始换行(CRLF 不会被通用换行翻译成 LF)
        try:
            with open(path, "rb") as fb:
                raw = fb.read(MAX_WRITE_BYTES + 4)
        except OSError as e:
            return ToolResult(call_id, False, "IO_ERROR", str(e))
        if len(raw) > MAX_WRITE_BYTES:
            return ToolResult(call_id, False, "PATCH_TOO_LARGE", "文件超过 1 MiB,不支持 patch")
        had_bom = raw.startswith(b"\xef\xbb\xbf")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return ToolResult(call_id, False, "UNSUPPORTED_ENCODING", "仅支持 UTF-8 等文本编码")
        expected = args["expected_text"]
        count = text.count(expected)
        if count != 1:
            return ToolResult(call_id, False, "MATCH_COUNT_INVALID",
                              f"expected_text 出现 {count} 次,必须恰好一次")
        new_text = text.replace(expected, args["replacement"])
        # 原样回写:BOM 若存在就保留,换行按读取到的原始字节输出(_atomic_write 用 newline="")
        payload = ("\ufeff" + new_text) if had_bom else new_text
        if len(payload.encode("utf-8")) > MAX_WRITE_BYTES:
            return ToolResult(call_id, False, "RESULT_TOO_LARGE", "替换结果超过 1 MiB 上限")
        self._atomic_write(path, payload)
        return ToolResult(call_id, True, "OK", f"已修改 {path}",
                          {"size": path.stat().st_size, "bom_preserved": had_bom})

    def _outside_root_blocked(self, path: Path) -> bool:
        """允许根之外的目标一律拒绝,与 _atomic_write 的根约束一致。

        策略层若保留 `allow_outside_root_mutations` 开关把这类变更变成 ASK,
        执行层也不放行 → 主人批准了却什么都没发生,因此该开关不保留,两侧语义统一。
        """
        return not _is_within(str(path), str(self.policy.settings.allowed_root))

    def _file_move(self, call_id, args):
        src = Path(args["source"])
        dst = Path(args["destination"])
        if not src.is_file():
            return ToolResult(call_id, False, "NOT_A_FILE", "源不是文件")
        if self._outside_root_blocked(src) or self._outside_root_blocked(dst):
            return ToolResult(call_id, False, "PATH_OUTSIDE_ROOT", "移动的源或目标在允许根之外,已拒绝")
        if dst.exists():
            return ToolResult(call_id, False, "MOVE_COLLISION", "目标已存在,拒绝覆盖")
        try:
            # Windows 上目标已存在时 os.rename 抛 FileExistsError,不会像 os.replace 那样静默覆盖
            os.rename(src, dst)
        except FileExistsError:
            return ToolResult(call_id, False, "MOVE_COLLISION", "目标已存在,拒绝覆盖")
        return ToolResult(call_id, True, "OK", f"已移动到 {dst}")

    def _file_trash(self, call_id, args):
        path = Path(args["path"])
        if not path.exists():
            return ToolResult(call_id, False, "NOT_FOUND", "目标不存在")
        # 目录守卫:工具描述承诺的是"回收单个目标",而 _is_protected_target 只挡
        # "允许根/盘根精确相等" —— 没有这道守卫时,一次批准就能回收整棵目录树
        # (含其中所有文件),与主人看到的审批卡语义完全不符。目录回收请主人在
        # 资源管理器里手动完成(可预览、可撤销、可部分恢复)。
        if path.is_dir():
            return ToolResult(call_id, False, "IS_DIRECTORY", "目录回收需人工在资源管理器中完成")
        if self._outside_root_blocked(path):
            return ToolResult(call_id, False, "PATH_OUTSIDE_ROOT", "目标在允许根之外,已拒绝")
        self.trash_adapter(path)
        return ToolResult(call_id, True, "OK", f"已移入回收站 {path}")


def build_file_executor(policy: PermissionPolicy, trash_adapter=None) -> ToolExecutor:
    return ToolExecutor(policy, trash_adapter=trash_adapter)
