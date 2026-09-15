# -*- coding: utf-8 -*-
"""时崎狂三 · 长期记忆模块
把关于「主人」的重要信息存到本地 memory.json，下次启动自动加载。

三条统一约定（load / save / add 共用同一套语义，避免互相矛盾）：
- 结构校验：只保留 {"content": 非空字符串} 的条目，单条截断到 MAX_MEMORY_LEN；
- 去重：按 content 去重，重复时保留时间较新的那条（位置仍在首次出现处）；
- 上限：条目数超过 MAX_MEMORY_ITEMS 时 FIFO 淘汰**最旧**的条目，保留最近 MAX_MEMORY_ITEMS 条。

损坏保护：解析失败时绝不静默丢掉主人的记忆——先把原文件整体改名备份为
``<memory.json>.corrupt-<时间戳>``，再返回空列表；调用方可读模块级 ``last_load_error``
或改用 ``load_memories_with_status()`` 感知失败原因。

并发保护：同进程用 threading 锁串行化读写，跨进程用 ``<memory.json>.lock`` 锁文件 +
随机名临时文件 + ``os.replace`` 原子替换，并对 Windows"文件仍被别的句柄打开"导致的
PermissionError 做短暂重试。
"""
import contextlib
import datetime
import json
import os
import re
import tempfile
import threading
import time

# JSON 镜像仍在**项目根目录**（与 agent_data/ 并列），不随本模块搬进 kurumi/：
# 它是旧版本回退与人工查看用的数据文件，位置一变，已有安装就会读到一份"空的记忆"。
# 因此这里取包目录的上一级（= 项目根），而不是模块所在目录。
MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory.json")

MAX_MEMORY_ITEMS = 200    # 记忆总条数上限（超出时淘汰最旧）
MAX_MEMORY_LEN = 200      # 单条记忆最大字符数

# 最近一次 load 的失败原因；None 表示本次加载正常（含文件不存在/空文件）。
last_load_error = None

# 最近一次 load_memories_with_status() 的可信度。保存侧据此 fail-closed：
# 读取失败时调用方手里只有"空列表 + 本次新增"，直接落盘会把主人的旧记忆静默销毁
# （此时落盘：原有 2 条记忆被抹掉，文件里只剩本次新增的 1 条，且没有任何备份）。
_last_load_status = None
# 损坏文件是否已成功留档。已留档时覆盖旧文件是安全的；留档失败就必须拒绝写入。
_last_load_backup_ok = False

_LOCK_TIMEOUT = 5.0    # 拿锁最长等待秒数；超时则降级为"无锁写入"，绝不因为锁而丢记忆
_LOCK_STALE = 30.0     # 超过该秒数的锁文件视为残留（持有者已异常退出），可清理

_REPLACE_RETRIES = 8   # Windows 上目标文件被别的进程打开时会拒绝替换，短暂重试即可成功
_REPLACE_DELAY = 0.05  # 重试间隔（线性递增：0.05 + 0.10 + …）

# 同进程内的读/写串行化：多线程同时 save/load 时，Windows 会因"文件仍被别的句柄打开"
# 而让 os.replace / open 报 PermissionError。先过进程内锁，再拿跨进程文件锁。
_PROCESS_LOCK = threading.RLock()


def _retry_if_busy(func, retries=_REPLACE_RETRIES, delay=_REPLACE_DELAY):
    """重试"文件正被占用"这类瞬时错误。

    Windows 上只要还有别的进程/线程打开着同一个文件（没有 FILE_SHARE_DELETE），
    MoveFileEx（os.replace）与 open 都会以 PermissionError [WinError 5/32] 失败——
    例如主人正用编辑器打开 memory.json，或另一个实例正在 load/save。这种占用通常是
    瞬时的，重试几次即可成功；其它 OSError（磁盘满、路径非法）重试没有意义，直接抛出。
    """
    for attempt in range(retries):
        try:
            return func()
        except OSError as e:
            # WinError 5 = 拒绝访问，WinError 32 = 共享冲突（Python 通常映射为 PermissionError，
            # 但不同 Python/平台可能给出裸 OSError，这里两种都认）
            transient = isinstance(e, PermissionError) or getattr(e, "winerror", None) in (5, 32)
            if not transient or attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _valid_entry(item):
    """校验单条记忆结构并补齐扩展字段(类型/来源/更新时间/替代/置顶/有效位)。

    旧条目(只有 content/time)会被补成 kind 按内容判定 / source=legacy / active=true;
    校验与补全逻辑在 `memory.model.normalize_entry`(纯函数,单独可测)。
    """
    from memory import model as memory_model

    entry = memory_model.normalize_entry(item)
    if not entry:
        return None
    entry["content"] = entry["content"][:MAX_MEMORY_LEN]
    return entry


def _normalize_entries(items):
    """校验 + 去重,保持首次出现的顺序。

    同话题的**新条目会作废旧条目**(见 `memory.model.apply_supersede`):
    旧条目保留在列表里但 `active=False`,因此没有数据丢失,主人仍可回溯与纠正。
    """
    from memory import model as memory_model

    out = []
    for item in items:
        entry = _valid_entry(item)
        if not entry:
            continue
        out, _superseded = memory_model.apply_supersede(out, entry)
    return out


def _keep_recent(entries):
    """上限淘汰:优先丢弃**已失效**的旧条目,绝不淘汰置顶条目。

    纯 FIFO(只留最近 MAX_MEMORY_ITEMS 条)会让"关系约定"和临时任务事实一起被淘汰、
    被取代的失效条目反而占着名额。因此:
    - 置顶条目永不淘汰(除非置顶本身超过上限,那时按最旧置顶先丢);
    - 其次淘汰失效条目(它们已保留在历史里,丢失风险最低);
    - 最后才按 FIFO 淘汰最旧的生效条目。
    """
    if len(entries) <= MAX_MEMORY_ITEMS:
        return entries
    pinned = [e for e in entries if e.get("pinned")]
    inactive = [e for e in entries if not e.get("pinned") and not e.get("active", True)]
    active = [e for e in entries if not e.get("pinned") and e.get("active", True)]

    keep_pinned = pinned[-MAX_MEMORY_ITEMS:]
    room = MAX_MEMORY_ITEMS - len(keep_pinned)
    keep = []
    if room > 0:
        # 先用最新的**生效**条目填满名额,只有还剩位置时才保留失效条目。
        # 反过来写(先塞失效条目)会在"1 条失效 + 3 条生效、上限 3"时丢掉一条生效记忆 ——
        # 那正是"优先淘汰失效"要避免的事。
        keep_active = active[-room:]
        room -= len(keep_active)
        if room > 0:
            keep = inactive[-room:]
        keep = keep + keep_active
    kept_ids = {id(e) for e in keep_pinned + keep}
    # 保持原有顺序(时间序),否则"最近"的语义会被打乱
    return [e for e in entries if id(e) in kept_ids]


def _read_text(path):
    """读取记忆文件文本：先 utf-8-sig（兼容 BOM），失败再按 gbk 重试。

    读盘本身也走瞬时占用重试：别的实例正在替换文件时 open 会报 PermissionError，
    此时直接返回 [] 会让 UI 随后把整份记忆覆盖为空。
    """
    def _read_bytes():
        with open(path, "rb") as f:
            return f.read()

    raw = _retry_if_busy(_read_bytes)
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文件既不是 UTF-8 也不是 GBK 编码")


def _backup_corrupt_file(path):
    """把解析失败的原文件改名备份，避免随后的 save_memories 覆盖掉仅存的记忆。"""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = f"{path}.corrupt-{stamp}"
    seq = 1
    while os.path.exists(target):
        target = f"{path}.corrupt-{stamp}-{seq}"
        seq += 1
    try:
        os.replace(path, target)
    except OSError:
        return None
    return target


@contextlib.contextmanager
def _file_lock(path):
    """跨进程写锁：O_CREAT|O_EXCL 抢一个 ``<memory.json>.lock`` 锁文件。

    拿不到锁时（等待超时、或锁文件因占用/权限报错）**不抛异常，直接继续写入**：
    宁可让最后一次写入胜出，也不能因为锁问题丢掉主人的记忆；真正的原子性由
    mkstemp 随机临时名 + os.replace 保证，锁只是减少互相覆盖。
    持有者异常退出留下的陈旧锁（超过 _LOCK_STALE 秒）会被清理，避免永久卡住。
    """
    lock_path = path + ".lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > _LOCK_STALE:
                    os.unlink(lock_path)   # 清理残留锁；删除失败（被占用）则忽略
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                break          # 等不到锁：降级为"无锁写入"，仍然原子替换
            time.sleep(0.02)
        except OSError:
            # 创建/探测锁文件出错（权限、被别的进程占用等）：不阻塞保存，直接不加锁写入
            break
    try:
        yield fd is not None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(lock_path)
            except OSError:
                pass


def load_memories_with_status():
    """加载记忆并返回 (memories, status)。

    status 取值：
      "ok"          正常读取（合法 JSON，非法条目已跳过）
      "missing"     文件不存在（首次运行，正常）
      "empty"       文件存在但为空（正常）
      "corrupt"     解析/结构非法：原文件已备份为 <文件>.corrupt-<时间戳>，返回 []
      "read_error"  文件不可读（被占用/无权限/是目录），未做备份，返回 []
    """
    global last_load_error, _last_load_status, _last_load_backup_ok
    path = MEMORY_FILE
    with _PROCESS_LOCK:      # 与 save_memories 串行，避免读到"正被替换"的文件
        if not os.path.exists(path):
            last_load_error = None
            _last_load_status = "missing"
            return [], "missing"
        try:
            text = _read_text(path)
        except OSError as e:
            last_load_error = f"读取记忆文件失败：{e}"
            _last_load_status = "read_error"
            return [], "read_error"
        except ValueError as e:
            last_load_error = f"记忆文件解码失败：{e}"
            _last_load_status = "corrupt"
            _last_load_backup_ok = bool(_backup_corrupt_file(path))
            return [], "corrupt"
        if not text.strip():
            last_load_error = None
            _last_load_status = "empty"
            return [], "empty"
        try:
            data = json.loads(text)
        except ValueError as e:
            last_load_error = f"记忆文件 JSON 解析失败：{e}"
            _last_load_status = "corrupt"
            _last_load_backup_ok = bool(_backup_corrupt_file(path))
            return [], "corrupt"
        if not isinstance(data, list):
            last_load_error = f"记忆文件结构非法（顶层是 {type(data).__name__}，应为列表）"
            _last_load_status = "corrupt"
            _last_load_backup_ok = bool(_backup_corrupt_file(path))
            return [], "corrupt"
        last_load_error = None
        _last_load_status = "ok"
        return _keep_recent(_normalize_entries(data)), "ok"


def load_memories():
    """加载记忆：损坏/手工编辑过的 JSON 安全降级，非法条目自动跳过并去重。

    返回类型保持 list（兼容 UI.py / agent）。解析失败时返回 []，但原文件已备份为
    ``<memory.json>.corrupt-<时间戳>``，可通过模块级 ``last_load_error`` 或
    ``load_memories_with_status()`` 判断本次是"真的没有记忆"还是"解析失败"。
    """
    memories, _status = load_memories_with_status()
    return memories


def _safe_to_write_empty() -> bool:
    """判断"用空列表覆盖 MEMORY_FILE"是否安全。

    加载失败(文件被长期占用 / 权限异常 / 目录不可读)时,调用方拿到的是空列表;
    若无条件保存,就会把主人的整份记忆清空。这里只在能**证明**目标为空或不存在时
    才允许写空:
    - 文件不存在 / 内容全空白        → 安全(没有数据可丢);
    - 能读出非空列表                → 危险,拒绝;
    - 读不到或解析不了              → 无法证明,按危险处理(fail-safe)。
    真要清空记忆时显式传 ``allow_empty=True``。
    """
    try:
        with open(MEMORY_FILE, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return True
    except OSError:
        return False                     # 读不到就不敢覆盖
    if not raw.strip():
        return True
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError):
        # 解析不了:交给加载路径去备份/提示,绝不在保存时静默清空
        return False
    return not (isinstance(data, list) and data)


def save_memories(memories, allow_empty: bool = False) -> bool:
    """原子写入：进程内锁 + 跨进程锁 + 随机临时名（mkstemp）+ os.replace。

    临时文件用固定名 ``MEMORY_FILE + ".tmp"`` 时，两个实例（或将来多线程）同时保存会
    互相覆盖，甚至把半个文件替换成正式文件；因此临时文件与目标同目录、随机命名，写入
    完成才原子替换，占用导致的 PermissionError 会短暂重试；超过 MAX_MEMORY_ITEMS 时
    同样 FIFO 淘汰最旧条目。

    空列表写入默认受保护:加载失败会让调用方拿到空列表,直接保存等于清空记忆,
    因此仅在 `_safe_to_write_empty()` 成立或显式 `allow_empty=True` 时才写。

    **非空写入同样受保护**:最近一次加载是 `read_error`(文件被占用/无权限,磁盘内容很可能
    完好)时,调用方手里的列表缺的正是"读不到的那部分",写下去等于用残缺内容覆盖整份记忆
    —— 会静默销毁主人的旧记忆且不留备份。此时直接拒绝,由调用方提示并重新加载
    (瞬时占用解除后即可自愈)。`corrupt` 只在**旧内容留档失败**时才拒绝:留档成功时
    旧文件已改名保全,覆盖它是安全的(否则主人永远存不进新记忆)。
    返回 True 表示已写入,False 表示被保护拦下(调用方据此提示)。
    """
    global _last_load_status
    clean = _keep_recent(_normalize_entries(list(memories)))
    if not allow_empty:
        if _last_load_status == "read_error":
            return False
        if _last_load_status == "corrupt" and not _last_load_backup_ok:
            return False
        if not clean and not _safe_to_write_empty():
            return False
    directory = os.path.dirname(os.path.abspath(MEMORY_FILE)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = None
    # 先过进程内锁（同进程多线程的唯一入口），再拿跨进程锁文件，最后随机名 + 原子替换
    with _PROCESS_LOCK, _file_lock(MEMORY_FILE):
        fd, tmp_path = tempfile.mkstemp(prefix="memory.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(clean, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            _retry_if_busy(lambda: os.replace(tmp_path, MEMORY_FILE))
            tmp_path = None
            # 写成功后磁盘内容就等于 clean,加载可信度恢复(否则一次瞬时读失败会永久封死保存)
            _last_load_status = "ok"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    return True


def validated_memory_contents(memories):
    """返回通过结构与长度校验的记忆条目文本，供 Agent 提示构建使用。"""
    out = []
    for m in memories:
        entry = _valid_entry(m)
        if entry:
            out.append(entry["content"])
    return out


def build_memory_context(memories, max_items=30):
    """把记忆拼成一段上下文，注入 system 提示。"""
    if not memories:
        return ""
    recent = memories[-max_items:]
    lines = "\n".join("- " + m.get("content", "") for m in recent)
    return "\n【你与主人之间的重要回忆（请自然地融入对话，不要逐条复述）】\n" + lines + "\n"


_ROLE_LABELS = {"user": "主人", "assistant": "狂三"}


def history_to_text(history):
    """把 history（[{"role","content"}]）转成纯文本，供记忆提炼使用。

    按角色映射说话人：只有 user→主人、assistant→狂三，其余（system/tool 等）标为
    「（系统）」——若把所有非 user 角色都写成"狂三"，系统提示或工具结果就会被
    当成狂三的话，污染记忆提炼。

    标签由 `PersonaProfile.role_labels()` 提供（不再硬编码副本）：
    以后改称呼只需改人设一处，转录与记忆提炼会同步变化。
    """
    from persona import DEFAULT_PERSONA

    labels = DEFAULT_PERSONA.role_labels()
    lines = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        label = labels.get(str(msg.get("role") or ""), "（系统）")
        lines.append(f"{label}：{msg.get('content', '')}")
    return "\n".join(lines)


def extract_prompt(history_text):
    return (
        "请从下面这段你和主人之间的对话里，提炼出关于「主人」值得长期记住的重要信息"
        "（例如：名字、称呼、喜好、讨厌、约定、经历、心情、身份、重要事件等）。\n"
        "要求：\n"
        "1. 用简短的中文条目列出，每行一条，以「- 」开头；\n"
        "2. 只写客观事实或约定，不要写对话过程；\n"
        "3. 如果对话里没有值得长期记住的新信息，只回复「无」。\n"
        "4. 若某条**改变了**已有记忆（例如以前喜欢咖啡、现在改成喝茶），"
        "在该条后面另起一行写「取代: <被取代的旧内容原文>」；\n"
        "5. 若某条属于角色自身的虚构经历（狂三的经历、天使、刻刻帝等，"
        "**不是主人的真实经历**），在该条后另起一行写「类型: fiction」；\n"
        "6. 其余条目可省略第 4、5 条指令。\n\n"
        "对话内容：\n" + history_text + "\n\n记忆条目："
    )


_LIST_MARKER_RE = re.compile(r"^[-*]\s+")
_HAS_WORD_RE = re.compile(r"[\w\u4e00-\u9fa5]")
# 模型"没有新信息"时的各种答法，都不能当成记忆写进 memory.json
_EMPTY_ANSWERS = {"无", "（无）", "(无)", "【无】", "无。", "none", "null", "n/a", "na",
                  "没有", "暂无", "无内容", "没有新信息"}


def _is_empty_answer(item):
    """判断是否是"没有新信息"的空答（去掉尾部标点后再比对）。"""
    text = item.strip().strip("。.！!~、，,；;：:").strip()
    return text.lower() in _EMPTY_ANSWERS


def parse_memories(reply):
    """从模型输出里解析出「- xxx」形式的条目,以及可选的「取代:/类型:」指令。

    - 只剥掉**一个**列表标记（``- `` / ``* ``），内容里自带的 ``-`` 与缩进必须保留
      （``line.lstrip("- ")`` 会把 "--no-quantize" 这类内容剥成 "no-quantize"）；
    - 「无」「（无）」「无。」「None」等空答一律丢弃，不写进记忆；
    - 支持两条**紧跟条目之后**的指令行（见 `extract_prompt` 的要求 4/5）:
      `取代: <旧内容>` → 该条生效时作废旧条目;`类型: fiction` → 标记为角色虚构经历。
      指令行自身不会被当成记忆内容。

    返回 `list[dict]`(含 content/kind/supersedes),供 `add_memories` 直接并入。
    """
    from memory import model as memory_model

    entries = []
    for raw in str(reply or "").splitlines():
        line = raw.strip()
        if line.startswith(("-", "*")):
            item = _LIST_MARKER_RE.sub("", line, count=1).strip()
            # 只有标记没有内容（"- " / "-" / "---"）或纯标点，跳过
            if not _HAS_WORD_RE.search(item):
                continue
            if _is_empty_answer(item):
                continue
            entry = {"content": item[:MAX_MEMORY_LEN]}
            kind = memory_model.parse_kind_directive(item)
            if kind:
                entry["kind"] = kind
            entries.append(entry)
            continue
        if not entries:
            continue                     # 指令必须跟在某条记忆之后
        target = entries[-1]
        kind = memory_model.parse_kind_directive(line)
        if kind:
            target["kind"] = kind
            continue
        old = memory_model.parse_supersede_directive(line)
        if old:
            target["supersedes"] = old[:MAX_MEMORY_LEN]
    return entries


def add_memories(memories, new_entries):
    """并入新记忆(去重 + 同话题替代),返回真实新增条数。

    `new_entries` 接受两种形态:
    - 纯字符串(旧调用方):按内容新建条目;
    - dict(`parse_memories` 的输出):可携带 `kind` / `supersedes` 指令。

    到上限时按 `_keep_recent` 的策略淘汰(优先丢失效、绝不丢置顶)——纯 FIFO 一旦存满
    200 条就再也学不到任何新记忆(静默失效)。
    """
    from memory import model as memory_model

    added = 0
    for item in new_entries:
        if isinstance(item, str):
            text = item.strip()[:MAX_MEMORY_LEN]
            entry = {"content": text} if text else None
        elif isinstance(item, dict):
            entry = dict(item)
        else:
            entry = None
        if not entry:
            continue
        content = str(entry.get("content") or "").strip()[:MAX_MEMORY_LEN]
        if not content:
            continue
        entry["content"] = content
        entry.setdefault("time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        normalized = _valid_entry(entry)
        if not normalized:
            continue
        before = {m.get("content") for m in memories if isinstance(m, dict)}
        replaced, superseded = memory_model.apply_supersede(
            [m for m in memories if isinstance(m, dict)], normalized)
        memories[:] = replaced
        # 只有"确实带来了新内容"才计为新增(同内容刷新/替代旧条目不算新增条目)
        if content not in before or superseded:
            added += 1
    memories[:] = _keep_recent(memories)
    return added


# ==================== 查看 / 纠正 / 遗忘 / 置顶 ====================
def find_memories(memories, keyword="", include_inactive=False):
    """查看记忆:按关键词(内容子串)过滤;默认只看生效条目。"""
    key = str(keyword or "").strip().lower()
    out = []
    for item in memories or []:
        if not isinstance(item, dict):
            continue
        if not include_inactive and not item.get("active", True):
            continue
        content = str(item.get("content", "") or "")
        if key and key not in content.lower():
            continue
        out.append(item)
    return out


def _resolve_targets(memories, target):
    """把 target(序号或关键词)解析成命中条目。

    序号以 `format_memory_list` 的编号为准(1 基,只数生效条目)。
    """
    text = str(target or "").strip()
    if not text:
        return []
    lowered = text.lower()
    visible = [m for m in (memories or []) if isinstance(m, dict) and m.get("active", True)]
    if text.isdigit():
        index = int(text) - 1
        if 0 <= index < len(visible):
            return [visible[index]]
        return []
    return [m for m in visible if lowered in str(m.get("content", "") or "").lower()]


def forget_memory(memories, target):
    """遗忘:把命中条目置为失效(不物理删除)。返回 (被遗忘的内容列表, 命中数)。"""
    hits = _resolve_targets(memories, target)
    forgotten = []
    for entry in hits:
        entry["active"] = False
        forgotten.append(str(entry.get("content", "")))
    return forgotten, len(hits)


def correct_memory(memories, target, new_content):
    """纠正:改写命中条目的内容,原内容记入 supersedes。返回 (旧内容列表, new_content)。

    保留 time 与 pinned(主人标记过的重要性和原始时间都不该因为一次纠正而丢失),
    只刷新 updated;内容变了就重新判定类型。
    """
    from memory import model as memory_model

    hits = _resolve_targets(memories, target)
    text = str(new_content or "").strip()[:MAX_MEMORY_LEN]
    if not hits or not text:
        return [], text
    replaced = []
    for entry in hits:
        old = str(entry.get("content", ""))
        replaced.append(old)
        entry["content"] = text
        entry["supersedes"] = old
        entry["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        entry["kind"] = memory_model.classify_kind(text, str(entry.get("source") or ""))
    return replaced, text


def set_pinned(memories, target, pinned=True):
    """置顶 / 取消置顶。返回命中条目的内容列表。"""
    hits = _resolve_targets(memories, target)
    for entry in hits:
        entry["pinned"] = bool(pinned)
    return [str(entry.get("content", "")) for entry in hits]


_KIND_LABELS = {
    "preference": "偏好", "agreement": "约定", "state": "状态",
    "task_fact": "任务", "fiction": "虚构", "fact": "事实",
}


def format_memory_list(memories, include_inactive=False):
    """查看结果的文本呈现(带序号,供 /忘记、/纠正、/置顶 引用)。"""
    entries = find_memories(memories, include_inactive=include_inactive)
    if not entries:
        return "（还没有记住关于主人的事）"
    lines = []
    for index, entry in enumerate(entries, start=1):
        label = _KIND_LABELS.get(str(entry.get("kind") or ""), "事实")
        marks = []
        if entry.get("pinned"):
            marks.append("置顶")
        if not entry.get("active", True):
            marks.append("已失效")
        suffix = ("（" + "·".join(marks) + "）") if marks else ""
        lines.append(f"{index}. [{label}]{suffix} {entry.get('content', '')}")
    return "\n".join(lines)


# ==================== 可选 SQLite 后端 ====================
# - 未 attach 时行为与纯 JSON 路径**完全一致**,因此默认路径零风险;
# - attach 之后 DB 为准,同时把内容镜像导出到 memory.json(回退到旧版本仍有数据);
# - DB 出错时自动退回 JSON 并记录原因,绝不让一条坏 DB 使聊天不可用。
_STORE = None
_store_error = None


def attach_store(store):
    """把 SQLite 存储挂为记忆的权威来源。任何异常都退回 JSON 路径。"""
    global _STORE, _store_error
    _STORE = store
    _store_error = None
    return store


def detach_store():
    """回到纯 JSON 路径(回退方法:去掉调用即可)。"""
    global _STORE, _store_error
    _STORE = None
    _store_error = None


def store_attached() -> bool:
    return _STORE is not None


def last_store_error():
    return _store_error


def _store_call(func, fallback):
    """调用存储层;失败则记录原因并退回 JSON 结果。"""
    global _STORE, _store_error
    if _STORE is None:
        return fallback()
    try:
        return func(_STORE)
    except Exception as e:
        _store_error = f"{type(e).__name__}: {e}"
        print(f"[!] 记忆存储不可用,已退回 JSON: {_store_error}")
        _STORE = None
        return fallback()


def load_memories_store_first():
    """优先从 DB 读;未挂载或失败时读 JSON(供 UI 启动使用)。"""
    return _store_call(lambda s: s.load_memories(), load_memories)


def save_memories_store_first(memories, allow_empty=False):
    """写 DB 并镜像导出 JSON;未挂载或失败时只写 JSON。

    **同一套"防误清空"判定必须保留**:走 DB 时若跳过它,一次空写就会把 DB 里的记忆
    全删掉 —— 那正是 JSON 路径上同样必须防住的事。因此这里先按同样的规则判定,
    再决定是否落盘。
    """
    global _last_load_status, _STORE, _store_error, last_load_error
    clean = _keep_recent(_normalize_entries(list(memories)))
    if _STORE is None:
        return save_memories(clean, allow_empty=allow_empty)
    if not allow_empty:
        if _last_load_status == "read_error":
            return False
        try:
            existing = _STORE.load_memories()
        except Exception as e:
            _last_load_status = "read_error"
            last_load_error = f"读取记忆存储失败:{e}"
            return False
        if existing and not clean:
            return False              # 库里有内容却要写空 → 拒绝
    try:
        _STORE.save_memories(clean)
        _STORE.export_json(MEMORY_FILE)   # 镜像:回退到旧版本时数据仍在
        _last_load_status = "ok"
        return True
    except Exception as e:
        _store_error = f"{type(e).__name__}: {e}"
        print(f"[!] 记忆存储写入失败,已退回 JSON: {_store_error}")
        _STORE = None
        return save_memories(clean, allow_empty=allow_empty)
