# -*- coding: utf-8 -*-
"""SQLite 存储层:记忆与会话的持久化。

为什么需要它:
- 记忆改由 SQLite 承载:单个 `memory.json` 没有事务、没有版本、也不便按类型查询;
- 会话（哪些轮次说过什么）必须落盘并在重启后恢复 —— "重启后能恢复会话和明确约定"
  是硬要求,重启即丢不可接受。

设计约束:
1. **导入旧 JSON 必须走 `kurumi_memory.load_memories_with_status()`** —— 复用既有的损坏备份、
   GBK 回退与防误清空判定;若自己读文件,一次损坏的 memory.json 就会在迁移中被静默丢掉;
2. 导入只做一次(`meta.imported_from_json`),且**不删原文件**;
3. DB 打不开或迁移失败 → 由调用方回退到 JSON,绝不让一条坏 DB 使聊天不可用。
"""
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA_VERSION = 2

# 每条迁移在**单个事务**里执行;新增版本时往后追加,不要改动已发布的条目。
_MIGRATIONS = {
    1: """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS memories (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        content     TEXT    NOT NULL,
        kind        TEXT    NOT NULL DEFAULT 'fact',
        source      TEXT    NOT NULL DEFAULT 'legacy',
        created_at  TEXT    NOT NULL DEFAULT '',
        updated_at  TEXT    NOT NULL DEFAULT '',
        supersedes  TEXT    NOT NULL DEFAULT '',
        pinned      INTEGER NOT NULL DEFAULT 0,
        active      INTEGER NOT NULL DEFAULT 1
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_content ON memories(content);
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        started_at TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL DEFAULT '',
        engine     TEXT NOT NULL DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS turns (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT    NOT NULL,
        seq        INTEGER NOT NULL,
        role       TEXT    NOT NULL,
        content    TEXT    NOT NULL,
        engine     TEXT    NOT NULL DEFAULT '',
        created_at TEXT    NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, seq);
    """,
    # 崩溃前已经落盘、但没得到回答的提问要**留着**并标记,不能删。
    2: """
    ALTER TABLE turns ADD COLUMN incomplete INTEGER NOT NULL DEFAULT 0;
    """,
}


def _column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


# 每条迁移的"**是否已经生效**"判断。
#
# 为什么必须有:一次迁移是**两步** —— 先改结构(ALTER/建表),再写 `user_version`。
# 进程正好在两步之间被杀,库里就会留下"结构已是新的、版本号还是旧的"这种状态;
# 下次启动把同一条 SQL 再跑一遍,得到的是 `duplicate column name`,
# 于是**这个库再也打不开**(记忆与会话会一起降级成 JSON 回退,数据还在却读不出来)。
# 有了这个判断,那种库只会补一个版本号就恢复。
#
# **新增含 ALTER / ADD COLUMN / CREATE INDEX 的迁移时,必须在这里登记一条判断**
# —— 这是硬性要求:不登记就等于放弃了"中断后可恢复"这条保证,而且没有自动化用例
# 会替你发现。
_MIGRATION_APPLIED = {
    2: lambda conn: _column_exists(conn, "turns", "incomplete"),
}


class MemoryStore:
    """记忆与会话的 SQLite 存储。所有写操作都在事务里完成。"""

    def __init__(self, path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(f"PRAGMA busy_timeout={5000}")
            self.migrate()
        except Exception:
            # 初始化失败(库损坏、迁移出错)必须**关掉连接再抛出**:否则调用方虽然拿到了
            # 异常,文件句柄却一直开着 —— Windows 上连删除文件都会失败(WinError 32)。
            self.close()
            raise

    # ---------- 内务 ----------
    def close(self):
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def migrate(self) -> list:
        """把库升到 SCHEMA_VERSION,返回本次应用的版本号列表。

        **必须能容忍"上次没做完"**:迁移是"先改结构、再写版本号"两步,中间进程被杀会
        留下"结构已改、版本号还旧"的库;直接重跑 SQL 会报 `duplicate column name`,
        那个库就再也打不开了(记忆与会话会一起降级成 JSON 回退)。所以每条迁移在应用前都先用
        `_MIGRATION_APPLIED` 里的判断看一遍:结构已经是目标样子就**只补版本号**。
        """
        applied = []
        with self._lock:
            current = self.schema_version()
            for version in sorted(_MIGRATIONS):
                if version <= current:
                    continue
                check = _MIGRATION_APPLIED.get(version)
                if check is not None and check(self._conn):
                    # 上一次崩在"改完结构、还没写版本号"之间:补版本号即可,不重跑 SQL
                    with self._conn:
                        self._conn.execute(f"PRAGMA user_version={version}")
                    print(f"[!] 迁移 {version} 的结构已经存在(上次中断),只补上版本号")
                    applied.append(version)
                    continue
                with self._conn:                     # 事务
                    self._conn.executescript(_MIGRATIONS[version])
                    self._conn.execute(f"PRAGMA user_version={version}")
                applied.append(version)
            self._set_meta("schema_version", str(self.schema_version()))
        return applied

    def _set_meta(self, key, value):
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                               (str(key), str(value)))

    def set_meta(self, key, value):
        """公开的元数据写入(界面用它记"清空边界"这类标记)。"""
        with self._lock:
            self._set_meta(key, value)

    def get_meta(self, key, default=None):
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?",
                                     (str(key),)).fetchone()
        return row["value"] if row else default

    # ---------- 记忆 ----------
    def import_from_json(self, json_path=None, force=False):
        """把旧 `memory.json` 导入库;返回导入条数(0 表示未导入)。

        **必须**经 `kurumi_memory.load_memories_with_status()` 读取:损坏文件会被改名留档、
        GBK 能读出、读不到时不会把记忆当成空。绕开它自己读文件就会丢掉这些保护。
        """
        if not force and self.get_meta("imported_from_json"):
            return 0
        import kurumi_memory

        if json_path is not None:
            original = kurumi_memory.MEMORY_FILE
            kurumi_memory.MEMORY_FILE = str(json_path)
            try:
                memories, status = kurumi_memory.load_memories_with_status()
            finally:
                kurumi_memory.MEMORY_FILE = original
        else:
            memories, status = kurumi_memory.load_memories_with_status()

        if status == "missing":
            self._set_meta("imported_from_json", "skipped")
            self._set_meta("imported_status", status)
            return 0
        if status in ("corrupt", "read_error"):
            # 源文件有问题:不导入、也不打"已导入"标记,等主人修好后再试。
            # 原文件此时已被 kurumi_memory 改名留档(corrupt)或保持原样(read_error)。
            self._set_meta("imported_status", status)
            return 0
        self.save_memories(memories)
        self._set_meta("imported_from_json", "1")
        self._set_meta("imported_at", str(int(self.clock())))
        self._set_meta("imported_status", status)
        return len(memories)

    def load_memories(self) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT content, kind, source, created_at, updated_at, supersedes, "
                "pinned, active FROM memories ORDER BY id").fetchall()
        return [{
            "content": row["content"],
            "time": row["created_at"],
            "updated": row["updated_at"],
            "kind": row["kind"],
            "source": row["source"],
            "supersedes": row["supersedes"],
            "pinned": bool(row["pinned"]),
            "active": bool(row["active"]),
        } for row in rows]

    def save_memories(self, memories) -> None:
        """全量替换记忆(事务内):先清空再写入,避免留下已删除的旧行。"""
        import memory_model

        rows = []
        for item in memories or []:
            entry = memory_model.normalize_entry(item)
            if not entry:
                continue
            rows.append((
                entry["content"], entry["kind"], entry["source"],
                entry["time"], entry["updated"], entry.get("supersedes", ""),
                1 if entry.get("pinned") else 0, 1 if entry.get("active", True) else 0,
            ))
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM memories")
            self._conn.executemany(
                "INSERT OR REPLACE INTO memories "
                "(content, kind, source, created_at, updated_at, supersedes, pinned, active) "
                "VALUES (?,?,?,?,?,?,?,?)", rows)

    def export_json(self, path) -> None:
        """把库内容镜像导出成 JSON(原子写入)。

        这样主人回退到旧版本时数据仍在;导出的是**同样的结构**,旧代码的
        `_valid_entry` 会忽略多出来的字段,因此向下兼容。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp-export")
        payload = json.dumps(self.load_memories(), ensure_ascii=False, indent=2)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)

    # ---------- 会话 ----------
    def _session_stamp(self, session_id) -> str:
        """给会话算一个"保证它是最新的"时间戳。

        `latest_session()` 按 `updated_at DESC, id DESC` 取,而 `clock` 是**秒级**的:
        两个会话在同一秒里被写,排序就退化成随机 id 比较。所以只要库里还有别的会话,
        就把时间戳顶到"别的会话的最大值 + 1",让**正在写的这个会话一定是最新的**。

        注意只跟**别的**会话比:只有一个会话时时间戳就是当前时钟,不会随着每轮对话
        一路往前漂。`start_session` 与 `append_turn` **共用这一个**算法 ——
        两边各写一套的话,新会话刚登记完就会被第一轮对话用原始时钟覆盖回去
        (登记成 1001、写完第一轮又变 1000,重启就会载入旧对话)。
        """
        row = self._conn.execute(
            "SELECT MAX(CAST(updated_at AS INTEGER)) AS newest FROM sessions WHERE id != ?",
            (session_id,)).fetchone()
        other_newest = int(row["newest"] or 0)
        return str(max(int(self.clock()), other_newest + 1))

    def start_session(self, session_id, engine="") -> None:
        """登记/更新一个会话,并保证它成为**最新的那条**。"""
        with self._lock, self._conn:
            stamp = self._session_stamp(session_id)
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions (id, started_at, updated_at, engine) "
                "VALUES (?, COALESCE((SELECT started_at FROM sessions WHERE id = ?), ?), ?, ?)",
                (session_id, session_id, stamp, stamp, engine))

    def append_turn(self, session_id, role, content, engine="") -> int:
        """追加一轮,返回它的 `seq`(调用方用它回滚未完成的轮次)。

        写入前先 `INSERT OR IGNORE` 一条会话行 —— 万一 `sessions` 里没有它的父记录
        (登记失败、老库残缺),也不会写出一条"永远载入不到"的孤立轮次。
        """
        with self._lock, self._conn:
            stamp = self._session_stamp(session_id)
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, started_at, updated_at, engine) "
                "VALUES (?, ?, ?, ?)", (session_id, stamp, stamp, engine))
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM turns WHERE session_id = ?",
                (session_id,)).fetchone()
            seq = int(row["seq"]) + 1
            self._conn.execute(
                "INSERT INTO turns (session_id, seq, role, content, engine, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, seq, str(role), str(content), engine, str(int(self.clock()))))
            self._conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?",
                               (stamp, session_id))
            return seq

    def drop_turns_from(self, session_id, seq) -> int:
        """删掉 `seq` 及其之后的轮次(回滚一次**没有完成**的回合),返回删除条数。

        主人那句话在"发送时"就落盘(崩溃也留着),于是失败/取消时必须在库里
        把它撤掉 —— 否则重启恢复会话时,这些孤立/重复的轮次会被重新载入内存历史,
        污染上下文(成功聊天会留下 user/user/assistant,失败聊天会留下孤立 user)。
        """
        if seq is None:
            return 0
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM turns WHERE session_id = ? AND seq >= ?",
                (session_id, int(seq)))
            return int(cursor.rowcount or 0)

    def load_turns(self, session_id, limit=40, include_incomplete=False) -> list:
        """按时间顺序取最近的 limit 条轮次(用于重启后恢复会话)。

        `incomplete=1` 的轮次(崩溃前发出、没得到回答的提问)默认**不返回** ——
        它没有回答,放进模型上下文只会得到"两条 user 挨着"的坏序列。原始记录仍在库里,
        需要时用 `include_incomplete=True` 取出来。
        """
        clause = "" if include_incomplete else "AND COALESCE(incomplete, 0) = 0 "
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, engine, seq FROM turns WHERE session_id = ? "
                f"{clause}ORDER BY seq DESC LIMIT ?", (session_id, int(limit))).fetchall()
        return [{"role": row["role"], "content": row["content"], "engine": row["engine"],
                 "seq": row["seq"]}
                for row in reversed(rows)]

    def latest_session(self):
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC, id DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def clear_session(self, session_id) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def backup_file(self, suffix="bak") -> str:
        """用 **SQLite 在线备份 API** 复制整库,并校验备份真的可用。

        为什么不用 `shutil.copy2`:WAL 模式下"已提交但还没 checkpoint"的记录在 `-wal`
        文件里,只复制主库会得到一份**缺数据**的备份(源库 3 条时,`copy2` 备份 0 条)。
        用备份 API 则会把 WAL 里的内容一起写进目标库(同样场景下 3 条都在)。

        返回备份路径;**任何一步失败(含校验不过)都返回空串**,调用方必须据此放弃
        破坏性操作(见 `repair_turn_sequence`)。
        """
        try:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            target = self.path.with_name(f"{self.path.name}.{suffix}-{stamp}")
            with self._lock:
                if target.exists():
                    target.unlink()
                destination = sqlite3.connect(str(target))
                try:
                    self._conn.backup(destination)
                finally:
                    destination.close()
                if not self._verify_backup(target):
                    target.unlink(missing_ok=True)
                    return ""
            return str(target)
        except Exception as e:
            print(f"[!] 会话库备份失败: {e}")
            return ""

    def _verify_backup(self, target) -> bool:
        """备份必须"打得开、过得去完整性检查、条数与源库一致"才算数。"""
        try:
            source_counts = {table: self._conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("turns", "memories", "sessions")}
            conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
            try:
                if str(conn.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
                    print("[!] 备份校验失败:完整性检查未通过")
                    return False
                for table, expected in source_counts.items():
                    actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    if int(actual) != int(expected):
                        print(f"[!] 备份校验失败({table}):源库 {expected} 条,备份 {actual} 条")
                        return False
            finally:
                conn.close()
            return True
        except Exception as e:
            print(f"[!] 备份校验失败: {e}")
            return False

    def repair_duplicate_user_turns(self, session_id) -> int:
        """删掉"相邻两条内容完全相同的 user"里的后一条,返回删除条数(幂等)。

        数据迁移:提问在**发送时**与**完成时**各写一次,库里就会出现
        `user, user, assistant`。只动"内容一字不差"的相邻重复 ——
        主人连发两句不同的话必须原样保留。

        **任何一条带 `incomplete=1` 的相邻对都不算重复**。真实时序是
        `未完成的提问(旧) → 主人按提示原样重发(新) → 回答` —— 两句话一字不差,
        但新的那条是**有效提问**,旧的只是"上次没答上"的记录。按内容判重会删掉新的那条,
        恢复后只剩一条孤立 assistant。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, role, content, incomplete FROM turns WHERE session_id = ? "
                "ORDER BY seq", (session_id,)).fetchall()
            doomed = [row["seq"] for index, row in enumerate(rows)
                      if index and self._is_duplicate_user_pair(rows[index - 1], row)]
            if not doomed:
                return 0
            with self._conn:
                self._conn.executemany("DELETE FROM turns WHERE session_id = ? AND seq = ?",
                                       [(session_id, seq) for seq in doomed])
            return len(doomed)

    @staticmethod
    def _is_duplicate_user_pair(previous, current) -> bool:
        """相邻两条 user 是否算"同一条被写了两遍"。"""
        if previous["role"] != "user" or current["role"] != "user":
            return False
        if previous["content"] != current["content"]:
            return False
        return not previous["incomplete"] and not current["incomplete"]

    def mark_trailing_user_turn_incomplete(self, session_id) -> bool:
        """把末尾那条"提了但没得到回答"的 user 标成**未完成**,返回是否标了。

        为什么**不删**:那条消息是主人在崩溃前真的发出去的,`_begin_turn_persistence`
        特意在发送时就落盘,为的就是"进程被杀也不丢"。启动恢复把它删掉等于把承诺反着做。
        正确做法是保留原记录、打上标记,并在载入历史时**排除**它 —— 它没有回答,
        留在上下文里只会得到"两条 user 挨着"的坏序列。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT seq, role, incomplete FROM turns WHERE session_id = ? "
                "ORDER BY seq DESC LIMIT 1", (session_id,)).fetchone()
            if row is None or row["role"] != "user" or row["incomplete"]:
                return False
            with self._conn:
                self._conn.execute(
                    "UPDATE turns SET incomplete = 1 WHERE session_id = ? AND seq = ?",
                    (session_id, row["seq"]))
            return True

    def repair_turn_sequence(self, session_id, backup=True) -> dict:
        """启动恢复时的一次性整理(数据迁移)。

        做两件事,别的什么都不碰:

        1. 删掉"相邻且内容一字不差"的重复提问 —— 同一句话的另一份还在,删的是纯噪声;
        2. 把末尾那条没有回答的提问**标记**为未完成(不删,见上面的说明)。

        安全性:**先备份、校验通过,才允许动手**。备份失败就直接放弃整理,
        宁可让旧数据留着,也不能在没有退路的情况下删东西。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, role, content, incomplete FROM turns WHERE session_id = ? "
                "ORDER BY seq", (session_id,)).fetchall()
        # 判重规则与 repair_duplicate_user_turns 必须**同一条**:
        # 任何带 incomplete=1 的相邻对都不算重复
        has_duplicate = any(index and self._is_duplicate_user_pair(rows[index - 1], row)
                            for index, row in enumerate(rows))
        has_unanswered = bool(rows) and rows[-1]["role"] == "user" \
            and not rows[-1]["incomplete"]
        if not (has_duplicate or has_unanswered):
            return {"duplicates": 0, "incomplete": 0, "backup": "", "aborted": False}

        backup_path = ""
        if backup:
            backup_path = self.backup_file("before-turn-repair")
            if not backup_path:
                print("[!] 会话库备份未成功,已放弃整理(不删、不改任何记录)")
                return {"duplicates": 0, "incomplete": 0, "backup": "", "aborted": True}

        duplicates = self.repair_duplicate_user_turns(session_id)
        incomplete = 1 if self.mark_trailing_user_turn_incomplete(session_id) else 0
        return {"duplicates": duplicates, "incomplete": incomplete,
                "backup": backup_path, "aborted": False}
