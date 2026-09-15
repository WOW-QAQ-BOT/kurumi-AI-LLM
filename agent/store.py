# -*- coding: utf-8 -*-
"""Agent 运行/工具事件/审批/来源的 SQLite 持久化。仅存脱敏数据。

线程安全:连接以 check_same_thread=False 创建(UI 与其他工作线程共享同一实例),
因此所有读写方法都在同一把 RLock 下串行执行,并设置 busy_timeout 容忍外部进程
(如另一个 AgentStore 实例或 sqlite3 命令行)短暂占用写锁,避免 database is locked。
"""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from agent.audit import redact, summarize_text
from agent.task_result import from_dict as _task_result_from_dict
from agent.task_result import has_side_effects as _has_side_effects
from agent.task_result import to_dict as _task_result_to_dict

# 终态集合:只有这些状态才写 finished_at;运行态/等待审批态保持 NULL
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "interrupted"})

# 遇到 database is locked 时的等待上限(毫秒),而不是立刻抛错
_BUSY_TIMEOUT_MS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  request TEXT NOT NULL,
  state TEXT NOT NULL,
  code TEXT NOT NULL DEFAULT '',
  message TEXT NOT NULL DEFAULT '',
  started_at REAL NOT NULL,
  finished_at REAL
);
CREATE TABLE IF NOT EXISTS tool_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  tool_name TEXT NOT NULL,
  arguments TEXT NOT NULL,
  result_code TEXT,
  result_summary TEXT,
  duration_ms INTEGER
);
CREATE TABLE IF NOT EXISTS approvals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  approval_id TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  arguments TEXT NOT NULL,
  decision TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  title TEXT,
  url TEXT NOT NULL,
  accessed_at REAL NOT NULL
);
"""

# 结构化任务结果落库,供重启后恢复"上次到底做成了什么"。
#
# 版本 1 = 上表这批原始 schema(已随项目发布,故**不要**改动它,
# 否则老库(PRAGMA user_version=1)会跳过本应执行的后续迁移)。
SCHEMA_VERSION = 3

_MIGRATIONS = {
    1: _SCHEMA,
    # task_result 存 JSON 字符串;老记录为 NULL,读取时按"没有结构化结果"处理。
    2: """
ALTER TABLE runs ADD COLUMN task_result TEXT;
""",
    # 运行要记住它属于**哪一段会话**。恢复"上次 Agent 结果"若按全局取最近一条,
    # 清空会话之后旧任务就会被注入新会话;按时间戳设"清空边界"也不可靠
    # (同秒/时钟回拨分不出先后)。绑定 session_id 之后这个问题从根上不存在。
    # 老记录留空串 —— 它们不属于任何已知会话,于是**不会**被注入。
    3: """
ALTER TABLE runs ADD COLUMN session_id TEXT NOT NULL DEFAULT '';
""",
}


def _column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


# 每条迁移的"是否已经生效"判断(与 memory.store 同一套做法)。
# 迁移是"先改结构、再写 user_version"两步,中间进程退出会留下"结构已改、版本号还旧"的库;
# 直接重跑 ALTER 会报 duplicate column name,那个库就再也打不开了。
_MIGRATION_APPLIED = {
    2: lambda conn: _column_exists(conn, "runs", "task_result"),
    3: lambda conn: _column_exists(conn, "runs", "session_id"),
}


# 结构化结果里"可能混入密钥"的字段(自由文本);其余字段(路径/URL/数字)原样保留。
_TASK_RESULT_TEXT_FIELDS = ("message", "summary")


def _redact_task_result_text(payload: dict) -> dict:
    """只对自由文本脱敏的结构化结果副本。

    为什么不能整体 `redact()`:它会把数字字段也打码(`prompt_tokens` → `"***"`),
    于是读取端把已知用量看成"未知";而产物路径/来源 URL 本来就需要保留 ——
    它们是"文件在哪"的唯一依据。
    """
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if _TASK_RESULT_TEXT_FIELDS[0] in out:
        out["message"] = redact(out["message"])
    actions = []
    for action in (out.get("actions") or []):
        if isinstance(action, dict):
            action = dict(action)
            if _TASK_RESULT_TEXT_FIELDS[1] in action:
                action["summary"] = redact(action["summary"])
        actions.append(action)
    if actions:
        out["actions"] = actions
    return out


class AgentStore:
    def __init__(self, path: Path, retention_days: int = 30, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        # 连接级重入锁:start_run 会调用 start_run_id,必须可重入
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            self.migrate()
        self.retention_days = retention_days
        self.purge_expired()
        # 建库即对账:上一次崩溃/关机残留的 running/awaiting_approval 必须标成 interrupted,
        # 否则 Agent 页永远显示那些早已不存在的进程"仍在运行/卡在审批"——
        # mark_interrupted_runs 没有别的调用者,建库时漏掉这一步就没有东西能纠正它们。
        # 前提是"一个进程只建一次 store":UI 侧复用已有连接,不重复构造。
        self.mark_interrupted_runs()

    def close(self):
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

    # ---------- schema 迁移 ----------
    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def migrate(self) -> list:
        """把库升到 SCHEMA_VERSION,返回本次应用的版本号列表。

        与 memory.store 用同一套做法:DDL 在事务里执行,中途失败整体回滚,
        不会留下"升了一半"的库。幂等 —— 已是最新版本时不执行任何语句。

        注意:迁移**必须能容忍重复执行**(老库里 ALTER 过的列再 ALTER 会报错),
        因此每条迁移在应用前都再校验一次先决条件。
        """
        applied = []
        with self._lock:
            current = self.schema_version()
            for version in sorted(_MIGRATIONS):
                if version <= current:
                    continue
                check = _MIGRATION_APPLIED.get(version)
                if check is not None and check(self._conn):
                    # 上一次崩在"改完结构、还没写版本号"之间:只补版本号,不重复 ALTER
                    with self._conn:
                        self._conn.execute(f"PRAGMA user_version={version}")
                    applied.append(version)
                    continue
                with self._conn:                     # 事务
                    self._conn.executescript(_MIGRATIONS[version])
                    self._conn.execute(f"PRAGMA user_version={version}")
                applied.append(version)
        return applied

    def _has_column(self, table: str, column: str) -> bool:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(r[1] == column for r in rows)

    # ---------- runs ----------
    def start_run(self, request: str, session_id: str = "") -> str:
        run_id = uuid.uuid4().hex
        self.start_run_id(run_id, request, session_id)
        return run_id

    def start_run_id(self, run_id: str, request: str, session_id: str = "") -> None:
        """登记一次运行;`session_id` 记它属于哪一段会话。"""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, request, state, started_at, session_id) "
                "VALUES (?,?,?,?,?)",
                (run_id, summarize_text(redact(request), 2000), "running", self.clock(),
                 str(session_id or "")),
            )

    def set_run_state(self, run_id: str, state, code: str = "", message: str = "") -> None:
        """更新运行状态;仅终态写 finished_at,非终态(含 awaiting_approval)保持 NULL。"""
        state_value = state.value if hasattr(state, "value") else str(state)
        finished_at = self.clock() if state_value in _TERMINAL_STATES else None
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET state=?, code=?, message=?, finished_at=? WHERE run_id=?",
                (state_value, code, summarize_text(redact(message), 2000), finished_at, run_id),
            )

    def set_task_result(self, run_id: str, result) -> None:
        """把结构化任务结果落库。

        `result` 可以是 `TaskResult`、也可以已经是 dict;为 None 时**不写**
        (保留可能已存在的旧值,避免把"没结果"覆盖成"明确为空")。

        **只对自由文本脱敏**:产物路径与来源 URL 是"文件在哪"的唯一依据,统计是数字,
        它们都必须原样保留。对整个 payload 调 `redact()` 会把
        `prompt_tokens` 这类数字字段也打码成 `"***"`,读取端于是把已知用量退化成"未知"
        (而且在库里留下无法解析的历史数据)。
        """
        if result is None:
            return
        payload = result if isinstance(result, dict) else _task_result_to_dict(result)
        payload = _redact_task_result_text(payload)
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            return                      # 无法序列化就不写:审计是旁路,不能成为故障源
        with self._lock, self._conn:
            self._conn.execute("UPDATE runs SET task_result=? WHERE run_id=?", (text, run_id))

    def get_task_result(self, run_id: str):
        """读回结构化结果;没有/损坏时返回 None(调用方按"没有结构化结果"处理)。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT task_result FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row or not row["task_result"]:
            return None
        try:
            data = json.loads(row["task_result"])
        except (TypeError, ValueError):
            return None
        return _task_result_from_dict(data)

    def latest_task_result(self, with_side_effects: bool = False, session_id=None):
        """最近一次**有结构化结果**的运行结果(供重启后恢复)。

        `with_side_effects=True` 时只认"确实改过文件"的那些 —— 界面恢复会话时
        只关心这一类(纯读失败没有恢复价值,反而会污染历史)。
        `session_id` 给定时只认属于该会话的运行。
        """
        return self.latest_task_result_entry(with_side_effects, session_id)[0]

    def latest_task_result_entry(self, with_side_effects: bool = False, session_id=None):
        """返回 `(TaskResult, started_at)`;没有则 `(None, None)`。

        `session_id` 给定时只认**属于该会话**的运行 —— 界面恢复会话时用当前会话的 id 查,
        于是"上一段对话里的任务结果"不会跨过清空边界复活。
        传 `None`(默认)表示不按会话过滤,只看全局最近一条(老调用方/审计用途)。

        老库里的行 `session_id=''`,不属于任何已知会话,因此按会话查时**不会**被选中。

        排序按 `rowid DESC`(**插入顺序**,SQLite 维护的单调序号),**不用** `started_at`:
        时间戳是浮点秒,同一秒内的两次运行分不出先后,系统时钟回拨时更会把新运行排到旧运行
        后面 —— 这两种情况都会取到**旧**任务,于是重启恢复出的是过时的结果。
        `started_at` 从此只承担"展示时间"的作用。
        """
        where = ["task_result IS NOT NULL"]
        params: list = []
        if session_id is not None:
            where.append("session_id = ?")
            params.append(str(session_id))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT task_result, started_at FROM runs WHERE {' AND '.join(where)} "
                "ORDER BY rowid DESC", params).fetchall()
        for row in rows:
            try:
                data = json.loads(row["task_result"])
            except (TypeError, ValueError):
                continue
            result = _task_result_from_dict(data)
            if result is None:
                continue
            if with_side_effects and not _has_side_effects(result):
                continue
            started = row["started_at"]
            return result, (float(started) if started is not None else None)
        return None, None

    def get_run(self, run_id: str):
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def mark_interrupted_runs(self) -> None:
        """把上次崩溃/关机残留的运行标记为 interrupted(终态),并补上结束时间。

        只改 state 而不写 finished_at 会与"终态才有结束时间"的语义不一致,
        事后统计运行时长时这类记录会被当成"仍在运行"。
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET state='interrupted', finished_at=? "
                "WHERE state IN ('running','awaiting_approval')",
                (self.clock(),),
            )

    # ---------- events ----------
    def record_tool_event(self, run_id: str, event) -> None:
        payload = redact(getattr(event, "payload", {}) or {})
        # 工具名来自模型输出:入库前限长并统一成字符串(payload 里可能被写成 dict)
        args = summarize_text(str(payload.get("tool_name", "")), 200)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO tool_events (run_id, seq, tool_name, arguments, result_code, result_summary, duration_ms) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    run_id,
                    getattr(event, "sequence", 0),
                    args,
                    # 工具参数同样要脱敏:模型可以把密钥当参数传给工具(如 file_write 的 content)
                    summarize_text(redact(str(payload.get("arguments", ""))), 1000),
                    str(payload.get("code", "")),
                    summarize_text(redact(str(payload.get("summary", ""))), 500),
                    int(payload.get("duration_ms", 0) or 0),
                ),
            )

    def record_approval(self, run_id: str, approval) -> None:
        decision = getattr(approval, "allowed", None)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO approvals (run_id, approval_id, tool_name, arguments, decision, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    run_id,
                    getattr(approval, "approval_id", getattr(approval, "id", "")),
                    getattr(approval, "tool_name", ""),
                    summarize_text(str(redact(getattr(approval, "canonical_arguments", {}))), 1000),
                    "allow" if decision is True else ("deny" if decision is False else "pending"),
                    self.clock(),
                ),
            )

    def record_approval_decision(self, approval_id: str, allowed: bool) -> None:
        """审批最终决定回写:允许/拒绝,替换初始 pending。"""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE approvals SET decision=? WHERE approval_id=?",
                ("allow" if allowed else "deny", approval_id),
            )

    def record_source(self, run_id: str, source: dict) -> None:
        src = redact(source or {})
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sources (run_id, title, url, accessed_at) VALUES (?,?,?,?)",
                (run_id, str(src.get("title", "")), str(src.get("url", "")), self.clock()),
            )

    # ---------- maintenance ----------
    def purge_expired(self) -> None:
        cutoff = self.clock() - self.retention_days * 86400
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,))
            self._conn.execute(
                "DELETE FROM tool_events WHERE run_id NOT IN (SELECT run_id FROM runs)"
            )
            self._conn.execute(
                "DELETE FROM approvals WHERE run_id NOT IN (SELECT run_id FROM runs)"
            )
            self._conn.execute(
                "DELETE FROM sources WHERE run_id NOT IN (SELECT run_id FROM runs)"
            )

    def clear_all(self) -> None:
        """清空 Agent 审计记录;不触碰外部文件(如人物记忆 memory.json)。"""
        with self._lock, self._conn:
            for table in ("tool_events", "approvals", "sources", "runs"):
                self._conn.execute(f"DELETE FROM {table}")
