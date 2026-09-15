# -*- coding: utf-8 -*-
"""Agent 共享数据契约:不可变数据结构与枚举。"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunState(str, Enum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    code: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def denied(cls, call_id: str, reason: str):
        return cls(call_id, False, "POLICY_DENIED", reason)

    @classmethod
    def user_denied(cls, call_id: str):
        return cls(call_id, False, "USER_DENIED", "User denied this operation")


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    canonical_arguments: dict[str, Any]
    arguments_hash: str
    risk_summary: str
    data_leaves_device: bool
    file_fingerprint: dict[str, Any] | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    approval_id: str
    allowed: bool

    @classmethod
    def allow_once(cls, approval_id: str):
        return cls(approval_id=approval_id, allowed=True)

    @classmethod
    def deny(cls, approval_id: str):
        return cls(approval_id=approval_id, allowed=False)


@dataclass(frozen=True)
class AgentEvent:
    run_id: str
    sequence: int
    kind: str
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProtocolError:
    code: str
    message: str


@dataclass(frozen=True)
class ModelTurn:
    text: str = ""
    output_items: tuple[dict[str, Any], ...] = ()
    function_calls: tuple[ToolCall, ...] = ()
    web_actions: tuple[dict[str, Any], ...] = ()
    citations: tuple[dict[str, Any], ...] = ()
    protocol_errors: tuple[ProtocolError, ...] = ()
    # 端点返回的用量统计。拿不到时为 None —— 不能猜,
    # 否则统计会把"没有数据"伪装成 0 用量(比不显示更糟)。
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class RunOutcome:
    state: RunState
    code: str
    message: str
    # 可核对的结构化结果(为 None 时不影响既有行为)
    task_result: Any = None
