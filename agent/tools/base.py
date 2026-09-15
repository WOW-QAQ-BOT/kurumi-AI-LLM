# -*- coding: utf-8 -*-
"""工具定义、注册表与审批绑定契约。工具实现不直接暴露给模型。"""
import copy
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from agent.types import ToolCall


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool


@dataclass(frozen=True)
class PreparedToolCall:
    call: ToolCall
    definition: ToolDefinition
    canonical_arguments: dict[str, Any]
    arguments_hash: str
    policy: Any
    file_fingerprint: dict[str, Any] | None = None


@dataclass(frozen=True)
class ApprovalGrant:
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments_hash: str
    file_fingerprint: dict[str, Any] | None

    @classmethod
    def from_prepared(cls, run_id: str, prepared: PreparedToolCall):
        return cls(
            run_id=run_id,
            tool_call_id=prepared.call.id,
            tool_name=prepared.call.name,
            arguments_hash=prepared.arguments_hash,
            file_fingerprint=prepared.file_fingerprint,
        )

    def validate(self, prepared: PreparedToolCall) -> None:
        """精确匹配校验:调用 ID、工具名、参数哈希与文件指纹任一不符即拒绝。

        **参数哈希必须按当前参数重算**,不能只比字符串:`PreparedToolCall` 虽是
        frozen dataclass,但它的 `canonical_arguments` 是**可变 dict** —— 谁拿到
        prepared 之后改一下参数,缓存的哈希不会跟着变,已签发的授权就会继续放行。
        重算一次的成本可以忽略,换来的是"改了参数就执行不了"这个契约真的成立。
        """
        from agent.policy import hash_arguments

        current_hash = self.arguments_hash == prepared.arguments_hash
        arguments_unchanged = True
        try:
            arguments_unchanged = (
                hash_arguments(prepared.canonical_arguments or {}) == self.arguments_hash)
        except (TypeError, ValueError):
            arguments_unchanged = False      # 参数已不可哈希/结构被改坏 → 一律视为不符

        # 注:这里**不能**检查"出口是否被改" —— 本方法是 ApprovalGrant 上的,
        # `self` 是那份**冻结的授权**,拿不到执行器的当前配置:写成
        # `getattr(self, "_egress_changed")` 恒为 None,检查形同虚设。
        # 出口检查必须放在执行器自己的 `execute()` 里(`WebToolExecutor._egress_changed`),
        # 因为只有那里同时知道"批准的出口"与"现在的出口"。
        if (
            self.tool_call_id != prepared.call.id
            or self.tool_name != prepared.call.name
            or not current_hash
            or not arguments_unchanged
            or self.file_fingerprint != prepared.file_fingerprint
        ):
            raise PermissionError("approval does not match prepared call")


class ToolRegistry:
    """注册表:注册时校验 Schema,prepare 时拒绝未知工具、多余字段与错误类型。"""

    def __init__(self):
        self._defs: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        Draft202012Validator.check_schema(definition.parameters)
        self._defs[definition.name] = definition

    def names(self) -> list[str]:
        return list(self._defs)

    def definitions(self) -> list[ToolDefinition]:
        return list(self._defs.values())

    def prepare(self, name: str, arguments: dict[str, Any]) -> tuple[ToolDefinition, dict[str, Any]]:
        """查找定义并严格校验参数(additionalProperties:false 拒绝多余字段)。

        未知工具名抛 KeyError;参数非法抛 ValueError;返回规范化深拷贝。
        """
        definition = self._defs[name]
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        errors = sorted(
            Draft202012Validator(definition.parameters).iter_errors(arguments),
            key=lambda e: list(e.absolute_path),
        )
        if errors:
            raise ValueError("参数校验失败: " + errors[0].message)
        return definition, copy.deepcopy(arguments)
