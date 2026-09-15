# -*- coding: utf-8 -*-
"""组合执行器:文件工具 + 系统工具 (+ 可选的网页读取),统一 prepare/execute 契约与注册表。"""
from agent.tools.base import ToolRegistry
from agent.tools.files import build_file_executor
from agent.tools.system import build_system_executor
from agent.tools.web import WebToolExecutor


def _registry_with(registry, keep):
    """按名字过滤注册表:只开搜索时不要把网页读取一起放出去(反之亦然)。"""
    filtered = ToolRegistry()
    for definition in registry.definitions():
        if definition.name in keep:
            filtered.register(definition)
    return filtered


class CompositeExecutor:
    def __init__(self, file_executor, system_executor, web_executor=None):
        self.file = file_executor
        self.system = system_executor
        self.web = web_executor
        self.registry = ToolRegistry()
        for d in self.file.registry.definitions():
            self.registry.register(d)
        for d in self.system.registry.definitions():
            self.registry.register(d)
        if self.web is not None:
            for d in self.web.registry.definitions():
                self.registry.register(d)

    def _executor_for(self, name):
        if name in self.file.registry.names():
            return self.file
        if name in self.system.registry.names():
            return self.system
        if self.web is not None and name in self.web.registry.names():
            return self.web
        return None

    def prepare(self, call):
        target = self._executor_for(call.name)
        if target is None:
            raise KeyError(call.name)
        return target.prepare(call)

    def execute(self, prepared, approval=None):
        target = self._executor_for(prepared.call.name)
        if target is not None:
            return target.execute(prepared, approval)
        from agent.types import ToolResult
        return ToolResult.denied(prepared.call.id, f"未注册工具: {prepared.call.name}")


class _EmptySystemExecutor:
    """extra_tools 关闭时的占位:不注册任何系统工具。"""

    def __init__(self):
        self.registry = ToolRegistry()

    def prepare(self, call):
        raise KeyError(call.name)

    def execute(self, prepared, approval=None):
        from agent.types import ToolResult
        return ToolResult.denied(prepared.call.id, "系统工具已禁用")


def build_agent_executor(policy, trash_adapter=None, opener=None, psutil_mod=None,
                         include_system: bool = True, include_web: bool = False,
                         web_opener=None, web_timeout: float = 10.0,
                         include_web_search: bool = False, web_search_proxy: str = "",
                         web_search_timeout: float = 15.0, web_search_engine: str = "auto",
                         web_search_max_results: int = 5,
                         web_search_allow_private: bool = False) -> CompositeExecutor:
    """组装执行器。

    `include_web` / `include_web_search` 默认 **False**:网页工具必须显式开启
    (`agent.web_tools_enabled` / `agent.web_search_enabled`),
    否则**不注册任何网页工具,默认行为保持不变**。

    `web_search` 由 ddgs 库提供;`web_search_proxy` 为空时直连,
    受限链路(本机需要代理才能访问搜索引擎)必须填它才搜得到。
    `web_search_max_results` 是模型没给 `count` 时的默认条数。
    """
    system = build_system_executor(policy, opener=opener, psutil_mod=psutil_mod) if include_system \
        else _EmptySystemExecutor()
    web = None
    if include_web or include_web_search:
        web = WebToolExecutor(policy, opener=web_opener, timeout=web_timeout,
                              search_proxy=web_search_proxy,
                              search_timeout=web_search_timeout,
                              search_engine=web_search_engine,
                              search_max_results=web_search_max_results,
                              allow_private=web_search_allow_private)
        if not include_web:
            # 只开搜索时,不要把 web_fetch 也顺带放出去
            web.registry = _registry_with(web.registry, keep=("web_search",))
        if not include_web_search:
            web.registry = _registry_with(web.registry, keep=("web_fetch",))
    return CompositeExecutor(
        build_file_executor(policy, trash_adapter=trash_adapter),
        system,
        web,
    )
