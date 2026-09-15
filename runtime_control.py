# -*- coding: utf-8 -*-
"""共享的取消原语。普通聊天与 Agent 执行器共用，避免在 Qt UI 中引入 agent 依赖。"""
from threading import Event


class CancellationToken:
    """一次性取消令牌：线程安全、幂等。

    正常流程是"一次运行一个令牌"：新建 → 需要中断时 ``cancel()`` → 执行方用
    ``cancelled`` 轮询或 ``wait(timeout)`` 阻塞等待。``cancel/cancelled/wait``
    的语义与线程安全保证保持不变（UI 与 agent 都依赖它们）。
    """

    def __init__(self):
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def reset(self) -> None:
        """把令牌恢复为未取消状态。

        仅用于**复用同一个令牌**跑多次运行的场景（例如调试脚本连续调用 runner）。
        正在运行的执行方不会被回滚：它们可能在 reset 之前已经读到过 cancelled=True。
        因此一般应当每次运行新建令牌，而不是复用后 reset。
        """
        self._event.clear()

    def __repr__(self) -> str:
        return f"CancellationToken(cancelled={self.cancelled})"
