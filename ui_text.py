# -*- coding: utf-8 -*-
"""气泡文本与几何(从 UI.py 拆出)。

这些函数只依赖 Qt 的 QLabel 与常量,不依赖窗口状态,因此可以独立成模块。

**关键设计**:模块内部互相调用时一律走**本模块的名字**(例如 `_append_bubble_text`
调用本模块的 `_relayout_bubble`),而 `UI.py` 只是重导出它们。这样"替换 `UI.<name>`"
只影响外部调用方,不会让模块内的既有调用半新半旧。谁若把内部调用改成走 `UI.<name>`,
就会绕开本模块的实现 —— 改之前先想清楚。
"""
import re
import time as _time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from ui_constants import (
    BUBBLE_GEOM_INTERVAL_S,
    BUBBLE_MIN_WIDTH,
    BUBBLE_TEXT_PADDING,
    BUBBLE_ZWSP_MIN_RUN,
    BUBBLE_ZWSP_STEP,
)


def now() -> float:
    """时钟(与 time.monotonic 同义);独立成函数便于替换。"""
    return _time.monotonic()


def _bubble_text(text):
    """长无空格 token(URL、路径)按固定步长插入零宽空格,保证 QLabel 完整换行显示,不被视觉裁剪。"""

    def _break(match):
        token = match.group(0)
        return "\u200b".join(
            token[i:i + BUBBLE_ZWSP_STEP] for i in range(0, len(token), BUBBLE_ZWSP_STEP)
        )

    # \S{40,}:仅对超过 BUBBLE_ZWSP_MIN_RUN 的无空格长串做折断(花括号需转义)
    pattern = rf"\S{{{BUBBLE_ZWSP_MIN_RUN},}}"
    return re.sub(pattern, _break, str(text))


def _bubble_text_width(text, fm, max_width):
    """按文字实际宽度算气泡宽度(取最长行,夹在 [最小宽度, max_width])。"""
    ideal = max((fm.horizontalAdvance(line) for line in text.splitlines()), default=0)
    return max(BUBBLE_MIN_WIDTH, min(ideal + BUBBLE_TEXT_PADDING, int(max_width)))


def _relayout_bubble(label, text):
    """重算气泡固定宽高;宽高未变化时不重复 setFixed*(避免无谓的整窗重排)。"""
    if label is None:
        return
    max_width = getattr(label, "_max_width", 400)
    width = _bubble_text_width(text, label.fontMetrics(), max_width)
    if width != getattr(label, "_cur_width", None):
        label.setFixedWidth(width)
        label._cur_width = width
    height = label.heightForWidth(width)
    if height != getattr(label, "_cur_height", None):
        label.setFixedHeight(height)
        label._cur_height = height


def _relayout_after_mount(view):
    """控件挂进窗口后再重算一次固定宽高。

    坑:QSS 里的 `font-size: 14px` 只有在控件进入窗口(style polish)后才生效,而
    `_make_bubble_view` 是在**无父**状态下用默认字体(9pt)算几何的 —— 默认字体比 QSS
    的小,算出的固定高度偏小,用户消息这类"创建后不再重算"的气泡最后一行就会被裁掉
    (实机表现:气泡里"少了一部分文字")。因此必须在插入布局之后补一次重算。
    """
    if view is None:
        return
    view.ensurePolished()                      # 触发样式解析,让 QSS 字体先生效
    _relayout_bubble(view, view.text())


def _make_bubble_view(text, name, max_width):
    """气泡标签:显式固定宽高(按文字实际宽度与换行高度计算),规避布局裁剪与塌缩。"""
    label = QLabel()
    label.setObjectName(name)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label._max_width = int(max_width)
    text = _bubble_text(text)
    label.setText(text)
    _relayout_bubble(label, text)
    return label


def _set_bubble_text(label, text):
    """一次性替换气泡内容（最终回复、错误提示等）并重算固定宽高。"""
    if label is None:
        return
    text = _bubble_text(text)
    label.setText(text)
    _relayout_bubble(label, text)


def _append_bubble_text(label, full_text, throttle_s=BUBBLE_GEOM_INTERVAL_S):
    """流式追加：先出字，宽高重算按 throttle_s 节流。

    返回本次是否重算了宽高，供调用方决定是否需要滚动到底部。
    """
    if label is None:
        return False
    text = _bubble_text(full_text)
    label.setText(text)
    current = now()
    if current - getattr(label, "_last_relayout", 0.0) < throttle_s:
        return False
    label._last_relayout = current
    _relayout_bubble(label, text)
    return True
