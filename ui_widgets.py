# -*- coding: utf-8 -*-
"""聊天区控件构建(从 UI.py 拆出)。

统一约定:每个函数的第一个参数是**窗口对象**,只使用下面这组最小接口 ——

- `chat_layout`           聊天区布局
- `width()`              窗口宽度(算气泡最大宽度)
- `_bubble_inner_width()` 文本可用宽度
- `_scroll_bottom()`      滚到底部
- `agent_controller`      审批卡回调(允许一次/拒绝)

这样"怎么画"可与窗口解耦,单独验证;窗口只负责状态与业务。
"""
import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout

from ui_constants import BUBBLE_H_PADDING, BUBBLE_WIDTH_RATIO
from ui_text import _make_bubble_view, _relayout_after_mount


def _insert_card(window, card, view):
    """卡片插到聊天区末尾(输入区之前),挂载后重算宽高并滚到底。"""
    window.chat_layout.insertWidget(window.chat_layout.count() - 1, card, 0, Qt.AlignLeft)
    _relayout_after_mount(view)
    QTimer.singleShot(0, window._scroll_bottom)
    return card


def add_bubble(window, text, is_user):
    """在聊天区追加一个气泡,返回文本标签(调用方后续用它更新内容)。"""
    bubble = QFrame()
    bubble.setObjectName("bubbleUser" if is_user else "bubbleBot")
    bubble.setMaximumWidth(int(window.width() * BUBBLE_WIDTH_RATIO))
    bl = QVBoxLayout(bubble)
    bl.setContentsMargins(12, 8, 12, 8)
    view = _make_bubble_view(text, "msgUser" if is_user else "msgBot",
                             window._bubble_inner_width())
    bl.addWidget(view)
    window.chat_layout.insertWidget(window.chat_layout.count() - 1, bubble, 0,
                                    Qt.AlignRight if is_user else Qt.AlignLeft)
    _relayout_after_mount(view)            # QSS 字体生效后重算,避免"少字"
    QTimer.singleShot(0, window._scroll_bottom)
    return view


def add_neutral_card(window, text):
    """中性提示卡(虚线边框):工具结果、来源、任务面板等。"""
    card = QFrame()
    card.setObjectName("bubbleBot")
    card.setStyleSheet("QFrame#bubbleBot { border: 1px dashed #5a3a44; }")
    bl = QVBoxLayout(card)
    view = _make_bubble_view(text, "msgBot", window._bubble_inner_width())
    bl.addWidget(view)
    return _insert_card(window, card, view)


def add_action_card(window, text, buttons):
    """带按钮的卡片(审批卡、迁移确认卡);回调收到卡片本身。"""
    card = QFrame()
    card.setObjectName("bubbleBot")
    bl = QVBoxLayout(card)
    view = _make_bubble_view(text, "msgBot", window._bubble_inner_width())
    bl.addWidget(view)
    row = QHBoxLayout()
    for label_text, callback in buttons:
        btn = QPushButton(label_text)
        btn.setObjectName("clear")
        btn.clicked.connect(lambda _=False, b=btn, cb=callback, c=card: cb(c))
        row.addWidget(btn)
    row.addStretch()
    bl.addLayout(row)
    return _insert_card(window, card, view)


def add_approval_card(window, event):
    """审批卡:允许一次 / 拒绝。批准仅对本次精确调用有效。

    **风险说明必须说清数据去哪**:不能只凭 `data_leaves_device` 为真就固定写
    "该操作会把文件内容发送至 DeepSeek 云端" —— 这对 `web_search` 是**错的**:
    搜索外发的是关键词、去的是搜索引擎,与 DeepSeek 无关。
    """
    payload = event.payload
    args_text = json.dumps(payload.get("arguments", {}), ensure_ascii=False, indent=2)
    tool_name = payload.get("tool_name", "?")
    leaves = bool(payload.get("data_leaves_device"))
    if not leaves:
        egress = ""
    elif tool_name == "web_search":
        # 必须说清**两段外发** —— 请求发给搜索引擎,结果随后回传给 DeepSeek。
        # 只写"不是 DeepSeek 云端"会漏掉后半段(工具结果会作为 function_call_output
        # 回传),让主人以为这次操作跟云端无关。
        egress = ("⚠️ 两段外发:①搜索关键词会发给搜索引擎;"
                  "②搜索到的摘要随后会随对话发给 DeepSeek")
    elif tool_name == "web_fetch":
        egress = ("⚠️ 两段外发:①目标网址会发给该站点;"
                  "②读取到的正文随后会随对话发给 DeepSeek")
    elif tool_name == "open_item":
        egress = "⚠️ 该操作会打开本机上的目标(不涉及云端)"
    else:
        # 其余情况(如读取文件内容并进入模型上下文)才适用"发往云端"的说法
        egress = "⚠️ 该操作会把文件内容发送至 DeepSeek 云端"
    text = (
        f"【需要主人批准】工具: {tool_name}\n"
        f"参数:\n{args_text}\n"
        f"风险: {payload.get('risk_summary', '')}\n"
        f"{egress}\n"
        f"批准仅对本次精确调用有效。"
    )
    approval_id = payload.get("approval_id", "")

    def allow(card):
        card.setEnabled(False)
        if window.agent_controller:
            window.agent_controller.approve_once(approval_id)

    def deny(card):
        card.setEnabled(False)
        if window.agent_controller:
            window.agent_controller.deny(approval_id)

    return add_action_card(window, text, [("允许一次", allow), ("拒绝", deny)])


def inner_width(window_width, ratio=BUBBLE_WIDTH_RATIO, padding=BUBBLE_H_PADDING):
    """气泡内文本可用宽度(与 KurumiWindow._bubble_inner_width 同一算法)。

    抽成纯函数是为了让"宽度算法"能被单独验证:它一旦变了,所有气泡的换行位置都会变。
    """
    return int(window_width * ratio) - padding
