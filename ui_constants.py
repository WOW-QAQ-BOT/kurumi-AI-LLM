# -*- coding: utf-8 -*-
"""UI 层可调常量(从 UI.py 拆出)。

拆出来的理由:`ui_workers.py` 与 `UI.py` 都要用这些数字,而 `ui_workers.py`
**不能**反向导入 `UI.py`(会形成循环导入)。因此放到无依赖的独立模块。

注意 `MIN_NEW_TOKENS` / `REPETITION_PENALTY` / `MEMORY_MAX_NEW_TOKENS` 会直接影响
本地模型的生成行为(取值不当会让本地模型空回复或重复):改动之后务必在本地模式下**手工**跑一遍,
确认不空、不重复、不半途截断(项目无自动化测试,只能靠这次手工运行兜底)。
"""

MIN_NEW_TOKENS = 30              # 生成至少产出的 token 数，抑制空回复
REPETITION_PENALTY = 1.15        # 本地生成重复惩罚
MEMORY_MAX_NEW_TOKENS = 512      # 记忆整理的最大输出长度(思考模式下会被思考占用,故留足余量)
GEN_JOIN_TIMEOUT_S = 10          # 等待底层 generate 线程退出的秒数

BUBBLE_WIDTH_RATIO = 0.82        # 气泡最大宽度占窗口宽度比例
BUBBLE_H_PADDING = 24            # 气泡左右内边距合计（与 QSS 的 12px 对应）
BUBBLE_TEXT_PADDING = 12         # 文字两侧留白，避免贴边裁字
BUBBLE_MIN_WIDTH = 60            # 气泡最小宽度
BUBBLE_ZWSP_STEP = 24            # 长无空格 token 每隔 N 字符插一个零宽空格
BUBBLE_ZWSP_MIN_RUN = 40         # 只有这么长的无空格串才需要插零宽空格
BUBBLE_GEOM_INTERVAL_S = 0.05    # 流式输出时重算气泡固定宽高的最小间隔（节流 ~50ms）

ORPHAN_EXIT_WAIT_S = 2.0         # 退出前对保活线程的限时等待总上限

# ==================== 凭据外发域名(安全相关,改动需谨慎) ====================
# 信任根之一:内置官方域名。另一个是用户在确认卡上点过「允许」并写入凭据管理器的主机。
OFFICIAL_API_HOSTS = ("https://api.deepseek.com",)
ALLOWED_HOST_ACCOUNT_PREFIX = "allowed-host:"
ALLOWED_HOST_CONFIRMED_VALUE = "confirmed"
