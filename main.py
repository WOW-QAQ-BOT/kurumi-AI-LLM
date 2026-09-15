# -*- coding: utf-8 -*-
"""时崎狂三 · 启动入口。

用法：
  python main.py               # 自动选择：目录下有 api_config.json（填了 Key）走 DeepSeek API，否则本地模型 + LoRA
  python main.py --local       # 强制本地模型模式（忽略显存检测与 api_config.json）
  python main.py --no-local    # 禁用本地模型加载（仅使用 DeepSeek API）
  python main.py --no-quantize # 本地模式切换 bf16 全精度
  python main.py --base <模型目录> --adapter <LoRA目录>

本文件只负责"**怎么启动**"：解析参数 → 建 QApplication → 建窗口 → 跑事件循环 →
退出前给保活线程一次限时收尾。界面实现全在 UI.py（KurumiWindow）与 ui_*.py。

这样分开的理由：启动参数（用哪个引擎、用哪份权重）与界面实现是两件独立的事，
改启动方式不必动那个两千行的窗口文件。参数含义见 README 第 6 节。
"""
import argparse
import sys

from PySide6.QtWidgets import QApplication

from UI import LOCAL_ADAPTER, LOCAL_BASE, KurumiWindow, _wait_for_orphaned_threads


def build_parser():
    """命令行参数。默认值由 UI.py 按"脚本同目录 → 上一级目录 → HuggingFace"探测得出。"""
    p = argparse.ArgumentParser(
        prog="main.py",
        description="时崎狂三 · 桌面聊天（本地模型 + LoRA，或 DeepSeek API）",
    )
    p.add_argument("--base", default=LOCAL_BASE, help="本地基座模型目录")
    p.add_argument("--adapter", default=LOCAL_ADAPTER, help="LoRA 适配器目录")
    p.add_argument("--no-quantize", action="store_true", help="本地模式改用 bf16 全精度加载")
    p.add_argument("--local", action="store_true", help="强制本地模型模式（忽略显存检测与 api_config.json）")
    p.add_argument("--no-local", action="store_true", help="禁用本地模型加载（仅使用 API）")
    p.add_argument("--min_vram_gb", type=float, default=4.0, help="自动模式选择本地模型所需的最小显存余量(GiB)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--remember_every", type=int, default=5, help="每 N 轮回复自动整理一次记忆（0 表示禁用）")
    p.add_argument("--history_max_chars", type=int, default=16000, help="对话历史最大字符数，超出自动裁剪旧内容")
    p.add_argument("--api_timeout_seconds", type=float, default=45, help="API 请求超时（秒）")
    p.add_argument("--api_thinking", choices=["disabled", "low", "high", "max"], default="disabled",
                   help="DeepSeek API 思考模式:disabled 关闭(默认;省 token 且让 temperature/top_p 生效)"
                        "/low/high/max 开启并指定思考强度")
    p.add_argument("--stream_stall_timeout", type=float, default=60, help="本地生成流停滞判定（秒）")
    p.add_argument("--shutdown_wait_ms", type=int, default=2000, help="关闭窗口时等待任务结束的毫秒数")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.local and args.no_local:
        parser.error("--local 与 --no-local 不能同时使用")

    app = QApplication([])
    win = KurumiWindow(args)
    win.show()
    app.exec()
    # 事件循环已结束：给关闭时保活下来的线程一次限时收尾机会，
    # 尽量让它们别在解释器析构 QThread 时还在运行（那会触发 Qt qFatal abort）。
    _wait_for_orphaned_threads()
    return 0


if __name__ == "__main__":
    sys.exit(main())
