# -*- coding: utf-8 -*-
"""
时崎狂三 · 原生桌面聊天 UI（PySide6）

启动入口在 **main.py**：`python main.py [--local|--no-local|...]`。
本文件只放界面实现（KurumiWindow 与 ui_*.py 的组装），不解析命令行参数 ——
"用哪个引擎、用哪份权重"由 main.py 决定后交给 KurumiWindow。

**本模块不在导入期加载本地推理栈**（torch/transformers/peft）。
纯 API 模式（--no-local）因此只需最小依赖集即可启动；需要推理栈的地方
（_check_vram_gb）用到时才导入。
"""
import copy
import json
import os
import subprocess
import time
import uuid
from threading import Thread

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import task_intent
from kurumi.conversation import ENGINE_AGENT, ENGINE_API, ENGINE_LOCAL, ConversationService
from kurumi.memory import (
    add_memories,
    attach_store,
    load_memories,
    load_memories_store_first,
    save_memories_store_first,
)
from memory import commands as memory_commands
from memory.store import MemoryStore
from runtime_control import CancellationToken

# 人设与知识库现在统一经 ContextBuilder 组装(见 kurumi.context),UI 不再直接依赖它们
#
# 后台线程/推理栈/思考层已拆到 ui/workers.py、chat_params.py,
# 这里不再导入 transformers(本地推理依赖只出现在 ui/workers.py)。

try:
    # OpenAI 只用于 `_HAS_OPENAI` 探测与未装 openai 时的占位异常;
    # 实际调用在 ui/workers.py(它自己 import)。因此这里不能删掉这次导入。
    from openai import BadRequestError, OpenAI  # noqa: F401
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

    class BadRequestError(Exception):     # 占位:未装 openai 时也不会在导入期炸
        pass

# Agent 模块依赖 jsonschema/keyring/send2trash;缺失时普通聊天照常可用
try:
    from agent.config import DEFAULT_CONFIG_PATH
    from agent.config import load_config as _load_agent_config
    from agent.controller import AgentController
    from agent.credentials import CredentialStore, migrate_legacy_key
    from agent.model.deepseek import DeepSeekAgentClient
    from agent.policy import PermissionPolicy
    from agent.runner import AgentRunner
    from agent.store import AgentStore
    from agent.task_result import has_side_effects as agent_has_side_effects
    from agent.task_result import history_note as agent_history_note
    from agent.task_result import state_label as agent_state_label
    from agent.task_result import task_panel_text as agent_task_panel
    from agent.tools.executor import build_agent_executor
    from agent.types import RunState
    _AGENT_AVAILABLE = True
except ImportError:
    _AGENT_AVAILABLE = False

    # Agent 依赖缺失时给出退化实现:面板不渲染、历史不保留,
    # 与"没有 task_result 字段"的情况一致(不做任何猜测)。
    def agent_has_side_effects(result):
        return False

    def agent_history_note(result):
        return ""

    def agent_task_panel(result):
        return ""

    def agent_state_label(result):
        return ""


# ==================== 可调常量与拆分模块（实现见各自文件） ====================
# 常量、后台线程、凭据校验各自独立成模块。
# 这里保留**同名绑定**是刻意的:调用方(以及将来任何调试脚本)仍以 `ui.<name>` 访问,
# 例如按端点能力调整 `ui._API_THINKING_*`、替换 `ui._HERE` / `ui._HAS_OPENAI`。
from api.config import (  # noqa: F401
    ALLOWED_HOST_ACCOUNT_PREFIX,
    ALLOWED_HOST_CONFIRMED_VALUE,
    OFFICIAL_API_HOSTS,
    _confirmed_host_account,
    _host_is_allowed,
    _is_candidate_host,
    _new_credential_store,
    _normalize_host,
)
from ui.constants import (  # noqa: F401
    BUBBLE_GEOM_INTERVAL_S,
    BUBBLE_H_PADDING,
    BUBBLE_MIN_WIDTH,
    BUBBLE_TEXT_PADDING,
    BUBBLE_WIDTH_RATIO,
    BUBBLE_ZWSP_MIN_RUN,
    BUBBLE_ZWSP_STEP,
    GEN_JOIN_TIMEOUT_S,
    MEMORY_MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    ORPHAN_EXIT_WAIT_S,
    REPETITION_PENALTY,
)
from ui.dialogs import host_confirmation_text, migrate_key_text, persist_allowed_host
from ui.text import (  # noqa: F401  —— 实现见 ui/text.py
    _append_bubble_text,
    _bubble_text,
    _bubble_text_width,
    _make_bubble_view,
    _relayout_after_mount,
    _relayout_bubble,
    _set_bubble_text,
)
from ui.widgets import (
    add_action_card,
    add_approval_card,
    add_bubble,
    add_neutral_card,
)
from ui.workers import (  # noqa: F401
    ApiChatWorker,
    ApiMemoryWorker,
    GenerationWorker,
    MemoryWorker,
    ModelLoader,
    _TokenStop,
)

# 关闭窗口时仍在运行的线程对象转存于此保活：
# QThread 在 run() 未返回时被 Python 析构会触发 Qt qFatal abort，必须在 accept() 前留引用。
# （ORPHAN_EXIT_WAIT_S 已随常量迁到 ui/constants.py,上面已导入,此处不再重复定义）
_ORPHANED_THREADS = []


def _keep_thread_alive(thread):
    """把仍在运行的 QThread 记到模块级列表，保证进程退出前它不会被析构。"""
    if thread is None:
        return
    if not any(t is thread for t in _ORPHANED_THREADS):
        _ORPHANED_THREADS.append(thread)


def _wait_for_orphaned_threads(total_timeout_s=ORPHAN_EXIT_WAIT_S):
    """退出前对保活的线程做一次限时等待（总时长有上限，不会卡住退出）。"""
    deadline = time.monotonic() + total_timeout_s
    for thread in _ORPHANED_THREADS:   # 只在主线程顺序调用，无并发增删
        remaining_ms = int(max(0.0, deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            return
        try:
            if thread.isRunning():
                thread.wait(remaining_ms)
        except Exception:
            pass


class _AgentRunnerFactory:
    """把 UI 侧组件包装成 controller 期望的工厂接口(probe_capabilities + create)。"""

    def __init__(self, client, executor, store, settings, instructions_provider):
        self._client = client
        self._executor = executor
        self._store = store
        self._settings = settings
        self._instructions_provider = instructions_provider

    def probe_capabilities(self):
        return self._client.probe_capabilities()   # CapabilityReport(分项能力)

    def create(self, event_sink):
        return AgentRunner(
            model=self._client,
            tools=self._executor,
            store=self._store,
            settings=self._settings,
            instructions=self._instructions_provider(),
            event_sink=event_sink,
        )

QSS = """
* { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; }
QMainWindow, QWidget { background: #14090c; }
#header { background: #1c0e13; border-bottom: 2px solid #e0314b; }
#title { color: #e0314b; font-size: 22px; font-weight: bold; }
#subtitle { color: #c9a24b; font-size: 12px; }
#status { color: #8a6a72; font-size: 11px; }
#modeBadge { color: #c9a24b; font-size: 11px; padding-left: 10px; }
QScrollArea { border: none; background: transparent; }
#chatContainer { background: transparent; }
QFrame#bubbleUser { background: #7d1520; border-radius: 12px; }
QFrame#bubbleBot { background: #241318; border: 1px solid #3a1e26; border-radius: 12px; }
QLabel#msgUser { color: #ffe8e8; font-size: 14px; background: transparent; border: none; }
QLabel#msgBot { color: #f0d9b5; font-size: 14px; background: transparent; border: none; }
QTextEdit { background: #1c0e13; color: #f0d9b5; border: 1px solid #3a1e26;
            border-radius: 8px; padding: 8px; font-size: 14px; }
QTextEdit:focus { border: 1px solid #e0314b; }
QPushButton#send { background: #e0314b; color: white; border-radius: 8px;
                   padding: 9px 22px; font-weight: bold; border: none; }
QPushButton#send:hover { background: #ff4d64; }
QPushButton#send:disabled { background: #4a2030; color: #8a6a72; }
QPushButton#clear { background: #241318; color: #c9a24b; border-radius: 8px;
                    padding: 9px 16px; border: 1px solid #3a1e26; }
QPushButton#clear:hover { background: #3a1e26; }
QScrollBar:vertical { background: #14090c; width: 8px; }
QScrollBar::handle:vertical { background: #3a1e26; border-radius: 4px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""




def context_builder(args=None):
    """构造统一的上下文组装器(聊天与 Agent 共用同一套人物状态与预算)。

    预算从运行参数派生,保证 `--history_max_chars` 这类开关是**单一事实来源**,
    而不是散落在各个调用点的魔数。

    `allowed_root` 只在能拿到时注入:它用于"能力说明"里告诉模型自己能在哪个目录里
    读写文件(缺了这段说明,聊天引擎会声称自己没有读文件的能力)。
    """
    from kurumi.context import ContextBudget, ContextBuilder
    from persona import DEFAULT_PERSONA

    budget = ContextBudget(
        history_max_chars=int(getattr(args, "history_max_chars", 16000) or 16000),
    )
    builder = ContextBuilder(DEFAULT_PERSONA, budget)
    root = str(getattr(args, "allowed_root", "") or "")
    if not root:
        try:
            from agent.config import DEFAULT_CONFIG_PATH, load_config
            root = str(load_config(DEFAULT_CONFIG_PATH).agent.allowed_root or "")
        except Exception:
            root = ""
    if root:
        builder.allowed_root = root
    return builder


# 注:失败回滚(移除孤立的 assistant 与未配对的 user)、历史裁剪、轮数计数
# 统一由 `ConversationService` 独占(见 kurumi.conversation),此处不留副本。


def load_api_config():
    """读取 API 配置，返回 (config, error)。

    取 Key 顺序：api_config.json 明文 → 环境变量 DEEPSEEK_API_KEY → Windows 凭据管理器。
    - config 非空  → API 模式
    - error 非空   → 配置有问题（缺 SDK / JSON 无效 / 域名未确认），需明确提示用户，而不是静默回退
    - 两者皆 None → 未配置 API，正常走本地模式
    """
    cfg_path = os.path.join(_HERE, "api_config.json")
    cfg: dict = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                cfg = loaded
        except Exception as e:
            return None, f"api_config.json 解析失败：{e}"
    key = (cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not key and _AGENT_AVAILABLE:
        # Key 已迁移到 Windows 凭据管理器的情况
        try:
            from agent.credentials import CredentialStore
            credential_id = str(cfg.get("credential_id") or "kurumi-deepseek")
            key = (CredentialStore().get(credential_id) or "").strip()
        except Exception:
            key = ""
    if not key or "在这里" in key or "你的" in key:
        return None, None   # 未配置 Key：按未配置处理，本地模式
    # 凭据外发域名门禁:普通聊天、记忆整理与 Agent 共用
    base_url = cfg.get("base_url") or "https://api.deepseek.com"
    if not str(base_url).lower().startswith("https://"):
        return None, f"API 地址必须是 https：{base_url}"
    if not _host_is_allowed(base_url):
        listed = "（它已列在 api_config.json 的 allowed_hosts 中，但那只是待确认候选）" \
            if _is_candidate_host(base_url, cfg.get("allowed_hosts")) else ""
        return None, (f"API 地址 {base_url} 尚未经你确认{listed},为保护凭据已拒绝发送。"
                      "请在界面确认卡上点击「允许」——确认结果会存入 Windows 凭据管理器;"
                      "仅填 api_config.json 的 allowed_hosts 不再放行。")
    if not _HAS_OPENAI:
        return None, "检测到已配置 Key，但未安装 openai 库。请执行：pip install openai"
    return {
        "api_key": key,
        "base_url": base_url,
        "model": cfg.get("chat_model") or cfg.get("model") or "deepseek-chat",
    }, None


from chat_params import (  # noqa: F401
    _API_THINKING_DEGRADED,
    _API_THINKING_SUPPORTED,
    _chat_create,
    _take_thinking_degradation,
    _thinking_kwargs,
    _unsupported_thinking_param,
)

# WDDM 下 `cudaMemGetInfo` 会把"驱动可以从别的程序回收/换出的显存"也算成空闲,
# 实测同一时刻比 nvidia-smi 乐观 0.7~0.8 GiB。拿不到 nvidia-smi 时按这个量扣掉。
_WDDM_FREE_HEADROOM_GIB = 0.75


def _nvidia_smi_free_gib(device_index):
    """用 nvidia-smi 读**实际**空闲显存(GiB);读不到返回 None。

    nvidia-smi 报的是所有程序共享的"还剩多少",不含可从别的进程回收的部分。
    它可能比 CUDA 报的低,但**低**才是安全方向 —— "够不够装下模型"要的就是它。
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line) / 1024.0)
        except ValueError:
            continue
    if not values:
        return None
    index = device_index if 0 <= device_index < len(values) else 0
    return values[index]


def _check_vram_gb():
    """返回 `(可用显存 GiB, 口径说明)`;无 CUDA 时显存为 None。

    torch **用到时才导入** —— 纯 API 模式不该因为没装 torch 而起不来。
    未装 torch 时按"未检测到可用 CUDA 显存"处理(与无 CUDA 的机器同一条路径)。

    两个口径取**较小值**。`cudaMemGetInfo` 在 Windows/WDDM 上把"驱动能从别的程序
    回收或换出的显存"也算成空闲,于是显卡其实很满时它仍给出乐观答案;
    只看它会出现"判定够 4 GiB → 选本地模型 → 加载时显存抢不到而卡住"。
    nvidia-smi 读不到时退回 torch 值,并预先扣掉这份 WDDM 余量。
    """
    try:
        import torch
    except ImportError:
        return None, "未安装 torch"
    try:
        if not torch.cuda.is_available():
            return None, "无可用 CUDA"
        free_bytes, _total = torch.cuda.mem_get_info()
        torch_free = free_bytes / (1024 ** 3)
        device_index = torch.cuda.current_device()
    except Exception:
        return None, "查询 CUDA 显存失败"
    smi_free = _nvidia_smi_free_gib(device_index)
    if smi_free is None:
        if os.name == "nt":
            return max(torch_free - _WDDM_FREE_HEADROOM_GIB, 0.0), "torch(已扣 WDDM 余量)"
        return torch_free, "torch"
    if smi_free <= torch_free:
        return smi_free, "nvidia-smi"
    return torch_free, "torch"


class KurumiWindow(QMainWindow):
    IDLE, GENERATING, REMEMBERING, AGENT_RUNNING = "idle", "generating", "remembering", "agent_running"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = self.tokenizer = None
        # 会话状态由 ConversationService 独占(追加/完成/回滚/裁剪/计数都在它内部,
        # 避免"Agent 路径不裁剪历史""Agent 回合不计轮数"这类各处各改一处的缺陷)
        self.conversation = ConversationService(
            int(getattr(args, "history_max_chars", 16000) or 16000))
        self.memories = []                # 长期记忆(下面建好存储后载入)
        self._memory_store = None
        self._session_id = ""
        self.bot_bubble = None
        self.turn_count = 0               # 已完成的回复轮数，用于自动记忆
        self.state = self.IDLE
        self.ready = False
        self.operation_id = 0             # 单调递增的操作编号，用于丢弃迟到回调
        self.active_operation_id = 0
        self.cancel_token = None          # 当前操作的取消令牌
        self.worker = self.mem_worker = None
        self._force_close = False
        self._closing = False             # 窗口已接受关闭：迟到回调不再触碰控件
        self._stream_text = ""            # 流式回复的本地累加缓冲（worker 只发 delta）
        self._stale_generation = False    # 上一次本地生成是否仍有线程占用 GPU
        self.agent_controller = None
        self.agent_ready = False
        self.agent_disabled_reason = ""
        self._agent_store = None
        self._agent_turn_appended_assistant = False
        self._task_status_fresh = False   # 状态栏是否刚被任务结果刷新(晚到的探测报告不覆盖它)
        # 待办的任务建议(聊天结束后才弹按钮,见 _offer_pending_task)
        self._pending_offer_text = ""
        # 上一轮走的是不是 Agent 路径:承接语("继续")要靠它才敢判成任务
        self._last_turn_was_agent = False
        # 待自动执行的回合编号(探测回来后据此判断"期间是否又发了新消息")
        self._pending_offer_turn = None
        # 能力确认期间排队的消息 —— **FIFO**(单槽会互相覆盖,会丢消息)。
        # 每项是 `(text, user_bubble_shown)`:气泡标志随消息一起排队,执行时才补。
        self._pending_agent_offers = []
        # 主人选了「用普通聊天」时,排队消息的其余部分转到这里,一条条答完
        # (普通聊天同时只能跑一个 worker,所以同样要排队 —— 但**一条都不能丢**)。
        self._plain_chat_backlog = []
        if args.local:
            self.api_cfg, self.api_error = None, None
        else:
            self.api_cfg, self.api_error = load_api_config()

        # 引擎选择:--local/--no-local 强制;默认按显存余量自动判断
        self._vram_note = ""
        if args.local:
            mode = "local"
        elif args.no_local:
            mode = "api"
        else:
            free_gb, vram_src = _check_vram_gb()
            if free_gb is None:
                self._vram_note = f"未检测到可用 CUDA 显存（{vram_src}）"
                mode = "api" if self.api_cfg else "none"
            elif free_gb >= args.min_vram_gb:
                self._vram_note = f"显存余量 {free_gb:.1f} GiB（按 {vram_src} 计，≥ {args.min_vram_gb:.1f} GiB），使用本地模型"
                mode = "local"
            elif self.api_cfg:
                self._vram_note = f"显存余量 {free_gb:.1f} GiB（按 {vram_src} 计）不足（< {args.min_vram_gb:.1f} GiB），自动切换 DeepSeek API"
                mode = "api"
            else:
                self._vram_note = f"显存余量 {free_gb:.1f} GiB（按 {vram_src} 计）不足且未配置 API"
                mode = "none"

        self._build_ui()
        # 存储必须**早于**任何用到 self.memories 的地方:建库 → 一次性导入旧 JSON →
        # 恢复上次会话。失败只提示,不影响聊天。
        store_note = self._setup_memory_store()
        self._restore_session()
        if store_note:
            self.status.setText("⚠️ " + store_note)
        if self.api_error:
            # 无论最终走哪种引擎,只要 API 侧有明确问题就先摆到主人眼前,
            # 并初始化 Agent——「域名未确认」时确认卡正是从这条路径进入界面。
            self.status.setText("⚠️ " + self.api_error)
            self.add_bubble("⚠️ " + self.api_error, is_user=False)
            self._setup_agent()
        if mode == "api":
            if self.api_cfg:
                self.ready = True
                self._set_state(self.IDLE)
                self.status.setText(
                    f"已连接 DeepSeek API（{self.api_cfg['model']}）· 已记得 {len(self.memories)} 条关于主人的回忆"
                )
                self._setup_agent()
            elif self.api_error:
                # 配置了 API 但有明确问题：原因已在上方提示；自动模式下回退本地，
                # --no-local 时不再加载本地。若确认域名后 API 可用，
                # _refresh_api_config_after_host_confirmation 会补上。
                if not args.no_local:
                    self._start_loading()
            else:
                self.status.setText("⚠️ 本地模型已禁用（--no-local），且未配置 API，无法启动引擎")
                self.add_bubble("本地模型已禁用且未找到 API 配置。\n请在 api_config.json 中填入 DeepSeek Key 后重启。", is_user=False)
        elif mode == "local":
            self._start_loading()
        else:   # none:显存不足且无 API
            if self.api_error:
                # 真正的原因已在上方给出，这里不再重复「未配置 API」的误判提示
                self.add_bubble("显存不足，且 API 因上述原因暂不可用，当前无法启动引擎。", is_user=False)
            else:
                self.status.setText("⚠️ " + self._vram_note + "，无法启动引擎")
                self.add_bubble(
                    "显存不足且未配置 DeepSeek API。\n"
                    "可任选其一:\n"
                    "1. 在 api_config.json 填入 DeepSeek Key 后重启(走 API);\n"
                    "2. 关闭占用显存的程序后重启;\n"
                    "3. 使用 --local 强制尝试本地模型。", is_user=False)

    def _next_operation_id(self):
        self.operation_id += 1
        return self.operation_id

    def _refresh_mode_badge(self):
        """刷新常驻徽标:一眼看出"下一条消息会走哪条路"。

        分流是**按条**判定的(见 `on_send` 里的意图识别),所以徽标描述的是决定走向的
        那个东西 —— Agent 当前处于哪种状态。状态栏那句话会被下一条消息覆盖,徽标不会。
        """
        badge = getattr(self, "mode_badge", None)
        if badge is None:
            return
        controller = getattr(self, "agent_controller", None)
        if controller is None:
            text = "💬 普通聊天"
        elif controller.capable:
            text = "🛠 任务自动执行"
        elif not getattr(self, "_agent_probed", False):
            text = "🛠 任务先征求同意"
        elif getattr(self, "agent_ready", False):
            text = "⚠️ 任务需重新确认能力"
        else:
            text = "💬 普通聊天 · Agent 已关闭"
        if badge.text() != text:
            badge.setText(text)

    def _set_state(self, state):
        self.state = state
        idle = state == self.IDLE
        # 遗留本地生成仍占用显存时必须继续禁止发送，否则两次 generate 抢同一份权重
        can_send = idle and self.ready and not getattr(self, "_stale_generation", False)
        self.send_btn.setEnabled(can_send)
        self.remember_btn.setEnabled(can_send)
        self.clear_btn.setEnabled(idle)
        if hasattr(self, "stop_btn"):
            # 生成/记忆整理/Agent 运行期间都提供「停止」入口
            self.stop_btn.setVisible(state in (self.GENERATING, self.REMEMBERING, self.AGENT_RUNNING))
        # 路径由**每条消息的意图**决定(见 on_send),按钮统一叫"发送";
        # "这一条会走哪条路"由常驻徽标交代,不再靠按钮文案暗示。
        if hasattr(self, "send_btn"):
            self.send_btn.setText("发送")
        self._refresh_mode_badge()

    def _build_ui(self):
        self.setWindowTitle("时崎狂三 · 梦魇")
        self.resize(520, 720)
        self.setStyleSheet(QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 头部
        header = QWidget()
        header.setObjectName("header")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(20, 14, 20, 12)
        title = QLabel("🕰️ 时崎狂三 · 梦魇")
        title.setObjectName("title")
        sub = QLabel("主人，贵安。我一直在等你开口。")
        sub.setObjectName("subtitle")
        hl.addWidget(title)
        hl.addWidget(sub)
        root.addWidget(header)

        # 聊天区
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_container.setObjectName("chatContainer")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()
        self.scroll.setWidget(self.chat_container)
        root.addWidget(self.scroll, 1)

        # 状态栏（放在对话框下方，不打扰对话观感）
        # 左半边是**临时**状态(会被下一条消息覆盖);右半边是**常驻**模式徽标 ——
        # 路由是按条判定的,所以徽标显示的是"决定走向的那个东西":Agent 的当前状态。
        status_row = QHBoxLayout()
        status_row.setContentsMargins(16, 4, 16, 4)
        status_row.setSpacing(0)
        self.status = QLabel("正在加载模型……")
        self.status.setObjectName("status")
        self.mode_badge = QLabel("")
        self.mode_badge.setObjectName("modeBadge")
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.mode_badge, 0)
        root.addLayout(status_row)

        # 输入区
        input_area = QWidget()
        il = QVBoxLayout(input_area)
        il.setContentsMargins(16, 8, 16, 14)
        self.input = QTextEdit()
        self.input.setPlaceholderText("对狂三说点什么吧……（Ctrl+Enter 发送）")
        self.input.setFixedHeight(64)
        il.addWidget(self.input)
        btns = QHBoxLayout()
        self.clear_btn = QPushButton("清空")
        self.clear_btn.setObjectName("clear")
        self.remember_btn = QPushButton("记住这段对话")
        self.remember_btn.setObjectName("clear")
        self.remember_btn.setEnabled(False)
        # 界面没有「Agent 联网」开关:Agent 功能已并入模型功能,无需主人手动切换 ——
        # 消息默认走 Agent 路径(能力未确认时先征求同意,不可用时退回普通聊天)。
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("clear")
        self.stop_btn.setVisible(False)
        self.reload_btn = QPushButton("重新加载模型")
        self.reload_btn.setObjectName("clear")
        self.reload_btn.setVisible(False)     # 仅在本地模型加载失败后出现
        self.send_btn = QPushButton("发送")
        self.send_btn.setObjectName("send")
        self.send_btn.setEnabled(False)
        btns.addWidget(self.clear_btn)
        btns.addWidget(self.remember_btn)
        btns.addWidget(self.reload_btn)
        btns.addStretch()
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.send_btn)
        il.addLayout(btns)
        root.addWidget(input_area)

        self.send_btn.clicked.connect(self.on_send)
        self.clear_btn.clicked.connect(self.on_clear)
        self.remember_btn.clicked.connect(self.on_remember)
        self.stop_btn.clicked.connect(self._on_stop_agent)
        self.reload_btn.clicked.connect(self._start_loading)
        # Ctrl+Enter/Ctrl+Return 发送：窗口级 keyPressEvent 会被 QTextEdit 吞掉，
        # 改用 QShortcut（父对象为窗口，C++ 侧持有，不会被 GC）
        self._send_shortcuts = []
        for seq in ("Ctrl+Return", "Ctrl+Enter"):
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(self.on_send)
            self._send_shortcuts.append(shortcut)

    def _start_loading(self):
        loader = getattr(self, "loader", None)
        if loader is not None and loader.isRunning():
            return
        if hasattr(self, "reload_btn"):
            self.reload_btn.setVisible(False)
        self.status.setText("正在加载本地模型……")
        self.loader = ModelLoader(self.args.base, self.args.adapter, not self.args.no_quantize)
        self.loader.loaded.connect(self.on_loaded)
        self.loader.failed.connect(self.on_load_failed)
        self.loader.start()

    def on_loaded(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.ready = True
        if hasattr(self, "reload_btn"):
            self.reload_btn.setVisible(False)
        self._set_state(self.IDLE)
        self.status.setText(f"就绪 · 已记得 {len(self.memories)} 条关于主人的回忆")

    def on_load_failed(self, msg):
        """本地模型加载失败：给出实际 --base 取值并提供界面内重试入口。

        API 可用时**直接改用 API** —— 主人要的是"本地装不下就走 API",而不是一个
        发送键始终灰着的界面(`ready=False` 会让普通聊天与 Agent 都发不出去)。
        `--local` 是主人显式要求本地,那时不擅自改判,只给重试入口。
        """
        detail = f"模型加载失败（--base={self.args.base}）：{msg}"
        if self.api_cfg and not getattr(self.args, "local", False):
            self.ready = True
            if getattr(self, "agent_controller", None) is None:
                self._setup_agent()
            self.status.setText(
                f"⚠️ 本地模型加载失败,已自动改用 DeepSeek API（{self.api_cfg['model']}）")
            self.add_bubble("⚠️ " + detail + "\n已自动改用 DeepSeek API,可以直接对话。"
                            "\n腾出显存后可点「重新加载模型」再试本地。", is_user=False)
        else:
            self.status.setText("⚠️ " + detail)
            self.add_bubble("⚠️ " + detail + "\n可点击「重新加载模型」重试，"
                            "或用 --base/--adapter 指定正确路径后重启。", is_user=False)
            self.ready = False
        if hasattr(self, "reload_btn"):
            # 无论是否已改用 API 都留着它:腾出显存后主人可以再试本地
            self.reload_btn.setVisible(True)
            self.reload_btn.setEnabled(True)
        self._set_state(self.IDLE)

    def _bubble_inner_width(self):
        """气泡内文本可用宽度（窗口宽度比例 - 气泡内边距）。"""
        return int(self.width() * BUBBLE_WIDTH_RATIO) - BUBBLE_H_PADDING

    def add_bubble(self, text, is_user):
        # 控件构建细节在 ui/widgets.py,窗口只做委派
        return add_bubble(self, text, is_user)

    def _scroll_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _warn(self, message):
        """状态栏 + 气泡双通道提示：避免用户输入被静默丢弃。"""
        self.status.setText("⚠️ " + message)
        self.add_bubble("⚠️ " + message, is_user=False)

    def _is_closing(self):
        """窗口是否已接受关闭。

        用 getattr 读取：实例可能是 KurumiWindow.__new__ 绕过 __init__ 构造出来的，
        那时实例属性并不存在。
        """
        return getattr(self, "_closing", False)

    def on_send(self):
        if self.state != self.IDLE:
            return
        text = self.input.toPlainText().strip()
        if not text or not self.ready:
            return
        if getattr(self, "_stale_generation", False):
            # 上一次本地生成仍有线程在跑：此时再 generate 会与它抢同一份权重
            self._warn("上一条本地生成尚未完全停止（仍可能占用显存），请稍候再发送。")
            return
        # 记忆命令(查看/纠正/遗忘/置顶)在本地完成,不发模型请求。
        # 这里只用**无副作用**的 parse_command 判断,真正的执行在 _run_memory_command 里
        # 做一次 —— 若判断时也调 handle_text,命令会被执行两遍(第二次找不到目标,
        # 于是改动被当成"没改动",既不写盘也不提示)。
        # 认不出的 "/xxx" 会返回 None,按普通消息照常发送,绝不吞掉主人的输入。
        if memory_commands.parse_command(text) is not None:
            self._run_memory_command(text)
            return
        # 先做一次**本地**意图识别(不发任何请求,规则见 task_intent):只有像任务的
        # 消息才考虑 Agent 路径。一句"你好"不该让主人面对"要不要花 1~2 次真实请求去
        # 探测 Agent 能力"的付费选择 —— 那是把闲聊也当成任务来问。
        # 漏判的代价只是"这条按普通聊天回答了",而且本轮结束后仍会按识别结果补一个
        # 「执行这个任务」按钮(见 _offer_pending_task)。
        intent = task_intent.looks_like_task(text)
        if intent is None and getattr(self, "_last_turn_was_agent", False) \
                and task_intent.is_continuation(text):
            # 承接上一轮任务的短句("继续""然后呢")本身没有动作动词,但上一轮在跑任务
            intent = task_intent.TaskIntent("generic", "承接上一轮的任务")
        if self.agent_controller is not None and intent is not None:
            if self.agent_controller.capable:
                self._start_agent_run(text)
                return
            if not getattr(self, "_agent_probed", False):
                self._pending_agent_offers.append((text, False))
                self._confirm_agent_probe(text)
                return
            if self.agent_ready:
                # 上次是瞬时故障:能力其实可用,直接交给 Agent(它会自己重试探测)
                self._pending_agent_offers.append((text, False))
                self.status.setText("正在确认 Agent 能力,确认后自动执行……")
                self._start_agent_probe()
                return
            # 像任务,但 Agent 已被关掉(主人拒绝探测 / 探测未通过):在状态栏如实说明,
            # 别让主人以为"它没听懂"。
            reason = getattr(self, "agent_disabled_reason", "") or "Agent 当前不可用"
            self.status.setText(f"⚠️ 这条像是在派活,但 Agent 不可用:{reason}（这次按普通聊天回复）")
        # 闲聊,或 Agent 确实不可用(端点不支持/凭据缺失/主人拒绝探测):
        # 退回普通聊天,而不是把主人的话丢掉。
        self._start_plain_chat(text)

    def _start_plain_chat(self, text, clear_input=True):
        """普通聊天路径(与 Agent 无关的那条)。抽出来是为了让"退回普通聊天"可复用。

        `clear_input=False` 用于**自动排空队列**的场景 —— 那时输入框里可能是主人
        正在打的下一条消息,清空它等于替主人删草稿。
        """
        op_id = self._next_operation_id()
        self.active_operation_id = op_id
        if clear_input:
            self.input.clear()
        self._last_turn_was_agent = False
        self.status.setText("💬 普通聊天回复中……")
        self.conversation.begin_turn(text, ENGINE_API if self.api_cfg else ENGINE_LOCAL)
        self._begin_turn_persistence(text, ENGINE_API if self.api_cfg else ENGINE_LOCAL)
        self.add_bubble(text, is_user=True)
        self.bot_bubble = self.add_bubble("……", is_user=False)
        self._stream_text = ""
        self._set_state(self.GENERATING)

        # 与 Agent 共用同一套人物状态与预算(同一份人设、同一批记忆、同一个知识库预算)
        history = self.conversation.snapshot()
        chat_context = context_builder(self.args).build_chat(history, self.memories, text)
        messages = ([{"role": "system", "content": chat_context.system}]
                    + copy.deepcopy(history))
        token = CancellationToken()
        self.cancel_token = token
        if self.api_cfg:
            self.worker = ApiChatWorker(self.api_cfg, messages, self.args, cancel_token=token)
        else:
            self.worker = GenerationWorker(self.model, self.tokenizer, messages, self.args, cancel_token=token)
        self.worker.chunk.connect(lambda delta, _id=op_id: self.on_chunk(delta, _id))
        self.worker.done.connect(lambda final, _id=op_id: self.on_done(final, _id))
        self.worker.failed.connect(lambda msg, _id=op_id: self.on_failed(msg, _id))
        if hasattr(self.worker, "stale"):
            self.worker.stale.connect(lambda msg, _id=op_id: self.on_generation_stale(msg, _id))
            self.worker.stale_cleared.connect(self.on_generation_stale_cleared)
        if hasattr(self.worker, "degraded"):
            self.worker.degraded.connect(self.on_thinking_degraded)
        self.worker.start()
        # 这条消息像"要求做事"吗?像就在**本轮聊天结束之后**给一个可点的执行按钮。
        # 刻意不在这里立刻给:此刻普通聊天请求还在飞,主人一点就会与它并发
        # (active_operation_id 被覆盖、两次 begin_turn,两个请求共用同一套会话状态)。
        self._pending_offer_text = text

    def _offer_pending_task(self, text, op_id=None):
        """本轮聊天已结束(成功/失败/取消)后,才把任务按钮放出来。

        按钮只可能在**没有在途请求**时出现,因此不存在"点一下变成两个并发请求"。
        op_id 用于丢弃迟到的旧回合回调(新消息已经发出去时,旧回合的结果不该弹卡)。
        """
        if self.state != self.IDLE:
            return
        if op_id is not None and op_id != self.active_operation_id:
            return
        self._pending_offer_text = ""
        self._offer_agent_run(text or "")

    def _offer_agent_run(self, text):
        """聊天里识别到任务意图时,在气泡旁放一个「执行这个任务」按钮。

        刻意的设计选择:

        - **不自动执行**:要不要动主人的文件/系统,应由主人点一下决定,而不是本地规则
          或模型替他决定(误判的代价只是多一个按钮);
        - **识别在本地**(见 `task_intent`):不发任何请求,也不让模型判断"这算不算任务";
        - 主人不点时,这次交互就是一次普通聊天 —— 与升级前完全一致。

        「主人点一下」就是授权本身:`_accept_task_offer` 会在能力探测通过后自动接续执行
        (探测未通过时如实说明原因),不需要主人点第二次。

        Agent 不可用时按钮照旧给出,但点击后会走 `_start_agent_run` 的既有校验
        (能力探测未通过会明确告知原因),而不是静默失败。
        """
        if not getattr(self, "agent_ready", False) or self.agent_controller is None:
            return
        if self.state != self.IDLE:
            # 生成/任务进行中不弹卡:此时点下去会和在途请求撞车
            return
        intent = task_intent.looks_like_task(text)
        if intent is None:
            return
        self._add_action_card(
            "🛠 " + task_intent.suggestion_text(intent),
            [("执行这个任务", lambda card: self._accept_task_offer(card, text))],
        )

    def _accept_task_offer(self, card, text):
        """主人点了「执行这个任务」:转入 Agent 执行(用户气泡已在聊天区,不重复添加)。

        能力未确认时不直接发探测请求,而是先征求同意(`_confirm_agent_probe`)。
        """
        card.setEnabled(False)
        # 兜底闸门:即使卡片的启用时机出了问题(或一张早已禁用的卡片被点),也绝不在生成/任务
        # 进行中启动第二次运行 —— 否则会有两个模型请求共用会话与 active_operation_id。
        if self.state != self.IDLE:
            self._warn("现在还在处理上一条消息:等它结束再执行这个任务,或先点「停止」。")
            return
        if not self.agent_ready or self.agent_controller is None:
            self._warn("Agent 当前不可用:" + (self.agent_disabled_reason or "未知原因"))
            return
        if self.agent_controller.capable:
            self._start_agent_run(text, user_bubble_shown=True)
            return
        # 还没确认能力:排队并征求同意(这条消息的**用户气泡已在聊天区**,故记 True)
        self._pending_agent_offers.append((text, True))
        self._confirm_agent_probe(text)

    def _flush_pending_offer(self):
        """**唯一启动闸门**:把排队中的消息依次交出去(能执行就执行,不能就如实说明)。

        待办要真正落地必须同时满足"当前空闲"与"回合没变过",否则一律放弃并告知 ——
        宁可让主人重发一次,也不能在生成中插进第二个任务。

        队列用 **FIFO** 而不是单槽:`(text, user_bubble_shown)` 一起入队,
        所以"排队时没显示过用户气泡"这件事会被记住,执行时补上 —— 固定传 True 的话,
        排队那条消息在聊天区里只会剩机器人气泡。

        返回"是否真的启动了任务":调用方(任务结束时的收尾)要靠它决定
        下一步该排空队列还是整理记忆 —— 顺序反了就会把排队消息当并发丢掉。
        """
        offers = getattr(self, "_pending_agent_offers", [])
        if not offers:
            return False
        if self.state != self.IDLE:
            # 期间主人又发了消息/已有任务在跑:不再自动执行,免得两个任务并发
            self._pending_agent_offers = []
            self._warn("刚才排队的任务没能自动执行:期间又有新的消息在处理。"
                       "需要的话,请再发一次。")
            return False
        turn = getattr(self, "_pending_offer_turn", None)
        self._pending_offer_turn = None
        if turn is not None and turn != getattr(self, "active_operation_id", None):
            self._pending_agent_offers = []
            self._warn("刚才排队的任务没能自动执行:回合已经变了(期间又发了新消息)。"
                       "需要的话,请再发一次。")
            return False
        if self.agent_controller is not None and self.agent_controller.capable:
            # **一次只放行队列头部一条**:连发多个 `_start_agent_run` 会互相覆盖
            # active_operation_id / 状态(两个请求共用的会话状态)。剩下的等本次任务结束后
            # 由 `_on_agent_finished` 再调本方法继续排空 —— 于是 FIFO 是**串行**的。
            text, bubble_shown = offers.pop(0)
            # 自动排空**不碰输入框**(那里可能是主人正在打的下一条消息)
            self._start_agent_run(text, user_bubble_shown=bubble_shown, clear_input=False)
            return True
        self._pending_agent_offers = []
        reason = self.agent_disabled_reason or "能力探测未通过"
        self._warn("刚才排队的任务没能执行:" + reason
                   + "\n（稍后再发一次即可重试;也可直接普通聊天）")
        return False

    # ==================== 存储与会话恢复 ====================
    def _setup_memory_store(self):
        """建库 → 一次性导入旧 memory.json → 挂为权威来源;失败则如实上报并退回 JSON。

        返回错误说明(成功时返回空串)。**绝不让存储问题使聊天不可用**:
        这里捕获所有异常,并把记忆退回 JSON 路径。
        """
        try:
            # 独立文件:不与 Agent 审计库(agent_data/agent.db)混用 —— 两套 schema
            # 混在一个文件里既难排查,也让"删掉记忆库重建"牵连审计记录。
            store = MemoryStore(os.path.join(_HERE, "agent_data", "memory.db"))

        except Exception as e:
            self.memories = load_memories()
            return f"记忆数据库不可用,已退回 JSON 存储:{type(e).__name__}: {e}"
        self._memory_store = store
        try:
            imported = store.import_from_json()
        except Exception as e:
            imported = 0
            note = f"旧记忆导入失败:{type(e).__name__}: {e}"
        else:
            note = f"已从 memory.json 导入 {imported} 条记忆" if imported else ""
            status = store.get_meta("imported_status")
            if status in ("corrupt", "read_error"):
                note = (f"memory.json 存在读取问题({status}),未导入;"
                        "原文件已留档,修好后重启即可导入")
        attach_store(store)
        self.memories = load_memories_store_first()
        return note

    def _restore_session(self):
        """恢复上次会话的轮次(用户验收:重启后能恢复会话和明确约定)。"""
        store = self._memory_store
        if store is None:
            self._session_id = uuid.uuid4().hex
            return
        try:
            last = store.latest_session()
            self._session_id = last or uuid.uuid4().hex
            if last:
                # 数据迁移:整理修复库里已写下的坏轮次。
                # 只删"内容完全相同"的重复提问;末尾那条没得到回答的提问**保留**并标注
                # 未完成(它没有回答,带入上下文只会得到"两条 user 挨着"的坏序列)。
                # 备份没成功就什么都不做(见 MemoryStore.repair_turn_sequence)。
                try:
                    repair = store.repair_turn_sequence(self._session_id)
                    if repair.get("aborted"):
                        print("[!] 会话记录整理已跳过:备份没成功,旧数据原样保留")
                        self.status.setText("⚠️ 会话记录整理已跳过:备份没成功,旧数据原样保留")
                    elif repair.get("duplicates") or repair.get("incomplete"):
                        print(f"[!] 会话记录已整理:去掉 {repair['duplicates']} 条重复提问、"
                              f"标注 {repair['incomplete']} 条未得到回答的提问"
                              + (f"(备份: {repair['backup']})" if repair.get("backup") else ""))
                        detail = []
                        if repair.get("duplicates"):
                            detail.append(f"去掉 {repair['duplicates']} 条重复提问")
                        if repair.get("incomplete"):
                            detail.append("上次有 1 条消息没来得及回答 —— 已保留在会话记录里、"
                                          "不再带入上下文;需要的话请再发一次")
                        self.status.setText("已整理会话记录:" + "、".join(detail))
                except Exception as e:
                    print(f"[!] 会话记录整理失败(不影响本次对话): {e}")
            turns = store.load_turns(self._session_id, limit=40) if last else []
            if turns:
                self.conversation.replace_history(
                    [{"role": t["role"], "content": t["content"]} for t in turns])
            store.start_session(self._session_id, engine="api" if self.api_cfg else "local")
        except Exception as e:
            print(f"[!] 会话恢复失败,本次以空会话开始: {e}")
            self._session_id = uuid.uuid4().hex

    def _restore_agent_result(self):
        """重启后把"上次 Agent 真正做成的事"注入历史。

        为什么需要:Agent 的审计(含产物路径)落在 agent.db,而会话轮次落在 memory.db ——
        进程重启后两者不会自动对齐,于是主人再问"那个文件生成了吗"时,
        模型与程序都不知道上次写过什么(运行内也有同类问题,这里只是跨了进程边界)。

        只恢复**有落盘副作用**的结果;是否重复靠"这条记录是否已在历史里"判断 ——
        不需要额外的"只跑一次"标志位:那个标志位与内容检查语义完全重叠,
        多一个标志位只是多一处可能失配的状态。

        恢复范围**按会话**划定,不按全局时间戳设"清空边界" —— 时间戳有两个洞:
        ①边界写不进去(AgentStore 还没初始化/写失败)时清空照样宣称成功,旧结果以后照样注入;
        ②同一秒内的新旧运行、或系统时钟回拨,`started_at` 分不出先后,还会把清空后的新结果拦掉。
        所以只恢复**属于当前会话**的结果 —— 靠 `session_id` 判定,与时钟无关。
        """
        store = getattr(self, "_agent_store", None)
        if store is None:
            return
        try:
            result, _started_at = store.latest_task_result_entry(
                with_side_effects=True, session_id=getattr(self, "_session_id", ""))
        except Exception as e:
            print(f"[!] 上次 Agent 结果恢复失败(不影响本次对话): {e}")
            return
        if result is None:
            return
        note = agent_history_note(result)
        if not note:
            return
        existing = "\n".join(str(m.get("content", "")) for m in self.conversation.snapshot())
        if note in existing:
            return
        self.conversation.append_note(
            note, ENGINE_AGENT,
            context_text="（系统记录：上次运行 Agent 的结果，来自程序记录）")
        self._record_turn("assistant", note, ENGINE_AGENT)
        self.status.setText("已恢复上次 Agent 的结果：" + (
            "、".join(result.artifacts) if result.artifacts else agent_state_label(result)))

    def _agent_result_before_clear_boundary(self, started_at) -> bool:
        """已废弃:清空边界由 `session_id` 判定,不用时间戳。

        保留这个名字是为了让老调用方立刻炸出来,而不是悄悄按"没有边界"处理 ——
        时间戳在"同一秒 / 时钟回拨"时分不出先后,不可靠。
        """
        raise RuntimeError("清空边界已改为按会话判定(见 _restore_agent_result),请勿再用时间戳")

    def _record_turn(self, role, content, engine=""):
        """把一轮写进会话存储(失败只提示,不影响对话本身);返回落盘用的 `seq`。"""
        # getattr 兜底:实例可能是 __new__ 造出来的窗口,这里不该因为少一个属性就炸
        store = getattr(self, "_memory_store", None)
        if store is None or not getattr(self, "_session_id", ""):
            return None
        try:
            return store.append_turn(self._session_id, role, content, engine)
        except Exception as e:
            print(f"[!] 会话轮次未落盘(不影响本次对话): {e}")
            return None

    def _begin_turn_persistence(self, text, engine):
        """本轮开始:把主人的话先落盘,并记住它的 `seq`。

        为什么发送时就写:生成要几秒到几十秒,期间进程被杀也不该把主人的问题弄丢。
        但**必须**记住 seq:否则失败/取消时库里会留下一条孤立 user,重启恢复会话时
        被重新载入内存历史,和"内存已回滚"的事实互相打架(成功聊天会留下
        `user/user/assistant`,失败聊天留下孤立 `user`)。
        """
        self._session_turn_seq = self._record_turn("user", text, engine)

    def _commit_turn_persistence(self):
        """本轮正常结束:库里保留已写入的 user/assistant,不再需要回滚。"""
        self._session_turn_seq = None

    def _rollback_turn_persistence(self):
        """本轮失败/取消:把这次写进库里的行全部撤掉(与内存回滚保持一致)。"""
        seq = getattr(self, "_session_turn_seq", None)
        self._session_turn_seq = None
        store = getattr(self, "_memory_store", None)
        if seq is None or store is None or not getattr(self, "_session_id", ""):
            return
        try:
            store.drop_turns_from(self._session_id, seq)
        except Exception as e:
            print(f"[!] 会话轮次回滚失败(重启后可能看到一条未完成的提问): {e}")

    def _save_memories(self):
        """统一的记忆写盘入口:走存储优先路径(含防误清空守卫与 JSON 镜像)。"""
        return save_memories_store_first(self.memories)

    def _run_memory_command(self, text):
        """执行记忆命令:本地改记忆 + 写盘 + 上屏,全程不调用模型。"""
        result = memory_commands.handle_text(text, self.memories)
        if result is None:
            return False
        reply, changed = result
        self.input.clear()
        self.add_bubble(text, is_user=True)
        self.add_bubble(reply, is_user=False)
        if changed:
            if self._save_memories():
                self.status.setText("记忆已更新并写盘")
            else:
                # 写盘被"疑似误清空"保护拦下:如实说明,并回滚内存,不能谎报成功
                self.memories = load_memories_store_first()
                self.status.setText("记忆未写入磁盘(被保护拦下),已重新载入磁盘上的内容")
        else:
            self.status.setText("记忆命令已执行(未改动记忆)")
        return True

    def on_chunk(self, delta, op_id=None):
        """流式增量：worker 只发 delta，这里累加并节流重算气泡宽高。"""
        if self._is_closing():
            return
        if op_id is not None and op_id != self.active_operation_id:
            return
        if self.bot_bubble:
            self._stream_text += delta
            if _append_bubble_text(self.bot_bubble, self._stream_text):
                QTimer.singleShot(0, self._scroll_bottom)

    def on_done(self, final, op_id=None):
        if self._is_closing():
            return
        if op_id is not None and op_id != self.active_operation_id:
            return
        self.conversation.complete_turn(final, ENGINE_API if self.api_cfg else ENGINE_LOCAL)
        # 落盘:重启后能恢复会话(用户验收)。失败只打印,不影响本次回复。
        # 主人的那句话**发送时就写过了**(崩溃也留着),这里只补 assistant 那一行 ——
        # 在这里再写一遍 user,库里就会变成 `user, user, assistant`。
        self._record_turn("assistant", final, ENGINE_API if self.api_cfg else ENGINE_LOCAL)
        self._commit_turn_persistence()
        if self.bot_bubble:
            # 收尾：补上节流期间可能遗漏的宽高重算
            _set_bubble_text(self.bot_bubble, final)
        self.bot_bubble = None
        self._stream_text = ""
        self.turn_count = self.conversation.turn_count
        self.cancel_token = None
        self._set_state(self.IDLE)
        # 收尾必须**改写**状态栏:开始那一轮时写的是"回复中……",不覆盖的话界面会
        # 永远停在"正在回复",主人无从判断是否结束(Agent 路径同样有收尾语)。
        self.status.setText("💬 普通聊天回复完成")
        # 先接着答"退回普通聊天"的积压消息(有的话,状态会重新变回 GENERATING)
        if self._drain_plain_backlog():
            return
        # 本轮聊天结束,现在才是弹出任务按钮的安全时机(见 _offer_pending_task)
        self._offer_pending_task(getattr(self, "_pending_offer_text", ""), op_id)
        if self.conversation.should_remember(self.args.remember_every):
            self.on_remember()

    def on_failed(self, msg, op_id=None):
        if self._is_closing():
            return
        if op_id is not None and op_id != self.active_operation_id:
            return
        # 失败不写入历史、不增加轮数、不触发记忆整理；移除未配对的 user 消息
        self.conversation.fail_turn()
        # 内存回滚了,库里也必须一起回滚(否则重启后那条孤立 user 会被重新载入)
        self._rollback_turn_persistence()
        cancelled = (msg == "已取消")
        if self.bot_bubble:
            _set_bubble_text(self.bot_bubble, "（已取消）" if cancelled else f"（出错了：{msg}）")
        self.bot_bubble = None
        self._stream_text = ""
        self.cancel_token = None
        self.status.setText("已取消本次生成" if cancelled else "生成失败，可重试：" + msg)
        self._set_state(self.IDLE)
        # 失败/取消也要接着答积压消息(主人的消息不能因为一次失败就没了)
        if self._drain_plain_backlog():
            return
        # 取消也要给按钮:主人喊停之后往往正是想改用"真的去做"这条路
        self._offer_pending_task(getattr(self, "_pending_offer_text", ""), op_id)

    def on_generation_stale(self, msg, op_id=None):
        """超时/取消后底层 generate 仍未退出：回 IDLE 但继续禁止发起新生成。"""
        if self._is_closing():
            return
        if op_id is not None and op_id != self.active_operation_id:
            return
        self._stale_generation = True
        self.conversation.fail_turn()
        # 与 on_failed 同理 —— 未完成的回合不能在库里留一条孤立 user
        self._rollback_turn_persistence()
        if self.bot_bubble:
            _set_bubble_text(self.bot_bubble, f"（出错了：{msg}）")
        self.bot_bubble = None
        self._stream_text = ""
        self.status.setText(f"⚠️ {msg}；上一次本地生成尚未完全停止，暂时不能再次发送。")
        self._set_state(self.IDLE)

    def on_generation_stale_cleared(self):
        """遗留生成线程已真正结束，解除封锁。"""
        if self._is_closing():
            return
        self._stale_generation = False
        self._set_state(self.state)
        self.status.setText("已就绪，可以继续对话了")
        # 之前因为"上一次生成还没退干净"而压着没答的消息,现在可以答了
        self._drain_plain_backlog()

    def on_thinking_degraded(self, msg):
        """端点不接受 thinking 参数:必须让用户知道 --api_thinking 实际没生效。

        这条提示不能在状态栏"常驻覆盖"生成进度,所以只在空闲时写状态栏,
        生成中/整理中改为打印到控制台,避免把"正在生成…"这类状态冲掉。
        """
        if self._is_closing():
            return
        print(f"[!] {msg}")
        if self.state == self.IDLE:
            self.status.setText(f"⚠️ {msg}")

    def on_clear(self):
        if self.state != self.IDLE:
            return
        self.conversation.clear()
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w:
                w.deleteLater()
        # 底部的 stretch 在 _build_ui 中只创建一次，这里复用，不再新增

        # 清空会连同聊天区里的**确认卡**一起删掉 —— 那些"还活着的入口"必须同步收拾,
        # 否则状态里仍写着"卡片可点",而屏幕上一张卡都没有:之后的消息进队列后
        # `_confirm_agent_probe` 只会提示去点那张不存在的卡,永久等待。
        pending = len(getattr(self, "_pending_agent_offers", []))
        backlog = len(getattr(self, "_plain_chat_backlog", []))
        self._pending_agent_offers = []
        self._plain_chat_backlog = []
        self._pending_offer_turn = None
        self._agent_probe_asked = False
        self._probe_card_live = False

        # 清空必须与**持久化**对齐。只清内存和控件的话,重启时 `latest_session()`
        # 会把刚清掉的内容整段载入。这里采用**非破坏**的做法:换一个新会话,
        # 屏幕上清干净、重启也不会再载入;上一段对话仍然留在会话记录里(没有不可逆删除),
        # 并把这件事如实告诉主人。
        rotated = self._rotate_session()
        if not rotated:
            # 新会话没登记成功就**不能**说"重启不会再载入" —— 那时 latest_session
            # 仍指向旧会话,重启照样把这段对话载入回来。
            self._warn("已清空聊天区,但新会话没能登记(会话存储写入失败):"
                       "重启后可能仍会载入这一段对话,请检查磁盘/权限。"
                       "为保证记录一致,后续消息仍写在原来的会话里。")
            return
        note = "聊天区已清空,并开始新会话;上一段对话仍保存在会话记录里(重启不会再载入)。"
        if pending or backlog:
            self._warn(f"聊天区已清空,并开始新会话;清空前排队等待的 {pending + backlog} 条消息"
                       "已取消。上一段对话仍保存在会话记录里(重启不会再载入)。")
        else:
            self.status.setText(note)

    def _rotate_session(self) -> bool:
        """换一个新的会话 id(清空 / 新开一段对话),返回**持久化是否已与清空一致**。

        为什么不能只清内存:会话库是"重启后恢复会话"的权威来源,`latest_session()` 会挑
        `updated_at` 最新的那条 —— 不换 id 的话,刚清掉的内容下次启动照样回来。

        登记失败时**保持原样**(沿用旧会话 id)并返回 False —— 否则后续轮次会写进
        一个 `sessions` 里根本不存在的 id,变成永远载入不到的孤立数据;而界面那边
        也会因为"看起来成功了"而对主人说假话。
        """
        store = getattr(self, "_memory_store", None)
        if store is None:
            # 没有会话存储(降级模式):内存清空就是清空,重启本来也不会恢复
            self._session_id = uuid.uuid4().hex
            self._session_turn_seq = None
            return True
        old_session = getattr(self, "_session_id", "")
        new_session = uuid.uuid4().hex
        # 清空**不需要**再往 agent.db 写"恢复边界" —— 边界就是会话本身:
        # 运行记录带着 session_id,恢复时只找当前会话的结果。于是"边界写不进去"
        # 这个失败模式连同它对成功文案的影响一起消失了。
        try:
            store.start_session(new_session, engine="api" if self.api_cfg else "local")
        except Exception as e:
            print(f"[!] 新会话未能登记(本次清空只清了屏幕): {e}")
            self._session_id = old_session        # 回到仍然有效的那个会话,不写孤立数据
            self._session_turn_seq = None
            return False
        self._session_id = new_session
        self._session_turn_seq = None
        print(f"[!] 已开始新会话(上一段 {old_session[:8]}… 仍保存在会话记录里)")
        return True

    def on_remember(self):
        if self.state != self.IDLE:
            return
        history_snapshot = self.conversation.snapshot()
        if not history_snapshot or not self.ready:
            return
        if getattr(self, "_stale_generation", False):
            self._warn("上一条本地生成尚未完全停止，暂不能整理记忆。")
            return
        op_id = self._next_operation_id()
        self.active_operation_id = op_id
        self._set_state(self.REMEMBERING)
        self.status.setText("正在整理记忆……")
        token = CancellationToken()
        self.cancel_token = token
        if self.api_cfg:
            self.mem_worker = ApiMemoryWorker(self.api_cfg, history_snapshot, self.args, cancel_token=token)
        else:
            self.mem_worker = MemoryWorker(self.model, self.tokenizer, history_snapshot, self.args,
                                          cancel_token=token)
        self.mem_worker.done.connect(lambda entries, reason, _id=op_id: self.on_memory_done(entries, reason, _id))
        if hasattr(self.mem_worker, "degraded"):
            self.mem_worker.degraded.connect(self.on_thinking_degraded)
        self.mem_worker.start()

    def on_memory_done(self, entries, reason="", op_id=None):
        if self._is_closing():
            return
        if op_id is not None and op_id != self.active_operation_id:
            return
        try:
            if entries is None:
                if reason == "已取消":
                    self.status.setText("记忆整理已取消")
                else:
                    self.status.setText(f"记忆整理失败：{reason or '未知原因'}")
                return
            added = add_memories(self.memories, entries)
            try:
                saved = self._save_memories()
            except Exception as e:                      # 磁盘满/权限等异常必须一并捕住:只捕 OSError 会让它们穿出去
                self.status.setText(f"记忆保存失败：{e}")
                return
            if not saved:
                # 守卫会拒绝"把已有记忆清空"(例如读取失败后拿到空列表):
                # 此时必须如实说明未写盘,并把内存里刚加进去的条目回滚,不能谎报"已记住"
                self.memories = load_memories()
                self.status.setText("记忆未写入磁盘：被「疑似误清空」保护拦下,已回滚本次新增")
                return
            msg = f"已记住 {len(self.memories)} 条回忆"
            if added:
                msg += f"（本次新增 {added} 条）"
            else:
                msg += "（无新内容）"
            self.status.setText(msg)
        finally:
            self.cancel_token = None
            self._set_state(self.IDLE)

    # ==================== Agent 模式 ====================
    def _disable_agent(self, reason):
        """统一关闭 Agent:记录原因、在界面上明示。

        能力探测失败、凭据缺失、初始化异常都走这里。界面上没有开关可禁用 Agent
        (「Agent 联网」按钮已移除),所以要保证主人仍能一眼看到"为什么这次没走 Agent 路径"。
        """
        self.agent_ready = False
        self.agent_disabled_reason = reason
        # Agent 判定为不可用之后,任何在飞的探测结论也不许再把它翻回来
        self._discard_pending_probe()
        if hasattr(self, "send_btn"):
            self.send_btn.setText("发送")
        self.status.setText("⚠️ " + reason + "（本次起消息退回普通聊天）")
        # 只按当前状态刷新按钮可用性:硬推 IDLE 会把正在进行的生成/整理切回可发送,
        # 从而出现两次并发 generate 抢同一份本地权重
        self._set_state(self.state)

    def _setup_agent(self):
        """API 模式下初始化 Agent:配置、凭据域确认、凭据、能力探测。任何失败都明确说明原因。

        整体包一层 try:AgentStore 建库(mkdir/connect/PRAGMA/executescript)等任一步骤失败
        都不应冒泡出 __init__ —— 否则连普通聊天都用不了,与模块顶部
        「Agent 依赖缺失时普通聊天照常可用」的设计意图相悖。
        """
        try:
            self._setup_agent_unchecked()
        except Exception as e:
            self._disable_agent(f"Agent 初始化失败: {e}")

    def _setup_agent_unchecked(self):
        if not _AGENT_AVAILABLE:
            self._disable_agent("Agent 依赖缺失(jsonschema/keyring/send2trash),请安装后重启")
            return
        try:
            self._agent_doc = _load_agent_config(DEFAULT_CONFIG_PATH)
        except Exception as e:
            self._disable_agent(f"Agent 配置读取失败: {e}")
            return
        if not self._agent_doc.agent.enabled:
            self._disable_agent("Agent 未在配置中启用(agent.enabled=false)")
            return
        # 凭据外发域名门禁:非官方地址必须经用户确认(信任根=Windows 凭据管理器)
        base = (self._agent_doc.api.base_url or "").strip().lower().rstrip("/")
        if not _host_is_allowed(base):
            self._ask_host_confirmation(base)
            return
        self._continue_agent_setup()

    def _continue_agent_setup(self):
        """继续初始化(确认卡、迁移卡等回调也会走到这里)，因此同样整体保护。"""
        try:
            self._continue_agent_setup_unchecked()
        except Exception as e:
            self._disable_agent(f"Agent 初始化失败: {e}")

    def _continue_agent_setup_unchecked(self):
        self._agent_credential_id = self._agent_doc.api.credential_id
        credential = CredentialStore().get(self._agent_credential_id)
        if not credential and self._agent_doc.has_legacy_secret:
            self._ask_migrate_legacy_key()
            return
        if not credential:
            self._disable_agent("未找到 DeepSeek 凭据(Windows 凭据管理器)")
            return
        self._agent_credential = credential
        self._create_agent_controller()
        # 能力探测(会产生真实付费请求)延后到用户首次勾选 Agent 开关时执行

    def _start_agent_probe(self):
        if getattr(self, "_agent_probed", False) or self.agent_controller is None:
            return
        self._agent_probed = True
        self._arm_probe_watchdog()
        Thread(target=self.agent_controller.probe, daemon=True).start()

    def _discard_pending_probe(self):
        """作废"在飞"的探测结论。

        探测线程卡在网络上时取消不掉,但只要把它的**代次**作废,它回来时控制器就会
        丢弃结论 —— 否则主人已经选了"用普通聊天",几秒后一条迟到的成功结论又会把
        Agent 标成可用(`agent_ready` 从 False 翻回 True)。
        """
        controller = getattr(self, "agent_controller", None)
        if controller is None:
            return
        try:
            controller.discard_pending_probe()
        except Exception as e:
            print(f"[!] 作废在飞探测失败(不影响本次对话): {e}")
    def _arm_probe_watchdog(self):
        """给探测装一个兜底计时器(卡死时在状态栏说明)。

        惰性创建 + 容错:`QTimer(self)` 要求 `self` 是**构造完成的** QObject,而实例可能
        是 `__new__` 造出来的窗口,没走完 `__init__`;建不出来就跳过兜底(它是提示,不是功能必需)。
        """
        watchdog = getattr(self, "_probe_watchdog", None)
        if watchdog is None:
            try:
                watchdog = QTimer(self)
            except (RuntimeError, TypeError):
                return
            watchdog.setSingleShot(True)
            watchdog.timeout.connect(self._on_probe_watchdog)
            self._probe_watchdog = watchdog
        watchdog.start(_PROBE_WATCHDOG_MS)

    def _ask_host_confirmation(self, base):
        """API 地址不在信任根内:发送凭据前必须经主人确认。"""

        def do_allow(card):
            card.setEnabled(False)
            # 1) 写入信任根(Windows 凭据管理器)。失败则恢复卡片让主人可重试,
            #    绝不允许「只写了 api_config.json 就当确认过」。
            try:
                self._persist_allowed_host(base)
            except Exception as e:
                card.setEnabled(True)
                self.status.setText(f"⚠️ 保存域名确认到凭据管理器失败: {e}")
                return
            # 2) 重新加载 Agent 配置(同样允许重试)
            try:
                self._agent_doc = _load_agent_config(DEFAULT_CONFIG_PATH)
            except Exception as e:
                card.setEnabled(True)
                self.status.setText(f"⚠️ 重新加载 Agent 配置失败: {e}")
                return
            self.status.setText(f"已确认向 {base} 发送凭据,继续初始化……")
            # 域名未确认时普通聊天也被 load_api_config 挡住,这里一并恢复
            self._refresh_api_config_after_host_confirmation()
            self._continue_agent_setup()

        def do_refuse(card):
            card.setEnabled(False)
            self._disable_agent(f"用户拒绝向 {base} 发送凭据,Agent 保持禁用")

        self._add_action_card(
            host_confirmation_text(base),
            [("允许,继续", do_allow), ("拒绝,禁用 Agent", do_refuse)],
        )

    def _persist_allowed_host(self, host):
        """把主人确认的域名写入信任根(实现见 ui/dialogs.py 的 persist_allowed_host)。

        路径作为显式参数传入,于是「凭据管理器写失败必须抛出、
        且不得写入 api_config.json」这类安全语义可以用临时文件完整验证。
        """
        return persist_allowed_host(host, DEFAULT_CONFIG_PATH)

    def _refresh_api_config_after_host_confirmation(self):
        """确认域名后重新读取 api_config.json,补回被域名门禁挡住的普通聊天。

        「域名未确认」时 load_api_config() 返回 (None, error)，若不在这里补上，
        主人点完「允许」依然没有可用的聊天引擎。
        """
        if self.args.local or self.api_cfg is not None:
            return
        cfg, err = load_api_config()
        if err:
            self.status.setText("⚠️ " + err)
            return
        if not cfg:
            return
        self.api_cfg, self.api_error = cfg, None
        self.ready = True
        self.status.setText(
            f"已连接 DeepSeek API（{cfg['model']}）· 已记得 {len(self.memories)} 条关于主人的回忆"
        )
        # 同上:确认域名只是补回引擎,不应打断正在进行的生成
        self._set_state(self.state)

    def _ask_migrate_legacy_key(self):
        """检测到旧版明文 Key:要求用户确认迁移到凭据管理器。"""

        def do_migrate(card):
            card.setEnabled(False)
            try:
                migrate_legacy_key(DEFAULT_CONFIG_PATH, CredentialStore(), self._agent_credential_id)
                self.status.setText("已迁移 Key 至 Windows 凭据管理器,正在重新加载配置……")
                self._agent_doc = _load_agent_config(DEFAULT_CONFIG_PATH)
                credential = CredentialStore().get(self._agent_credential_id)
                if credential:
                    self._agent_credential = credential
                    self._create_agent_controller()
                else:
                    self._disable_agent("迁移后仍未找到 DeepSeek 凭据")
            except Exception as e:
                # 迁移失败:恢复卡片让主人可重试,并显示真实异常
                card.setEnabled(True)
                self.status.setText(f"⚠️ 迁移失败: {e}")

        def do_refuse(card):
            card.setEnabled(False)
            self._disable_agent("用户拒绝了明文 Key 迁移,Agent 保持禁用")

        self._add_action_card(
            migrate_key_text(),
            [("迁移到凭据管理器", do_migrate), ("暂不,禁用 Agent", do_refuse)],
        )

    def _create_agent_controller(self):
        doc = self._agent_doc
        model_name = doc.api.agent_model or "deepseek-v4-flash"
        client = DeepSeekAgentClient(
            model=model_name, api_key=self._agent_credential, base_url=doc.api.base_url,
            max_output_tokens=doc.agent.max_output_tokens,
            # 流式反馈(正文增量实时上屏);端点不支持时适配器自己退回。
            stream=getattr(doc.agent, "stream", True),
        )
        executor = build_agent_executor(PermissionPolicy(doc.agent),
                                        include_system=doc.agent.extra_tools_enabled,
                                        # 网页读取工具默认关闭;开启后每次调用仍需审批。
                                        include_web=getattr(doc.agent, "web_tools_enabled", False),
                                        web_timeout=getattr(doc.agent, "web_timeout_seconds", 10),
                                        # 关键词搜索:默认关闭,由 ddgs 库提供。
                                        # 代理为空=直连;受限链路必须配代理才搜得到。
                                        include_web_search=getattr(doc.agent, "web_search_enabled",
                                                                   False),
                                        web_search_proxy=getattr(doc.agent, "web_search_proxy", ""),
                                        web_search_engine=getattr(doc.agent, "web_search_engine",
                                                                  "auto"),
                                        web_search_max_results=getattr(
                                            doc.agent, "web_search_max_results", 5),
                                        web_search_timeout=getattr(
                                            doc.agent, "web_search_timeout_seconds", 15),
                                        web_search_allow_private=getattr(
                                            doc.agent, "web_search_allow_private", False))
        # 复用已有连接:重复构造 AgentStore 会在同一次会话里又跑一遍
        # mark_interrupted_runs(),把**正在运行**的任务标成 interrupted;
        # 而且旧连接被 GC 掉后,进行中任务后续的审计写入会静默失败。
        store = self._agent_store
        if store is None:
            store = AgentStore(os.path.join(_HERE, "agent_data", "agent.db"),
                               retention_days=doc.agent.retention_days)
        # 保留引用：closeEvent 里要显式 close()，否则 SQLite 连接永不释放
        self._agent_store = store
        settings = doc.agent

        factory = _AgentRunnerFactory(
            client=client,
            executor=executor,
            store=store,
            settings=settings,
            # 与聊天共用同一套人设与记忆预算(ContextBuilder.agent_instructions):
            # 若这里传**全部**记忆,而转录只带预算内的记忆 ——
            # "两种模式记得的事不同"会从这个后门回来。
            instructions_provider=lambda: context_builder(self.args).agent_instructions(
                self.memories) + (
                f"\n【工作目录】允许根目录:{doc.agent.allowed_root}。"
                "文件新建/读取任务请直接使用该目录内的路径;"
                "如果主人没给出文件名或路径,先向主人确认,不要浏览系统目录(如 C:\\、E:\\ 根目录)。"),
        )
        self.agent_controller = AgentController(factory)
        self.agent_controller.event_received.connect(self._on_agent_event)
        self.agent_controller.run_finished.connect(self._on_agent_finished)
        self.agent_controller.capability_reported.connect(self._on_capability_reported)
        # 把上次运行"真正做成了什么"补进会话历史(重启后仍能回答"文件在哪")
        self._restore_agent_result()
        # **不在启动时自动探测**。探测是**付费请求**(两步,第二步还带内置
        # web_search 并要求"新鲜信息+来源",等于主动触发一次搜索),不能在主人没开口时
        # 就花他的钱。改为:第一次真正要执行任务时先征求同意,结论缓存在进程内。
        self.agent_ready = True
        self._agent_probed = False
        self._set_state(self.IDLE)

    def _confirm_agent_probe(self, text):
        """首次执行任务前,就"要不要发起能力探测"征求同意。

        探测内容如实写清:会产生 1~2 次真实请求(计费),其中一次会尝试联网搜索。
        主人拒绝时排队的消息**不丢** —— 退回普通聊天照常回答。
        """
        self._pending_offer_turn = getattr(self, "active_operation_id", None)
        # "已经问过"不等于"卡片还能点" —— 卡片一旦被点过就永久禁用(setEnabled(False)),
        # 此时再让主人去点那张卡等于把他锁死。只有**当前确实有活卡片**时才只提示。
        if getattr(self, "_agent_probe_asked", False) \
                and getattr(self, "_probe_card_live", False):
            self.status.setText("Agent 能力仍未确认:请点上面的卡片做选择"
                                "(允许探测 / 用普通聊天)")
            return
        self._agent_probe_asked = True
        self._probe_card_live = True

        def allow(card):
            # 与「执行这个任务」同一条兜底 —— 生成/整理中绝不启动第二次运行
            if not self._probe_choice_allowed(card):
                return
            self.status.setText("正在确认 Agent 能力……")
            self._start_agent_probe()

        def decline(card):
            if not self._probe_choice_allowed(card):
                return
            self._fall_back_to_plain_chat("主人选择了不探测 Agent 能力")

        self._add_action_card(
            "【需要主人确认】启用 Agent 能力\n"
            "这需要向 DeepSeek 端点发起 1~2 次**真实请求**(会计费),其中一次会尝试联网搜索。\n"
            "• 允许探测:确认后这条消息会作为任务执行(工具调用仍需逐次批准)\n"
            "• 用普通聊天:不探测、不额外计费,这条消息按普通聊天回复",
            [("允许探测", allow), ("用普通聊天", decline)],
        )

    def _fall_back_to_plain_chat(self, reason: str):
        """放弃 Agent、退回普通聊天(主人拒绝探测 / 探测超时两条路共用)。

        排队消息**一条都不能丢** —— 队列里有多少条就答多少条。
        普通聊天一次只能跑一个 worker,所以第 1 条立刻答,其余进 backlog,回复完接着答。
        """
        self._pending_offer_turn = None
        # 主人说不要 Agent —— 任何还挂在网络上的探测结论都不许再回来翻案
        self._discard_pending_probe()
        # 能力**已有定论**(不要 Agent),必须把 _agent_probed 一起置真:否则后续每条消息
        # 都会再走"尚未确认"分支排队,而没有探测在跑 → 永远没人放行,消息被静默吞掉。
        self._agent_probed = True
        self.agent_ready = False
        self.agent_disabled_reason = reason
        self._set_state(self.IDLE)
        offers = list(getattr(self, "_pending_agent_offers", []))
        self._pending_agent_offers = []
        queued = f"刚才排队的 {len(offers)} 条会依次回答。" if offers else ""
        self._warn("好的,这次就不探测了,接下来的消息都会用普通聊天回复。" + queued
                   + "想启用 Agent,重启后再发一条消息即可重新选择。")
        if offers:
            # 这两条都是"替主人自动开始",绝不能清掉他正在输入的草稿
            self._start_plain_chat(offers[0][0], clear_input=False)
            backlog = getattr(self, "_plain_chat_backlog", [])
            backlog.extend(text for text, _shown in offers[1:])
            self._plain_chat_backlog = backlog

    def _probe_choice_allowed(self, card) -> bool:
        """确认卡两个按钮的公共闸门。

        卡片弹出后可能已经在生成/整理记忆 —— 此时若照做,就会与在途请求撞车
        (覆盖 `active_operation_id`,刚发出去的那条回复被丢弃)。与 `_accept_task_offer`
        一样:不合格就**不消费这张卡**,并如实说明,主人等空闲后再点即可。
        """
        if self.state != self.IDLE:
            self._warn("现在还在处理上一条消息:等它结束再来选择,或先点「停止」。")
            return False
        card.setEnabled(False)
        # 这张卡被消费掉了 —— 以后不能再让主人去点它(它已经点不动了),
        # 于是"已问过"标记要连同这个事实一起被解读,见 `_confirm_agent_probe`。
        self._probe_card_live = False
        return True

    def _drain_plain_backlog(self):
        """普通聊天答完一条后,接着答下一条(退回普通聊天时不丢消息)。"""
        backlog = getattr(self, "_plain_chat_backlog", None)
        if not backlog:
            return False
        if self.state != self.IDLE or getattr(self, "_stale_generation", False):
            # 还在忙/上次本地生成没退干净:留到下一个结束点再答(绝不能并发抢同一份权重)
            return False
        # 自动答积压**不碰输入框** —— 主人可能正在打下一条消息
        self._start_plain_chat(backlog.pop(0), clear_input=False)
        return True

    def _on_probe_watchdog(self):
        """探测久久不回来时:不能只提示,**必须给出可执行的出路**。

        只写一句状态栏提示会形成死锁:探测线程卡住 → `_agent_probed` 一直是
        True → 之后每条消息都排队 → 而重新探测被 `_start_agent_probe` 的守卫直接挡回,
        队列永远没人放行,停止按钮又不可用(状态是空闲)—— 此时
        `_agent_probed=True`、队列里躺着消息、界面没有任何按钮可点。

        所以:复位守卫(允许重试)+ 明确告知 + 给两个真按钮(重新探测 / 改用普通聊天)。
        卡住的那个线程是守护线程,它若后来真的回来了,`_on_capability_reported` 照常
        采用它的结论 —— 那是真实信息,不需要丢弃。
        """
        if getattr(self, "_capability_report", None) is not None:
            return                      # 已经回来了,什么都不用说
        self._stop_probe_watchdog()
        self._agent_probed = False      # 复位:否则重试会被守卫挡回,消息永远排队
        # 放弃等待 = 那份结论不再采用(它若迟到回来,不许把 Agent 标成可用)
        self._discard_pending_probe()
        self.status.setText("⚠️ Agent 能力探测超过 20 秒仍未返回。为避免消息一直等着,"
                            "已停止等待 —— 请选择下面的下一步。")

        def retry(card):
            if not self._probe_choice_allowed(card):
                return
            self._agent_probed = False
            self.status.setText("正在重新确认 Agent 能力……")
            self._start_agent_probe()

        def use_plain(card):
            if not self._probe_choice_allowed(card):
                return
            self._fall_back_to_plain_chat("Agent 能力探测超时(尚未返回)")

        self._add_action_card(
            "【探测没有返回】Agent 能力仍未确认\n"
            "• 重新探测:再发起 1~2 次**真实请求**(会计费);成功后排队的消息会自动执行\n"
            "• 用普通聊天:不再探测,排队和之后的消息都用普通聊天回答",
            [("重新探测", retry), ("用普通聊天", use_plain)],
        )
        # 这张兜底卡就是当前**活着**的选择入口(被点过的卡一律永久禁用)
        self._agent_probe_asked = True
        self._probe_card_live = True

    def _capability_tooltip(self, report) -> str:
        """能力探测结论(现在只有一个用处:告知**端点内置**联网搜索的探测结果)。

        界面上没有「Agent 联网」开关,所以这段文字不挂在开关上,而是作为
        能力说明由 `_capability_summary()` 统一给出。
        """
        detail = report.web_search.evidence if not report.web_search_usable else ""
        if not detail:
            return ""
        return (f"**端点内置**联网搜索"
                f"{('已确认不可用' if report.web_search.verified else '未验证')}:{detail}")

    def _on_capability_reported(self, report, generation=None):
        """按分项能力更新界面状态(不再有开关要同步)。

        **不把整段报告写进状态栏**:能力探测要发 2 次请求(耗时 2~3 秒),而主人完全可能
        在这期间就发出任务 —— 报告回来时任务可能已经结束,写状态栏会把"任务完成"
        覆盖成探测结论。结论放进 `_capability_summary()`,状态栏留给正在发生的事。

        信号带着**探测代次**,这里再核对一次。跨线程信号是排队的:"探测已跑完、
        信号已排队"与"UI 真正处理它"之间,主人可能已经选了「用普通聊天」——
        只靠控制器发送前那次核对挡不住这种迟到送达(`agent_ready` 会被翻回 True)。
        """
        if generation is not None:
            controller = getattr(self, "agent_controller", None)
            current = getattr(controller, "probe_generation", generation)
            if generation != current:
                print(f"[!] 忽略了过期的能力探测结论(代次 {generation} ≠ {current})")
                return
        if report.agent_usable:
            self.agent_ready = True
            self._capability_report = report
            self._stop_probe_watchdog()
            self._set_state(self.state)
            self._flush_pending_offer()      # 待办任务 / 启动时排队的消息
            return
        reason = report.reason or "未知原因"
        self._stop_probe_watchdog()
        if report.retryable:
            # 瞬时故障(网络抖动/限流)不能永久禁用:保持可用,下次发送时再征求一次同意。
            # **但排队中的消息必须收尾** —— 不收尾会既没执行、也没告知,
            # 主人的那条消息凭空消失。收尾交给同一个闸门如实说明。
            self._agent_probed = False
            self.agent_ready = True
            # 上一次的确认卡已经**被点掉了**(用完即废),所以这里必须让下一次发送
            # 能重新弹卡 —— 否则 `_confirm_agent_probe` 会走"已经问过"分支,只提示去点
            # 一张早已禁用的卡片:消息进队列后再也没有出路。
            self._agent_probe_asked = False
            self._probe_card_live = False
            # 与成功分支同一条守卫:状态栏刚被任务结果刷新过时,探测结论不许覆盖它
            # (否则"Agent 任务完成: OK"会被这句警告顶掉)。
            if not getattr(self, "_task_status_fresh", False):
                self.status.setText("⚠️ Agent 能力探测未成功(疑似瞬时故障):" + reason
                                    + "\n（再发一条消息时会重新征求一次是否重新探测）")
            self._set_state(self.state)
            self._flush_pending_offer()
            return
        # 先置 agent_ready=False 再 _set_state：否则 _set_state 会按就绪状态把发送按钮
        # 的文案/可用性算错，导致消息被静默丢弃
        # `_agent_probed` 一并置真:这是**确定性**结论(端点/模型就是不支持),不是瞬时故障 ——
        # 保持 False 会让之后每一条任务类消息都重新弹一次付费探测卡,失败一次花一次钱。
        # 要重新试,重启一次即可(瞬时故障那条路径仍然会重新征求同意)。
        self._agent_probed = True
        self._disable_agent("DeepSeek 端点/模型不支持 Agent 能力: " + reason)
        self._flush_pending_offer()          # 待办任务要如实告知为什么没执行

    def _stop_probe_watchdog(self):
        """探测有结论了就撤掉兜底计时器(否则它会在 20 秒后误报"尚未返回")。"""
        watchdog = getattr(self, "_probe_watchdog", None)
        if watchdog is not None:
            try:
                watchdog.stop()
            except RuntimeError:      # 窗口没有真实计时器
                pass

    def _capability_summary(self) -> str:
        """能力说明(拼给主人看的完整结论,界面上按需展示)。"""
        report = getattr(self, "_capability_report", None)
        if report is None:
            return "Agent 能力尚未探测"
        parts = ["Agent 就绪" if self.agent_ready else f"Agent 不可用:{self.agent_disabled_reason}"]
        detail = self._capability_tooltip(report)
        if detail:
            parts.append(detail)
        return " · ".join(parts)

    def _on_stop_agent(self):
        """停止按钮:GENERATING/REMEMBERING/AGENT_RUNNING 都可用，按当前活跃对象分派。"""
        if self.state == self.AGENT_RUNNING:
            if self.agent_controller is not None:
                self.agent_controller.cancel()
            self.status.setText("正在停止 Agent 任务……")
            return
        if self.state == self.GENERATING:
            target, message = self.worker, "正在停止生成……"
        elif self.state == self.REMEMBERING:
            target, message = self.mem_worker, "正在停止记忆整理……"
        else:
            return
        if self.cancel_token is not None:
            try:
                self.cancel_token.cancel()
            except Exception:
                pass
        cancel = getattr(target, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        self.status.setText(message)

    def _start_agent_run(self, text, user_bubble_shown=False, clear_input=True):
        op_id = self._next_operation_id()
        self.active_operation_id = op_id
        # 自动排空 FIFO 时**不许**动输入框 —— 那里可能是主人正在打的下一条消息
        if clear_input:
            self.input.clear()
        self._last_turn_was_agent = True
        self.status.setText("🛠 正在执行任务……")
        # 走同一个会话服务:Agent 路径若只 append 不裁剪,历史会无限增长
        self.conversation.begin_turn(text, ENGINE_AGENT)
        self._begin_turn_persistence(text, ENGINE_AGENT)
        self._agent_turn_appended_assistant = False   # 本轮是否已写入 assistant 回合
        # 流式缓冲必须在**每次运行开始**时清空,
        # 否则上一轮的半截文字会被这一轮的增量拼在一起。
        self._agent_stream_text = ""
        self._agent_streamed_this_turn = False
        self._agent_thinking_noted = False
        if not user_bubble_shown:
            # 从聊天"一键执行"过来时,主人那句话的气泡**已经**在聊天区了,不要再来一个
            self.add_bubble(text, is_user=True)
        self.bot_bubble = self.add_bubble("……", is_user=False)
        self._set_state(self.AGENT_RUNNING)
        # 人物连续性:历史对话以**结构化角色**交给 Agent(不压成一条 user 输入、
        # 也不截取最后 3000 字符);本次知识库设定参考作为 instructions 补充。
        builder = context_builder(self.args)
        prior = self.conversation.snapshot()[:-1]      # 去掉本轮刚写入的 user
        agent_context = builder.build_agent(prior, self.memories, text)
        from kurumi.context import RunContext
        # 把会话 id 一并交给运行 —— 审计库里这次运行就归属这段对话,
        # 恢复"上次 Agent 结果"时按会话查,清空之后旧任务不会跨边界复活。
        # getattr 兜底:实例可能是 __new__ 造出来的窗口,少一个属性不该让整轮运行起不来。
        self.agent_controller.start(
            text, RunContext.from_agent_context(
                agent_context, session_id=getattr(self, "_session_id", "")))

    def _on_agent_event(self, event):
        if self._is_closing():
            return
        kind = event.kind
        if kind == "final_text":
            final = event.payload.get("text", "") or event.message
            if self.bot_bubble:
                _set_bubble_text(self.bot_bubble, final)
            self.bot_bubble = None
            self._agent_stream_text = ""
            self._agent_streamed_this_turn = False
            if final.strip():
                # 与聊天共用同一套会话状态:完成一轮会累加轮数(Agent 回合若不计入
                # turn_count,"每 N 轮整理记忆"在用 Agent 时永不触发)
                self.conversation.complete_turn(final, ENGINE_AGENT)
                self.turn_count = self.conversation.turn_count
                self._record_turn("assistant", final, ENGINE_AGENT)
                # 记录本轮已写入 assistant：若随后任务失败/取消，_on_agent_finished 要一并回滚
                self._agent_turn_appended_assistant = True
        elif kind == "persona_text":
            # 人设说明文字(与工具调用同回合):显示在气泡中,审批卡随后出现
            text = event.payload.get("text", "") or event.message
            if text.strip():
                if self.bot_bubble is None:
                    self.bot_bubble = self.add_bubble(text, is_user=False)
                elif getattr(self, "_agent_streamed_this_turn", False):
                    # **这一轮已经流式上屏过**同一段文字,
                    # 不能再追加一遍(追加会把同一句显示两次)。
                    _set_bubble_text(self.bot_bubble, text)
                else:
                    current = self.bot_bubble.text()
                    _set_bubble_text(self.bot_bubble, text if not current.strip() else current + "\n" + text)
                self._agent_stream_text = ""
                self._agent_streamed_this_turn = False
        elif kind == "agent_text_delta":
            # 流式正文增量。实时上屏,让主人第一时间看到模型在说什么,
            # 而不是盯着"……"等整轮结束(首个增量通常 2~3 秒就到)。
            #
            # 用 _append_bubble_text(自带节流):每个增量都重算宽高会拖慢界面。
            delta = event.payload.get("text", "") or event.message
            if not delta:
                return
            if self.bot_bubble is None:
                self.bot_bubble = self.add_bubble("", is_user=False)
                self._agent_stream_text = ""
            self._agent_stream_text = getattr(self, "_agent_stream_text", "") + delta
            self._agent_streamed_this_turn = True
            if _append_bubble_text(self.bot_bubble, self._agent_stream_text):
                self._scroll_bottom()
        elif kind == "reasoning_delta":
            # 思考内容**不上屏**:它是模型的过程,不是给主人的答复。
            # 只在状态栏提示"正在思考",避免主人以为界面卡住了。
            if not getattr(self, "_agent_thinking_noted", False):
                self._agent_thinking_noted = True
                self.status.setText("狂三正在思考……")
        elif kind == "tool_result":
            self._add_neutral_card(
                f"🔧 {event.payload.get('tool_name', '工具')} → {event.payload.get('code', '?')}\n"
                f"{event.payload.get('summary', '')[:200]}")
        elif kind == "web_action":
            self._add_neutral_card("🌐 正在联网搜索……")
        elif kind == "source":
            self._add_neutral_card("📎 来源: " + (event.payload.get("url") or event.message))
        elif kind == "approval_requested":
            self._add_approval_card(event)

    def _on_agent_finished(self, outcome):
        if self._is_closing():
            return
        state = getattr(outcome, "state", None)
        completed = (state == RunState.COMPLETED) or (getattr(state, "value", None) == "completed")
        code = str(getattr(outcome, "code", "?") or "?")
        message = str(getattr(outcome, "message", "") or "").strip()
        detail = code if not message else f"{code} — {message}"
        # 结构化结果(task_result 为 None 时退化为无结构化结果的路径)
        task = getattr(outcome, "task_result", None)
        # 气泡里「模型自己说的话」优先;模型没给文本时才用程序事实兜底 ——
        # 成功时若覆盖成「（任务完成）」,会把模型刚说的话抹掉。
        if self.bot_bubble is not None:
            current = self.bot_bubble.text().strip()
            if not current or current == "……":
                if task is not None and getattr(task, "artifacts", ()):
                    _set_bubble_text(self.bot_bubble, "已完成，产物：" + "、".join(task.artifacts))
                elif completed:
                    _set_bubble_text(self.bot_bubble, "（任务完成）")
                else:
                    _set_bubble_text(self.bot_bubble, f"（任务未完成：{detail}）")
        self.bot_bubble = None
        # 流式已经上屏、但本轮没有走到 final_text 的文字,
        # 必须一并留在历史里 —— 否则主人明明看到了半截答复,下一轮模型却完全不知道
        # 自己说过什么(把这段文字只留在界面上,就会出现这种不一致)。
        streamed = (getattr(self, "_agent_stream_text", "") or "").strip()
        kept_note = False
        if not completed:
            # 失败/取消:回滚本轮(含已写入的 assistant 回合 + 未配对的 user),避免上下文损坏。
            # 但**已经落盘的副作用不能说没就没**:若本轮真的写过文件,回滚后补一条事实记录,
            # 否则下一轮被问「文件生成了吗」时,模型与程序都无从知晓(主人点名的验收)。
            self.conversation.fail_turn(self._agent_turn_appended_assistant)
            if agent_has_side_effects(task):
                note = agent_history_note(task)
                self.conversation.append_note(
                    note, ENGINE_AGENT,
                    context_text="（系统记录：上一次任务中断，以下是程序记录的实际结果）")
                self._record_turn("assistant", note, ENGINE_AGENT)
                kept_note = True
            elif streamed and not self._agent_turn_appended_assistant:
                # 没有副作用、但屏幕上确实出现过模型的话:留下它,并标明这是**未完成**的
                note = f"[任务未完成：{code}] 我上一条（未说完）：{streamed}"
                self.conversation.append_note(
                    note, ENGINE_AGENT,
                    context_text="（系统记录：上一次任务中断，以下是我没说完的话）")
                self._record_turn("assistant", note, ENGINE_AGENT)
                kept_note = True
            # 库里的轮次要和内存"一致或更好":
            #   · 补了程序记录 → user + 这条记录是一对完好的轮次,连主人的问题一起保留;
            #   · 什么都没留下 → 把这次写进库的行全撤掉(否则是一条孤立 user,重启会被载入)。
            if kept_note:
                self._commit_turn_persistence()
            else:
                self._rollback_turn_persistence()
        else:
            self._commit_turn_persistence()
        self._agent_turn_appended_assistant = False
        self._agent_stream_text = ""
        self._agent_streamed_this_turn = False
        # 任务面板:程序如实生成的结构化交付清单(与人设语气无关)
        panel = agent_task_panel(task)
        if panel:
            self._add_neutral_card(("⚠️ " if not completed else "✅ ") + panel)
        self._set_state(self.IDLE)
        # 标记"状态栏刚被任务结果刷新过":晚到的能力探测报告不要覆盖它
        self._task_status_fresh = True
        # **成功也要写状态栏**:只在失败时写的话,任务跑完后状态栏会一直停在
        # "狂三正在思考……",主人无法判断是否结束。
        if not completed:
            self.status.setText(f"Agent 任务结束: {detail}")
        else:
            self.status.setText(f"Agent 任务完成: {detail}")
        # **先排空 FIFO,再考虑整理记忆**。反过来的话 `on_remember()` 会把状态改成
        # REMEMBERING,闸门随即把排队消息当成"并发"整队丢掉 —— 而且提示还会冤枉主人
        # ("期间又有新的消息在处理",可主人什么也没发)。
        started = self._flush_pending_offer() if getattr(self, "_pending_agent_offers", None) \
            else False
        # 记忆整理只在**成功**结束时计入节奏(失败/取消不计);
        # 上一次顺延下来的则一律补做。
        due = getattr(self, "_remember_due", False) or (
            completed and self.conversation.should_remember(self.args.remember_every))
        if started:
            # 记忆整理顺延到排队任务跑完。必须**记下来**:回合计数马上会变,
            # 不记的话 `should_remember` 之后就不再为真,这一轮永远不会整理(静默漏掉)。
            self._remember_due = due
        elif due:
            self._remember_due = False
            self.on_remember()

    def _add_neutral_card(self, text):
        # 控件构建细节在 ui/widgets.py
        return add_neutral_card(self, text)

    def _add_action_card(self, text, buttons):
        # 控件构建细节在 ui/widgets.py
        return add_action_card(self, text, buttons)

    def _add_approval_card(self, event):
        # 控件构建细节在 ui/widgets.py
        return add_approval_card(self, event)

    # ==================== 常规聊天 ====================
    def keyPressEvent(self, e):
        # 兜底:主力是 _build_ui 里注册的 QShortcut(窗口层 keyPressEvent 会被
        # QTextEdit 吞掉)。on_send 自身幂等,重复触发会因 state != IDLE 直接返回。
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and e.modifiers() & Qt.ControlModifier:
            self.on_send()
        else:
            super().keyPressEvent(e)

    def _close_agent_store(self):
        """释放 Agent 的 SQLite 连接(不显式 close 就永不释放)。"""
        store = getattr(self, "_agent_store", None)
        if store is None:
            return
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def closeEvent(self, event):
        # 关闭窗口：先协作式取消，限时等待；仍未结束则第一次请求确认，第二次强制退出。
        # 关键：QThread 在 run() 未返回时被 Python 析构会触发 Qt qFatal abort，
        # 因此 accept() 之前必须把所有仍在运行的线程对象转存到模块级列表保活。
        #
        # 注意：**不能在这里就作废在途回调**。首次关闭若因任务未结束而被 ignore()
        # （生成 2048 token 通常远超 shutdown_wait_ms），窗口仍然开着、本轮回复还得回来；
        # 早作废 op_id 会把 done/failed/chunk 全部吞掉，界面永久停在 GENERATING 且按钮全灰。
        # 因此作废与置 _closing 都推迟到真正 accept() 之前。
        if self.cancel_token is not None:
            try:
                self.cancel_token.cancel()
            except Exception:
                pass
        threads = [w for w in (getattr(self, "worker", None), getattr(self, "mem_worker", None),
                               getattr(self, "loader", None)) if w is not None]
        for w in threads:
            cancel = getattr(w, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
        agent_busy = False
        if self.agent_controller is not None:
            agent_busy = not self.agent_controller.shutdown(self.args.shutdown_wait_ms)
        running = [w for w in threads if w.isRunning()]
        if running or agent_busy:
            deadline = time.monotonic() + self.args.shutdown_wait_ms / 1000.0
            for w in running:
                remaining_ms = int(max(0.0, deadline - time.monotonic()) * 1000)
                try:
                    w.wait(remaining_ms)   # QThread.wait 不接受负值
                except Exception:
                    pass
            if (any(w.isRunning() for w in running) or agent_busy) and not self._force_close:
                self._force_close = True
                self.status.setText("仍有任务在运行：再次关闭窗口将强制退出")
                event.ignore()
                return
        # 真正要退出了：此刻才作废在途回调、禁止迟到回调触碰控件
        self.active_operation_id += 1
        self._closing = True
        # 仍在运行的线程转存保活，避免解释器析构 QThread 时 abort
        for w in threads:
            if w.isRunning():
                _keep_thread_alive(w)
        if not agent_busy:
            # Agent 线程仍活着时不动它的 SQLite 连接，避免正在写入的线程报错
            self._close_agent_store()
        event.accept()


# 默认路径：先在脚本同目录找模型与 LoRA，再找上一级目录，都找不到时回退 HuggingFace
_HERE = os.path.dirname(os.path.abspath(__file__))

# 能力探测的兜底时限(毫秒):超过它还没回来就在状态栏如实说明。
# 正常探测是 2~3 秒(2 次请求);给到 20 秒是为了容忍慢链路,同时不让
# "探测线程卡死"变成界面上的无声悬挂。
_PROBE_WATCHDOG_MS = 20000
_PARENT = os.path.normpath(os.path.join(_HERE, ".."))


def _first_existing(*candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


LOCAL_BASE = _first_existing(
    os.path.join(_HERE, "models", "Qwen3-4B"),
    os.path.join(_PARENT, "models", "Qwen3-4B"),
) or "Qwen/Qwen3-4B"
LOCAL_ADAPTER = _first_existing(
    os.path.join(_HERE, "saves", "qwen3-4b-kurumi"),
    os.path.join(_PARENT, "saves", "qwen3-4b-kurumi"),
) or ""


if __name__ == "__main__":
    # 启动入口已经搬到 main.py:这里只放界面实现。
    # 静默什么都不做会让人以为是程序没反应,所以直接说清楚该用哪个命令。
    raise SystemExit("启动入口是 main.py：请运行 `python main.py`（参数见 `python main.py --help`）。")
