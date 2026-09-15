# -*- coding: utf-8 -*-
"""任务意图识别(纯本地规则,不发任何请求)。

## 为什么用规则,而不是让模型判断

主人说"帮我在桌面建个文件"时,界面要决定"这是任务还是闲聊"。把这件事交给模型意味着:

1. **每次都要多花一次付费请求**(而且是主人还没决定要不要执行之前);
2. 更重要 —— "要不要动主人的文件/系统"这件事,判断权应该在**本地规则 + 主人的一次点击**,
   而不是模型的自由发挥。

因此这里的判定完全在本地:命中**动作动词**才算任务;只给目标、不给动作("桌面上的文件"
这种名词短语)一律不算。误判的代价是"多出一个可点的按钮"(无害),
漏判的代价是"任务被当闲聊"(主人会觉得没听懂),所以动词表要尽量贴近中文的常见说法。

## 与界面的分工

本模块只回答"这段文字像不像一个任务、像哪一类";**要不要执行由主人点按钮决定**
(见 `UI._offer_agent_run`),执行时仍然走完整的权限与审批链。
"""
import re
from dataclasses import dataclass

# 动作动词:命中即认为"这是一个要求做事的任务"。
#
# 刻意**不**收录的:
#   - "看/看看/查/搜"单独出现时多为提问("帮我看看这句话什么意思")—— 只有与具体目标
#     组合才有任务味,而那种组合里通常还有别的动词;
#   - "能不能/可以吗"这类**能力询问**("你能写文件吗")—— 见下方 _QUESTION_PATTERNS。
_FILE_VERBS = (
    "新建", "创建", "建一个", "建个", "建一", "写一个", "写个", "写入", "写到", "写进",
    "保存", "另存", "生成", "导出", "抓取", "爬取",
    "读取", "读一下", "读一读", "看下", "看看这个",
    "修改", "改成", "改为", "更新", "追加", "替换", "重命名", "改名", "覆盖",
    "删除", "删掉", "删了", "移除", "清理", "清空",
    "移动", "挪到", "移到", "剪切", "拷贝", "备份",
    "整理", "归类", "分类", "汇总", "合并", "拆分", "打个包", "解压",
)
# 这些词**既是动词也是名词**("我的下载目录""回收站""压缩包"):只有出现在
# "下载到…""把…回收""解压这个"这类**动作用法**里才算动作 —— 否则"我的下载目录"
# 会被误判成一条任务。
_AMBIGUOUS_VERB_USAGE = re.compile(
    r"(下载|回收|压缩|复制)\s*(到|进|下来|一下|这个|这份|它|它们|完)|"
    r"(把|将|帮我|帮忙|替我|给我|请|麻烦)\s*[^,。;!?]{0,12}(下载|回收|压缩|复制)")
_SYSTEM_VERBS = (
    "打开", "启动", "运行", "执行", "安装",
    "关机", "重启", "注销", "锁屏", "静音", "调音量",
)
_SEARCH_VERBS = ("搜索", "搜一下", "查一下", "查询", "检索", "上网找", "联网找")

# 目标线索分两档,因为它们的证明力不同:
#
# **强线索**:路径、扩展名、已知目录、"这个/那个+名词"、数量词 —— 指向一个**具体对象**,
#   足以把"能不能……"从"能力询问"变成"请求"("能帮我把报告写到桌面吗" vs "你能写文件吗");
# **弱线索**:文件/文档/数据这类**泛化名词** —— 出现在"支持创建文件夹吗"里并不能说明
#   主人在指派一件具体的事,所以**不参与**上面的区分。
_STRONG_TARGET_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),                       # E:\ 或 E:/
    re.compile(r"(桌面|下载|图片|视频|音乐|回收站)"),      # 常见目录名(不含泛化的"文件夹/目录")
    re.compile(r"\.[A-Za-z0-9]{1,5}\b"),                 # a.txt / report.md
    re.compile(r"(这个|那个|这些|那些|刚才|上次|上面|以下)[^,。;!?]{0,10}"
               r"(文件|文档|图片|表格|截图|记录|内容|数据)"),
    re.compile(r"(所有|全部|每个|批量|几个|若干)"),
    re.compile(r"\d+\s*(个|份|张|条|次|遍)"),             # "写 3 份""删 2 个"
)
_WEAK_TARGET_PATTERNS = (
    re.compile(r"(文件|文件夹|目录|路径|文档|数据|内容)"),
)
_ALL_TARGET_PATTERNS = _STRONG_TARGET_PATTERNS + _WEAK_TARGET_PATTERNS


# 明确的请求口吻(加强"这是命令而不是提问"的信号,但不作为必需条件)
_REQUEST_PREFIX = re.compile(r"^\s*(帮我|帮忙|替我|给我|请|麻烦|你去|狂三|你)?\s*")

# **问句**标记:这一类句子的默认解读是"在问",而不是"在派活"。
#
# 区分的关键不是"有没有能不能",而是"这句是不是在问":
#   "支持创建文件夹吗""怎么新建文件夹""为什么删除失败""要是能自动整理就好了" —— 都在问/假设;
#   "帮我建一个 test.md" —— 在派活。
# 因此有问句标记时,只有同时给出**强目标**(路径/扩展名/具体对象/数量)才算任务:
#   "能帮我把报告写到桌面吗" → 有"写到"+ 强目标 → 任务(这是最常见的礼貌请求说法);
#   "你能写文件吗" → 问句 + 只有泛化名词 → 不是任务。
_QUESTION_PATTERNS = (
    re.compile(r"[吗呢吧？?]\s*$"),                       # 句末疑问/祈使语气词
    re.compile(r"(怎么|如何|为什么|为何|是什么|什么是|哪些|哪种|教我|告诉我怎么)"),
    re.compile(r"(能不能|可不可以|可以吗|会不会|是否|支持不支持|行不行|办得到)"),
    re.compile(r"(如果|要是|假如|万一|倘若|若)"),
    re.compile(r"你(会|能|可以|支持)(不|做|写|读|删|改|打开|创建|移动|压缩)"),
)
# 句首的疑问词 = 求解释("怎么新建文件夹""为什么删除失败"),不是派活。
_QUESTION_LEAD = re.compile(r"^\s*(怎么|如何|为什么|为何|是什么|什么是|教我)")

# 交付/产物类说法(加强任务判定)
_DELIVERABLE = re.compile(
    r"(报告|周报|日报|月报|年报|总结|清单|表格|文件|文档|副本|备份|压缩包|截图|结果|"
    r"笔记|记录|代码|脚本|表格|图片|音频|视频)")


@dataclass(frozen=True)
class TaskIntent:
    """识别结果。`reason` 用于把"为什么认为这是任务"告诉主人(可核对,不是黑盒)。"""

    kind: str          # "file" / "system" / "search" / "generic"
    reason: str
    targets: tuple = ()


def _hit(text: str, words) -> list:
    return [w for w in words if w in text]


def ambiguous_verb_hits(text: str) -> list:
    """兼类词("下载/回收/压缩/复制")里**确实在做动词**的那些。

    单独暴露成函数是为了**可测**:这条守卫在完整判定链里只是第二道防线
    (前面还有问句、明确动作动词等规则),单看端到端句子无法把它单独区分出来 ——
    所以必须能直接对它断言,而不是靠端到端句子间接推断。
    """
    return [m.group(0).strip() for m in _AMBIGUOUS_VERB_USAGE.finditer(str(text or ""))]


def looks_like_task(text: str):
    """判断这段文字是否**像**一个任务;不像则返回 None。

    判定顺序(任一"否定信号"命中就直接判为不是任务):
    1. 空/过短 → 不是;
    2. 是问句/假设(吗、怎么、如果、能不能……)且**没有强目标** → 不是;
    3. 命中动作动词 → 是(按动词归属分类,并给出依据);
    4. 只命中目标、没有动作 → 不是(名词短语不等于命令)。
    """
    raw = str(text or "").strip()
    if len(raw) < 4:
        return None

    file_hits = _hit(raw, _FILE_VERBS)
    file_hits += ambiguous_verb_hits(raw)     # 兼类词只在动作用法里算动作
    system_hits = _hit(raw, _SYSTEM_VERBS)
    search_hits = _hit(raw, _SEARCH_VERBS)
    verb_hits = file_hits + system_hits + search_hits

    if not verb_hits:
        return None

    # 问句/假设:这类句子的默认解读是"在问",不是"在派活"。
    # 只有它同时给出**强目标**(路径/扩展名/具体对象/数量)时才算任务 ——
    # 那正是最常见的礼貌请求说法("能帮我把报告写到桌面吗");
    # 只带泛化名词("你能写文件吗""支持创建文件夹吗")一律不算。
    strong_hits = [rx.pattern for rx in _STRONG_TARGET_PATTERNS if rx.search(raw)]
    all_hits = [rx.pattern for rx in _ALL_TARGET_PATTERNS if rx.search(raw)]
    # "怎么/如何/为什么"开头的句子是**求解释**,即便里面有"桌面"这种具体词也不是派活。
    question_lead = _QUESTION_LEAD.search(raw[:8])
    question = [rx.pattern for rx in _QUESTION_PATTERNS if rx.search(raw)]
    # 显式祈使的搜索动词(搜索/搜一下/查一下/检索/上网找)本身就是"在派活"的信号,
    # 不该被句中的疑问词否掉 —— "搜索一下量子计算是什么"是在**让狂三去搜**,
    # 不是"什么是量子计算"那种纯提问。句首疑问词("怎么搜索文件")照旧算求解释。
    imperative_search = bool(search_hits) and not question_lead
    if (question or question_lead) and not (strong_hits and not question_lead) \
            and not imperative_search:
        return None

    targets = tuple(all_hits)
    if search_hits and not (file_hits or system_hits):
        return TaskIntent("search", f"命中搜索动词:{'、'.join(search_hits[:3])}", targets)
    if system_hits and not file_hits:
        return TaskIntent("system", f"命中系统动作:{'、'.join(system_hits[:3])}", targets)
    deliverable = "；还提到产物(" + _DELIVERABLE.search(raw).group(1) + ")" \
        if _DELIVERABLE.search(raw) else ""
    return TaskIntent("file", f"命中文件动作:{'、'.join(file_hits[:3])}{deliverable}", targets)


def suggestion_text(intent: TaskIntent) -> str:
    """气泡按钮上/下方显示的一句话,说明"为什么给你这个按钮"。"""
    return f"这看起来是个任务（{intent.reason}），要我现在去做吗？"


# 承接上文的短句:单独看没有动作动词,确实不像任务;但上一轮**正在做任务**时,
# 它就是在说"接着做"。—— 因此必须由界面结合"上一轮走的是哪条路"来判断,
# 单看这句话它更像闲聊(见 `is_continuation`)。
_CONTINUATION = re.compile(
    r"^\s*(继续|接着|接着来|往下|下一个|下一步|然后呢?|再来一次|再来|再试一次|再试|重试|"
    r"还没好吗|好了吗|做完了吗|怎样了|怎么样了|go\s*on|continue)\s*[。.!！?？~～、,，]*\s*$",
    re.IGNORECASE,
)


def is_continuation(text: str) -> bool:
    """这段文字是否在**承接上一轮的任务**("继续""然后呢""还没好吗")。

    调用方必须自己确认"上一轮真的在跑任务"再采用这个结果 —— 它是补充信号,
    不是独立的任务判定。
    """
    return bool(_CONTINUATION.match(str(text or "")))
