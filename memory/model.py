# -*- coding: utf-8 -*-
"""记忆的数据模型与纯函数(**不含 IO**,便于离线单独验证)。

每个条目带 **类型、来源、更新时间、替代关系、置顶与有效位**,并满足两条必须成立的验收要求:

1. "以前喜欢咖啡,现在只喝茶" → 旧条目作废、新条目生效,并记录 `supersedes`;
2. 角色**虚构经历**不能被记成主人的真实经历 → `kind=fiction`,且不参与主人的事实检索。

为什么必须有"显式指令 + 话题键启发式"两条替代路径:
语义矛盾只有模型能可靠判断(「咖啡」与「茶」字面上毫无交集),但离线验证与
主人手动录入又需要确定性行为,因此两者都要,且**显式指令优先**。
"""
import re

# 类型
KIND_PREFERENCE = "preference"     # 用户偏好
KIND_AGREEMENT = "agreement"       # 关系约定
KIND_STATE = "state"               # 短期状态
KIND_TASK_FACT = "task_fact"       # 任务事实
KIND_FICTION = "fiction"           # 角色虚构经历(不得当成主人的真实经历)
KIND_FACT = "fact"                 # 其它事实
KINDS = (KIND_PREFERENCE, KIND_AGREEMENT, KIND_STATE, KIND_TASK_FACT, KIND_FICTION, KIND_FACT)

# 来源
SOURCE_CHAT = "chat"
SOURCE_AGENT = "agent"
SOURCE_MANUAL = "manual"
SOURCE_LEGACY = "legacy"           # 旧文件里的条目(没有来源字段)
SOURCE_IMPORT = "import"

_DEFAULT_TIME = "1970-01-01 00:00"


def _now(now=None):
    if now:
        return str(now)
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


# ---------- 话题键:让"咖啡"与"茶"归到同一话题 ----------
# 领域词表:命中同一领域的两个说法视为**同一话题**,新条目会作废旧条目。
# 词表天然有限:覆盖不到的领域靠模型的显式 `取代:` 指令兜底。
_TOPIC_DOMAINS = {
    "饮品": ("咖啡", "茶", "奶茶", "可乐", "果汁", "牛奶", "豆浆", "酒", "啤酒", "水"),
    "食物": ("辣", "甜", "咸", "香菜", "海鲜", "牛肉", "猪肉", "素", "快餐", "外卖"),
    "颜色": ("红色", "蓝色", "绿色", "黑色", "白色", "紫色", "粉色", "黄色"),
    "宠物": ("猫", "狗", "仓鼠", "兔子", "鸟", "鱼"),
    "称呼": ("主人", "殿下", "先生", "小姐", "老板", "哥哥", "姐姐", "妹妹"),
    "作息": ("早睡", "熬夜", "早起", "午睡", "通宵"),
    "游戏": ("原神", "星穹铁道", "塞尔达", "我的世界", "王者荣耀", "英雄联盟"),
    "音乐": ("摇滚", "古典", "爵士", "民谣", "电音", "流行"),
}

# 谓词头:主体 + 谓词;主体**长者优先**(否则 "主人现在喝奶茶" 会先被 "主人" 吃掉,
# 剩下的 "现在喝奶茶" 匹配不到谓词 → 话题键为空 → 该替代的没替代)。
# 谓词覆盖"偏好"与"摄入"两类说法,因为它们说的是同一个话题(用户验收里正是
# "以前喜欢咖啡,现在只喝茶" 这种跨说法的更新)。
_PREDICATE_RE = re.compile(
    r"^(?:主人现在|主人最近|主人今天|主人|我)?"
    r"(喜欢|爱|讨厌|不喜欢|不爱|不吃|不喝|只喝|只吃|常喝|常吃|爱喝|爱吃|习惯|偏爱|"
    r"想喝|想吃|最爱|喝|吃)"
    r"(.*)$")

# 状态类谓词(当前状态:与偏好共用同一话题轴)
_STATE_PREDICATE_RE = re.compile(
    r"^(?:主人|我)?(?:现在|今天|最近|暂时)?(正在|在做|在忙|要|准备)(.*)$")


def topic_key(content: str) -> str:
    """返回同话题归并键;空字符串表示"无法归并"(不做自动替代)。

    规则:抽出"主体+谓词"确认这是一句状态/偏好陈述,再把宾语按**领域词表**归一。

    话题轴**只用领域,不区分谓词** —— "喜欢咖啡"与"只喝茶"说的是同一个话题(饮品),
    这正是用户验收里的更新场景。谓词差异(喜欢/讨厌)也算同话题的立场变化,
    由"更新即最新"处理。双方领域都认不出时返回空串:**宁可不替代,也不要错杀主人的记忆**。
    """
    text = str(content or "").strip().rstrip("。.!！~")
    if not text:
        return ""
    match = _PREDICATE_RE.match(text)
    if not match:
        return ""
    # **只在宾语部分**匹配领域词:否则"主人"这个主语会命中"称呼"领域,
    # 让每一句以"主人"开头的话都归到同一话题。
    obj = match.group(2)
    for name, words in _TOPIC_DOMAINS.items():
        if any(w in obj for w in words):
            return name
    return ""            # 认不出领域就不归并


# ---------- 类型判定 ----------
_FICTION_HINTS = ("精灵", "刻刻帝", "天使", "梦魇", "时空", "时间之眼", "子弹",
                  "我原本", "我的梦想", "三十年前", "空间震", "吞噬", "分身")
_AGREEMENT_HINTS = ("约定", "说好", "答应", "以后", "不许", "必须", "记住", "叫我", "称呼")
_PREFERENCE_HINTS = ("喜欢", "讨厌", "爱吃", "爱喝", "常喝", "常吃", "习惯", "偏爱",
                     "不喜欢", "不爱", "不吃", "不喝", "最爱", "只喝", "只吃")
_STATE_HINTS = ("今天", "现在", "正在", "暂时", "最近", "此刻", "刚刚")
_TASK_HINTS = ("文件", "报告", "已生成", "路径", "写入", "创建", "整理", "任务")

_KIND_ALIASES = {
    "偏好": KIND_PREFERENCE, "preference": KIND_PREFERENCE,
    "约定": KIND_AGREEMENT, "agreement": KIND_AGREEMENT,
    "状态": KIND_STATE, "state": KIND_STATE,
    "任务": KIND_TASK_FACT, "task": KIND_TASK_FACT, "task_fact": KIND_TASK_FACT,
    "虚构": KIND_FICTION, "fiction": KIND_FICTION, "角色": KIND_FICTION,
    "事实": KIND_FACT, "fact": KIND_FACT,
}


def classify_kind(content: str, source: str = "") -> str:
    """按内容判定记忆类型(显式指令由调用方优先处理)。"""
    text = str(content or "")
    if any(h in text for h in _FICTION_HINTS):
        return KIND_FICTION
    if any(h in text for h in _AGREEMENT_HINTS):
        return KIND_AGREEMENT
    if any(h in text for h in _PREFERENCE_HINTS):
        return KIND_PREFERENCE
    if any(h in text for h in _STATE_HINTS):
        return KIND_STATE
    if source == SOURCE_AGENT or any(h in text for h in _TASK_HINTS):
        return KIND_TASK_FACT
    return KIND_FACT


def parse_kind_directive(text: str):
    """从形如 `类型: preference` 的指令里取类型;识别不了返回 None。"""
    match = re.match(r"^\s*(?:类型|kind)\s*[:：]\s*([A-Za-z_]+|[\u4e00-\u9fa5]+)\s*$",
                     str(text or ""))
    if not match:
        return None
    return _KIND_ALIASES.get(match.group(1).strip().lower())


# ---------- 条目规范化(向后兼容) ----------
def normalize_entry(item, source: str = "", now=None):
    """补齐字段;返回 None 表示这条不合法。

    旧条目(只有 content/time)会被补成 kind=fact / source=legacy / active=true,
    **不改写调用方传来的原始数据**。
    """
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    content = content.strip()
    stamp = str(item.get("time") or "").strip() or _now(now)
    kind = item.get("kind")
    if kind not in KINDS:
        kind = classify_kind(content, source)
    src = str(item.get("source") or "").strip() or source or SOURCE_LEGACY
    entry = {
        "content": content,
        "time": stamp,
        "updated": str(item.get("updated") or "").strip() or stamp,
        "kind": kind,
        "source": src,
        "supersedes": str(item.get("supersedes") or "").strip(),
        "pinned": bool(item.get("pinned", False)),
        "active": bool(item.get("active", True)),
    }
    return entry


def is_relevant(entry, include_fiction: bool = False) -> bool:
    """能否进入"主人的事实检索"。"""
    if not entry.get("active", True):
        return False
    if not include_fiction and entry.get("kind") == KIND_FICTION:
        return False
    return True


# 自动替代要求"同性质":否则"主人今天喝了咖啡"(短期状态)会把
# "主人喜欢咖啡"(长期偏好)悄悄作废 —— 那是错杀,比漏替代更糟。
# 跨性质确实矛盾的(如偏好变了)由模型的显式 `取代:` 指令处理。
_AUTO_SUPERSEDE_GROUPS = (
    {KIND_PREFERENCE},
    {KIND_STATE},
    {KIND_AGREEMENT},
)


def _auto_supersede_compatible(kind_a, kind_b) -> bool:
    if kind_a == kind_b:
        return True
    return any(kind_a in group and kind_b in group for group in _AUTO_SUPERSEDE_GROUPS)


def apply_supersede(entries, new_entry):
    """把新条目并入列表,并作废同话题的旧条目。

    返回 `(新列表, 被作废的内容列表)`。

    **只把旧条目标成 inactive,不物理删除** —— 主人随时可以纠正,
    而且 `supersedes` 保留了原内容,是回退与回溯的依据。
    """
    superseded = []
    key = topic_key(new_entry.get("content", ""))
    explicit = str(new_entry.get("supersedes") or "").strip()
    result = []
    has_placeholder = False
    for entry in entries:
        if entry.get("content") == new_entry.get("content"):
            # 同内容:原位替换(刷新时间/字段),不留一条失效副本堆积在文件里。
            # 用占位符而不是直接放 new_entry:位置要对,而且此时 `supersedes`
            # 可能还没定下来;末尾统一回填。
            result.append(None)
            has_placeholder = True
            continue
        target = False
        if not entry.get("active", True):
            target = False
        elif explicit and entry.get("content") == explicit:
            target = True                     # 模型显式指定的被取代条目
        elif key and entry.get("pinned"):
            target = False                   # 置顶条目不被自动替代
        elif key and _auto_supersede_compatible(entry.get("kind"), new_entry.get("kind")) \
                and topic_key(entry.get("content", "")) == key:
            target = True                     # 同话题 + 同性质 → 启发式替代
        if target:
            superseded.append(entry.get("content"))
            replaced = dict(entry)
            replaced["active"] = False
            replaced["updated"] = new_entry.get("updated") or entry.get("updated")
            result.append(replaced)
            continue
        result.append(entry)
    if superseded and not new_entry.get("supersedes"):
        new_entry = dict(new_entry)
        new_entry["supersedes"] = superseded[0]
    if has_placeholder:
        return [new_entry if e is None else e for e in result], superseded
    result.append(new_entry)
    return result, superseded


def select_entries(entries, max_chars: int, kinds=None, include_inactive: bool = False,
                   include_fiction: bool = False):
    """按字符预算挑选记忆(两种模式共用)。

    规则:
    - 跳过失效条目;默认也跳过虚构(除非调用方明确要角色背景);
    - `kinds` 非空时只取这些类型(用于"分别管理");
    - **置顶条目优先且必留**(除非自身超预算);
    - 其余从最近向前填充。
    """
    candidates = []
    for position, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            entry = {"content": str(entry or "")}      # 兼容纯字符串形式的记忆
        if not include_inactive and not entry.get("active", True):
            continue
        if not include_fiction and entry.get("kind") == KIND_FICTION:
            continue
        if kinds and entry.get("kind") not in kinds:
            continue
        candidates.append((position, entry))

    picked = []
    used = 0

    def take(pair):
        nonlocal used
        _position, entry = pair      # 位置只用于排序,取用时不需要
        text = str(entry.get("content", "") or "").strip()
        if not text or used + len(text) > max_chars:
            # 超长条目整条丢弃,而不是塞进去把预算撑爆(与既有语义一致)
            return
        picked.append(pair)
        used += len(text)

    pinned = [pair for pair in candidates if pair[1].get("pinned")]
    rest = [pair for pair in candidates if not pair[1].get("pinned")]
    for pair in pinned:
        take(pair)
    for pair in reversed(rest):        # 从最近向前填充
        take(pair)
    # 展示顺序:置顶在前,其余按**时间先后**(旧→新)排,读起来才自然
    picked.sort(key=lambda pair: (0 if pair[1].get("pinned") else 1, pair[0]))
    return [entry for _position, entry in picked]


def memory_stats(entries) -> dict:
    """查看:按类型/来源计数与有效条目数(供 UI 与自检使用)。"""
    stats = {"total": 0, "active": 0, "inactive": 0, "pinned": 0,
             "by_kind": {}, "by_source": {}}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        stats["total"] += 1
        if entry.get("active", True):
            stats["active"] += 1
        else:
            stats["inactive"] += 1
        if entry.get("pinned"):
            stats["pinned"] += 1
        kind = entry.get("kind") or KIND_FACT
        stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
        source = entry.get("source") or SOURCE_LEGACY
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
    return stats


def parse_supersede_directive(text: str):
    """从形如 `取代: 主人喜欢咖啡` 的指令里取被取代的内容。"""
    match = re.match(r"^\s*(?:取代|替代|supersedes)\s*[:：]\s*(.+)$", str(text or ""))
    return match.group(1).strip() if match else None
