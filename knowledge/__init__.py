# -*- coding: utf-8 -*-
"""时崎狂三 · RAG 知识库（轻量关键词检索，零额外显存）

**本目录 = 知识库**：知识条目在同目录的 `knowledge.txt`（主人可以直接编辑/扩充），
本文件只是读它、切词、按相关度检索，再把命中的设定注入对话上下文。
导入名仍是 `knowledge`（`from knowledge import build_rag_context`），所以放进子目录
之后调用方一行都不用改。

检索策略：
- 先把提问按非文字字符切成若干"段"，并把停用词当作**段边界**，再在每段内部切 1~4 字词组；
  词组越长权重越高（4 字=16、3 字=9、2 字=4、1 字=1）。
  （停用词必须先当段边界、再在段内组词：若先删掉停用词字符再切词组，
   分处被删字符两侧的字会拼成假词组，例如「主人的妹妹」切出「人妹」，凭空命中设定条目。）
- 命中分数低于阈值（MIN_SCORE）视为无关，避免"今天心情不好"这类闲聊误触设定；
- 分数降序；同分时较短的事实（更聚焦）优先；同分同长保持知识库文件里的先后顺序
  （用 ``sorted(key=..., reverse=True)`` 的稳定性保证）。
"""
import os
import re

KNOWLEDGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.txt")

MIN_SCORE = 3          # 最低相关度：至少命中一个 2 字词（权重4）或 3 个单字
_MAX_NGRAM = 4         # 词组最大长度

_STOPWORDS = set("的了呢吗啊呀么吧你我他她它是很都就也不这那个有在什么怎么如何请问一下吗呢吧啊哈哦嗯")

_WORD_RE = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9]+")

# ---------- 行内元数据 ----------
# 行尾方括号元数据：`事实。[标签: 能力|天使][别名: 刻帝|Zafkiel][出处: 原作]`
# 只在**行尾**识别，正文里的方括号不受影响。
_META_BLOCK_RE = re.compile(r"\s*\[([^\[\]]+)\]\s*$")
_META_KEYS = {
    "标签": "tags", "tags": "tags", "tag": "tags",
    "别名": "aliases", "alias": "aliases", "aliases": "aliases", "又称": "aliases",
    "出处": "source", "source": "source", "来源": "source",
}
_META_SPLIT_RE = re.compile(r"[|｜,，、/]")
_MAX_META_BLOCKS = 16      # 单行元数据块数量上界(防死循环,见 parse_fact)

# 缓存：{(路径, mtime_ns, size): [事实dict...]}，检索路径不再每次读盘。
_cache_key = None
_cache_facts = []


def parse_fact(line):
    """解析一行事实 → (content, meta)。

    元数据块从**行尾**逐个剥出（支持多个相邻块）。无法识别的块**按正文保留**：
    宁可让正文里多一段方括号，也不能悄悄丢掉主人的设定。
    """
    text = str(line or "").strip()
    meta = {"tags": [], "aliases": [], "source": ""}
    # 循环次数上界:正常最多几个元数据块;这是"循环必定前进"的防御 ——
    # 若将来某个改动让正则匹配却不消费文本,`while True` 会**死循环**,
    # 把整个进程挂住,只能强杀。
    for _ in range(_MAX_META_BLOCKS):
        match = _META_BLOCK_RE.search(text)
        if not match:
            break
        body = match.group(1).strip()
        parts = re.split(r"[:：]", body, maxsplit=1)
        if len(parts) != 2:
            break                      # 不是元数据（例如正文里的 [注]）→ 按正文保留
        key, value = parts[0].strip(), parts[1].strip()
        field = _META_KEYS.get(key.lower()) or _META_KEYS.get(key)
        if field is None:
            break                      # 不认识的块按正文保留,不丢事实
        items = [v.strip() for v in _META_SPLIT_RE.split(value) if v.strip()]
        if field == "source":
            meta["source"] = items[0] if items else ""
        else:
            # 块是从**行尾往前**剥出的,所以整块插到前面:块与块之间保持文件顺序,
            # 块内部保持书写顺序(直接 reverse 整个列表会把块内顺序也翻掉)。
            merged = []
            for item in items:
                # 块内与块间都要去重(手写时写重复很常见)
                if item not in merged and item not in meta[field]:
                    merged.append(item)
            meta[field] = merged + meta[field]
        text = text[:match.start()].rstrip()   # 元数据已取出,正文里不该再留着它
    return text, meta


def _read_facts(path):
    """真正读盘（单独抽出来，便于单独验证缓存是否命中）。

    编码做回退（与 kurumi_memory 同策略）：主人把 knowledge.txt 另存成 GBK 或带 BOM 时，
    不能让 RAG 抛 UnicodeDecodeError —— 那会顺着 build_rag_context 炸掉整个发送路径，
    实机表现是界面永久停在「正在生成」、按钮全灰。

    返回**结构化事实**（content/tags/aliases/source）。
    """
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"知识库编码无法识别（既非 UTF-8 也非 GBK）：{path}")
    facts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        content, meta = parse_fact(line)
        if not content:
            continue
        facts.append({"content": content, "tags": meta["tags"],
                      "aliases": meta["aliases"], "source": meta["source"]})
    return facts


def _file_key(path):
    """文件指纹：mtime_ns + 大小，任一变化即视为知识库已更新。"""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)

def _copy_fact(fact) -> dict:
    """拷贝一条事实(连列表一起复制)。

    只做 `dict(f)` 是**浅拷贝**:`aliases`/`tags` 仍指向缓存里的同一列表,
    调用方 append 一下就把缓存污染了。
    """
    return {
        "content": fact.get("content", ""),
        "tags": list(fact.get("tags") or []),
        "aliases": list(fact.get("aliases") or []),
        "source": fact.get("source", ""),
    }

def load_facts(force_reload=False):
    """读取结构化事实（带 mtime+size 失效的缓存）。返回列表副本。"""
    global _cache_key, _cache_facts
    key = _file_key(KNOWLEDGE_FILE)
    if key is None:            # 文件不存在：清缓存，返回空
        _cache_key, _cache_facts = None, []
        return []
    if not force_reload and key == _cache_key:
        return [_copy_fact(f) for f in _cache_facts]
    try:
        facts = _read_facts(KNOWLEDGE_FILE)
    except (OSError, ValueError):
        # 读盘/解码失败**不能写缓存**：指纹没变，写进去就等于把"这次读失败"固定成
        # 永久结论 —— 故障恢复后仍返回 []，RAG 静默失效到进程重启。
        return []
    _cache_key, _cache_facts = key, [_copy_fact(f) for f in facts]
    return [_copy_fact(f) for f in _cache_facts]


def _segments(query):
    """把提问切成词组候选段：非文字字符与停用词都作为段边界，绝不跨边界组词。"""
    segments = []
    for chunk in _WORD_RE.findall(str(query or "")):
        current = []
        for ch in chunk:
            if ch in _STOPWORDS:
                if current:
                    segments.append("".join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            segments.append("".join(current))
    return segments


def _weighted_keywords(query):
    """返回 {词组: 权重}。词组越长权重越高（长词命中=高度相关）。"""
    kws = {}
    for seg in _segments(query):
        for n in range(1, _MAX_NGRAM + 1):
            w = n * n   # 1→1, 2→4, 3→9, 4→16
            for i in range(len(seg) - n + 1):
                gram = seg[i:i + n]
                # 同一个词组只保留最大权重
                if gram not in kws or w > kws[gram]:
                    kws[gram] = w
    return kws


def _search_text(fact) -> str:
    """用于匹配的文本：正文 + 别名。

    别名与正文**同权**：主人用「刻帝」提问时命中分数应与用「刻刻帝」一致 ——
    否则换个说法就检索不到，而这是"要不要上语义检索"里最先该用廉价手段解决的问题。
    """
    parts = [str(fact.get("content", ""))]
    parts.extend(str(a) for a in fact.get("aliases") or [])
    return " ".join(parts)


def retrieve_facts(query, top_k=3, tags=None):
    """按相关度检索**结构化事实**（供注入设定时带上出处）。

    `tags` 非空时只在该标签内检索（例如只看「能力」）。
    """
    facts = load_facts()
    if not facts or not query:
        return []
    if tags:
        wanted = {str(t).strip() for t in tags if str(t).strip()}
        facts = [f for f in facts if wanted & set(f.get("tags") or [])]
        if not facts:
            return []
    kws = _weighted_keywords(query)
    scored = []
    for fact in facts:
        text = _search_text(fact)
        matched = [kw for kw in kws if kw in text]
        if not matched:
            continue
        s = sum(kws[kw] for kw in matched)
        if s < MIN_SCORE:
            continue
        # 同分时的取舍顺序:
        # 1) 命中位置越靠前越优先 —— 概念问句(「天使指的是什么」)的正解通常是
        #    **以该词开头**的那条,只按"更短优先"会被带到另一条含同词的短事实上;
        # 2) 再比长度（短者更聚焦）；3) 仍相同则按文件顺序（sort 稳定性保证）。
        pos = min(text.find(kw) for kw in matched)
        scored.append((s, -pos, -len(fact.get("content", "")), fact))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [item[3] for item in scored[:top_k]]


def build_rag_context(query, top_k=3, tags=None, with_source=True):
    """组装注入给模型的设定参考。

    `with_source=True` 时每条附上「出处」,让模型能区分**原作设定**与**主人补充的习惯**。
    """
    facts = retrieve_facts(query, top_k=top_k, tags=tags)
    if not facts:
        return ""
    lines = []
    for fact in facts:
        source = str(fact.get("source") or "").strip()
        suffix = f"（出处：{source}）" if (with_source and source) else ""
        lines.append("- " + fact["content"] + suffix)
    return ("\n【设定参考（回答相关问题时自然融入，不要逐条机械复述）】\n"
            + "\n".join(lines) + "\n")
