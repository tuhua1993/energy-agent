"""文档切片 + BM25 关键词检索 + 可选向量检索 + RRF 混合召回。

零依赖实现：不配 API Key 时用纯本地 BM25，保证任何环境都能跑通全链路。
"""
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

from . import config

_WS = re.compile(r"\s+")


# ---------------- 分词（无词典方案：中文 bigram + 英文数字整词） ----------------
def tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            # 字级 bigram：解决无词典分词，且天然支持部分匹配
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


# ---------------- 递归切片 ----------------
def split_text(text: str, size: int, overlap: int):
    """按 段落 -> 句子 -> 硬切 三级递归切分，保留 overlap 防止语义被切断。"""
    blocks = [b.strip() for b in re.split(r"\n\s*\n|\n", text) if b.strip()]
    pieces = []
    for b in blocks:
        if len(b) <= size:
            pieces.append(b)
            continue
        # 超长块按句切
        sents = re.split(r"(?<=[。；！？;!?])", b)
        buf = ""
        for s in sents:
            if len(buf) + len(s) <= size:
                buf += s
            else:
                if buf:
                    pieces.append(buf)
                buf = s
        if buf:
            pieces.append(buf)
    # 最后一道：仍然过长的硬切
    final = []
    for p in pieces:
        while len(p) > size:
            final.append(p[:size])
            p = p[size:]
        if p:
            final.append(p)
    # 拼 overlap：重叠文本只用于检索索引，展示给用户的仍是干净原文
    if overlap <= 0 or len(final) <= 1:
        return final, list(final)
    out = []
    for i, p in enumerate(final):
        out.append(final[i - 1][-overlap:] + p if i > 0 else p)
    return final, out


class Chunk:
    __slots__ = ("cid", "doc", "text", "tokens", "tf", "vec")

    def __init__(self, cid, doc, text, search_text=None):
        self.cid = cid
        self.doc = doc
        self.text = text
        self.tokens = tokenize(search_text or text)
        self.tf = {}
        for t in self.tokens:
            self.tf[t] = self.tf.get(t, 0) + 1
        self.vec = None


class BM25:
    def __init__(self, chunks):
        self.chunks = chunks
        self.n = len(chunks)
        self.avgdl = sum(len(c.tokens) for c in chunks) / max(1, self.n)
        self.df = {}
        for c in chunks:
            for t in set(c.tokens):
                self.df[t] = self.df.get(t, 0) + 1
        # 预计算 IDF
        self.idf = {}
        for t, f in self.df.items():
            self.idf[t] = math.log(1 + (self.n - f + 0.5) / (f + 0.5))

    def scores(self, query_tokens):
        out = {}
        k1, b = config.BM25_K1, config.BM25_B
        for c in self.chunks:
            dl = len(c.tokens)
            s = 0.0
            for t in query_tokens:
                freq = c.tf.get(t, 0)
                if not freq:
                    continue
                idf = self.idf.get(t)
                if idf is None:
                    continue
                s += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / max(1e-9, self.avgdl)))
            if s > 0:
                out[c.cid] = s
        return out


def _embed(texts):
    """调用 OpenAI 兼容 /embeddings 接口；未配置则返回 None。"""
    if not (config.EMBED_MODEL and config.EMBED_API_KEY):
        return None
    url = config.EMBED_BASE_URL.rstrip("/") + "/embeddings"
    payload = {"model": config.EMBED_MODEL, "input": texts}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + config.EMBED_API_KEY},
    )
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [d["embedding"] for d in data["data"]]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


class Retriever:
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.chunks = []
        self._build()

    def _build(self):
        files = sorted(self.docs_dir.glob("*.md"))
        for f in files:
            text = f.read_text(encoding="utf-8")
            cleans, with_overlap = split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for i, piece in enumerate(cleans):
                cid = f"{f.stem}#{i}"
                self.chunks.append(Chunk(cid, f.stem, piece, with_overlap[i]))
        self.bm25 = BM25(self.chunks)
        # 有 embedding 配置时预计算索引向量
        if config.EMBED_MODEL and config.EMBED_API_KEY:
            try:
                vecs = _embed([c.text for c in self.chunks])
                if vecs:
                    for c, v in zip(self.chunks, vecs):
                        c.vec = v
            except Exception as e:
                print("[rag] 向量索引构建失败，降级为纯 BM25：", e)

    def _vector_scores(self, query):
        if not any(c.vec for c in self.chunks):
            return {}
        try:
            qv = _embed([query])
        except Exception:
            return {}
        if not qv:
            return {}
        qv = qv[0]
        return {c.cid: _cosine(qv, c.vec) for c in self.chunks if c.vec}

    def search(self, query, k=None):
        k = k or config.TOP_K
        qt = tokenize(query)
        bm = self.bm25.scores(qt)
        vec = self._vector_scores(query) if config.HYBRID else {}

        if vec:
            # RRF 融合：两路各自排名，按 1/(60+rank) 加权，规避分数量纲不一致
            fused = {}
            for rank, (cid, _) in enumerate(sorted(bm.items(), key=lambda x: -x[1])):
                fused[cid] = fused.get(cid, 0) + 1.0 / (60 + rank)
            for rank, (cid, _) in enumerate(sorted(vec.items(), key=lambda x: -x[1])):
                fused[cid] = fused.get(cid, 0) + 1.0 / (60 + rank)
            ranked = sorted(fused.items(), key=lambda x: -x[1])
        else:
            ranked = sorted(bm.items(), key=lambda x: -x[1])

        by_id = {c.cid: c for c in self.chunks}
        hits = []
        for cid, score in ranked[:k]:
            c = by_id[cid]
            hits.append({
                "id": cid,
                "doc": c.doc,
                "text": c.text,
                "score": round(float(score), 4),
                "bm25": round(float(bm.get(cid, 0)), 3),
                "vec": round(float(vec.get(cid, 0)), 3) if vec else None,
            })
        return hits


def hash_text(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()[:8]
