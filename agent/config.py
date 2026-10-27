"""全局配置。

没有 API Key 也能完整跑通：LLM 会走 Mock 模式，检索会走本地 BM25。
有 Key 时自动切换到真模型 + 混合检索。
"""
import os

# ============ LLM（OpenAI 兼容接口）============
# 支持 DeepSeek / 通义千问 / 智谱 / Kimi / 本地 vLLM / OpenAI 等一切 OpenAI 兼容端点
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = 60

# ============ Embedding（可选，不配则只走 BM25）============
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "") or LLM_API_KEY
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "") or LLM_BASE_URL
EMBED_MODEL = os.getenv("EMBED_MODEL", "")

# ============ 检索参数 ============
CHUNK_SIZE = 400          # 单片段字符数
CHUNK_OVERLAP = 80        # 相邻片段重叠字符数，防止语义被切断
TOP_K = 4                 # 最终进入上下文的片段数
RETRIEVE_K = 20           # 粗召数量（参与融合的候选）
BM25_K1 = 1.5             # BM25 词频饱和参数
BM25_B = 0.75             # BM25 长度归一化参数
HYBRID = True             # 配置了 embedding 时启用 BM25 + 向量 混合召回

# ============ 工具 ============
ENABLE_TOOLS = True
TOOL_MAX_ROUNDS = 3       # 工具调用最大轮数，防止死循环

# ============ 记忆 ============
MAX_HISTORY_TURNS = 6     # 保留最近 N 轮对话
MAX_CONTEXT_CHARS = 6000  # 上下文字符预算，超出后丢弃更早的历史

# ============ 服务 ============
HOST = "127.0.0.1"
PORT = 8848

# ============ 幻觉控制 ============
# 以 BM25 原始分为判定口径：最高分低于该阈值时判定为"语料中没有依据"，
# 直接回答"未检索到相关内容"，从源头掐掉编造，而不是事后靠 prompt 约束
RELEVANCE_THRESHOLD = 8.0
REQUIRE_CITATION = True   # 要求回答标注引用编号
