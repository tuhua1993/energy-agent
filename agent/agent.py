"""Agent 主循环：检索 -> 工具决策 -> 工具执行 -> 结果回灌 -> 流式生成。

采用 ReAct 思路：先拿到依据（检索/工具），再让模型基于依据作答，
而不是让模型凭记忆直接回答。所有对外输出以事件流形式抛出，便于前端展示"思考链路"。
"""
from . import config, llm, memory
from . import tools as toolmod
from .rag import Retriever

SYSTEM_TEMPLATE = """你是能源电力行业（光伏、储能、虚拟电厂）的智能运维助手，服务于现场运维与调度人员。

回答规范：
1. 只能依据下方知识片段中的内容作答，不得凭自身记忆编造参数、阈值与流程。
2. 使用知识片段内容时，必须在句末标注引用编号，如 [1]、[2]。
3. 知识片段为空或明显不相关时，直接回答"知识库中未检索到相关内容，不做推测"，并说明需要补充什么信息。
4. 涉及实时运行数据（设备状态、发电量、告警列表）时，必须调用对应工具查询，不得臆造数值。
5. 回答要给出可执行的处置步骤，用编号列表组织，不要泛泛而谈。
6. 只回答能源电力运维相关问题，其他问题礼貌拒答。

<<<KB>>>
{context}
<<</KB>>>
"""


def _build_context(hits):
    if not hits:
        return "（本次未检索到相关内容）"
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] 来源：{h['doc']}\n{h['text']}")
    return "\n---\n".join(parts)


class EnergyAgent:
    def __init__(self, docs_dir: str):
        self.retriever = Retriever(docs_dir)
        self.mem = memory.Memory()

    def reset(self):
        self.mem.clear()

    def chat(self, user_input: str):
        """产出事件流：retrieval / tool_call / tool_result / delta / citations / done"""

        # ---------- 1. 检索（RAG 召回） ----------
        hits = self.retriever.search(user_input)
        top_bm25 = hits[0]["bm25"] if hits else 0.0
        relevant = top_bm25 >= config.RELEVANCE_THRESHOLD
        yield {"type": "retrieval", "hits": hits, "relevant": relevant,
               "mode": "hybrid" if hits and hits[0]["vec"] is not None else "bm25"}

        ctx = _build_context(hits if relevant else [])
        messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=ctx)}]
        messages += self.mem.get()
        messages += [{"role": "user", "content": user_input}]

        # ---------- 2. 工具调用循环 ----------
        used_tools = []
        if config.ENABLE_TOOLS:
            for _ in range(config.TOOL_MAX_ROUNDS):
                resp = llm.chat(messages, toolmod.TOOLS)
                calls = resp.get("tool_calls") or []
                if not calls:
                    break
                for tc in calls:
                    fn = tc["function"]["name"]
                    args = tc["function"].get("arguments", "{}")
                    yield {"type": "tool_call", "name": fn, "args": args}
                    result = toolmod.execute(fn, args)
                    yield {"type": "tool_result", "name": fn, "result": result}
                    used_tools.append(fn)
                    messages.append({"role": "assistant", "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", "call_1"),
                                     "name": fn, "content": result})

        # ---------- 3. 流式生成最终答案 ----------
        buf = []
        for piece in llm.stream(messages):
            buf.append(piece)
            yield {"type": "delta", "text": piece}
        answer = "".join(buf)

        # ---------- 4. 写入记忆 + 回抛引用 ----------
        self.mem.add("user", user_input)
        self.mem.add("assistant", answer)

        cites = [{"id": h["id"], "doc": h["doc"], "text": h["text"][:200],
                  "score": h["score"]} for h in (hits if relevant else [])]
        yield {"type": "citations", "items": cites, "tools": used_tools}
        yield {"type": "done", "answer": answer}
