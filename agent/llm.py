"""LLM 客户端：OpenAI 兼容接口（流式 + 工具调用），无 Key 时自动降级为 Mock。

Mock 模式的价值：没有 API Key 也能完整演示"检索 -> 工具调用 -> 流式生成"全链路，
方便本地开发与面试演示；配置 Key 后自动切换到真模型，代码路径完全一致。
"""
import json
import time
import urllib.request

from . import config


def _headers():
    return {"Content-Type": "application/json",
            "Authorization": "Bearer " + config.LLM_API_KEY}


def _url():
    return config.LLM_BASE_URL.rstrip("/") + "/chat/completions"


def available():
    return bool(config.LLM_API_KEY)


# ------------------------------------------------------------------
# 非流式：用于判断是否需要调用工具（工具决策必须拿完整结构，流式解析成本高）
# ------------------------------------------------------------------
def chat(messages, tools=None, temperature=0.2):
    if not available():
        return _mock_chat(messages, tools)
    payload = {"model": config.LLM_MODEL, "messages": messages,
               "temperature": temperature, "stream": False}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    req = urllib.request.Request(_url(), data=json.dumps(payload).encode("utf-8"),
                                 headers=_headers())
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    return {"content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or []}


# ------------------------------------------------------------------
# 流式：最终答案逐 token 输出
# ------------------------------------------------------------------
def stream(messages, temperature=0.2):
    if not available():
        for piece in _mock_stream(messages):
            yield piece
        return
    payload = {"model": config.LLM_MODEL, "messages": messages,
               "temperature": temperature, "stream": True}
    req = urllib.request.Request(_url(), data=json.dumps(payload).encode("utf-8"),
                                 headers=_headers())
    with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                delta = json.loads(body)["choices"][0]["delta"]
            except Exception:
                continue
            if delta.get("content"):
                yield delta["content"]


# ------------------------------------------------------------------
# Mock：基于检索到的语料，组织出一段有依据、带引用的回答
# ------------------------------------------------------------------
def _last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _context_of(messages):
    for m in reversed(messages):
        if m.get("role") == "system" and "<<<KB>>>" in (m.get("content") or ""):
            return m["content"]
    return ""


def _mock_chat(messages, tools):
    """模拟工具决策：命中关键词就返回 tool_call；已执行过工具则直接进入作答阶段。"""
    q = _last_user(messages)
    if any(m.get("role") == "tool" for m in messages):
        return {"content": "", "tool_calls": []}
    if tools:
        if any(w in q for w in ("发电量", "发电", "电量")):
            return {"content": "",
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "query_generation",
                                                 "arguments": json.dumps({"station_id": "PV-JH-01",
                                                                          "date": _pick_date(q)},
                                                                         ensure_ascii=False)}}]}
        if any(w in q for w in ("告警", "报警", "告警列表")):
            return {"content": "",
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "query_alarms",
                                                 "arguments": json.dumps({"station_id": "PV-JH-01",
                                                                          "level": "all"},
                                                                         ensure_ascii=False)}}]}
        if any(w in q for w in ("状态", "正常吗", "运行吗", "设备")):
            return {"content": "",
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "query_device_status",
                                                 "arguments": json.dumps({"device_id": _pick_device(q)},
                                                                         ensure_ascii=False)}}]}
    return {"content": "", "tool_calls": []}


def _pick_date(q):
    import re as _re
    m = _re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", q)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return time.strftime("%Y-%m-%d")


def _pick_device(q):
    for d in ("逆变器", "储能", "汇流箱", "PCS", "组件"):
        if d in q:
            return {"逆变器": "INV-02", "储能": "ESS-01", "汇流箱": "CB-03",
                    "PCS": "PCS-01", "组件": "STR-05"}[d]
    return "INV-02"


def _is_item(s):
    if not s or s.startswith("[") or s.startswith("#") or "来源：" in s:
        return False
    if s.startswith("-") or s.startswith("•"):
        return True
    return len(s) > 2 and s[0].isdigit() and s[1] in ".、．"


def _mock_stream(messages):
    """把检索到的片段组织成"有依据、有引用、有结构"的答案，逐字吐出模拟流式。"""
    blocks, doc_names = [], []
    ctx = _context_of(messages)
    if "<<<KB>>>" in ctx:
        seg = ctx.split("<<<KB>>>")[1].split("<<</KB>>>")[0]
        for b in seg.split("\n---\n"):
            b = b.strip()
            if not b:
                continue
            head, _, body = b.partition("\n")
            if head.startswith("["):
                doc_names.append(head.split("来源：")[-1].strip())
                blocks.append(body)
            else:
                blocks.append(b)

    tool_blob = ""
    for m in messages:
        if m.get("role") == "tool":
            tool_blob += m.get("content", "") + "\n"

    items, seen = [], set()
    for b in blocks[:4]:
        for ln in b.split("\n"):
            s = ln.strip()
            if _is_item(s) and s not in seen:
                seen.add(s)
                items.append(s.lstrip("-•").strip())

    doc = doc_names[0] if doc_names else "运维规程"
    no_kb = (not blocks) or all(b.startswith("（本次未检索到") for b in blocks)
    parts = []
    if tool_blob.strip():
        parts.append("实时系统查询结果：\n" + tool_blob.strip() + "\n\n")
    if items:
        parts.append(f"依据《{doc}》[1]，处置要点如下：\n")
        for i, it in enumerate(items[:6], 1):
            parts.append(f"{i}. {it}\n")
        parts.append("\n以上结论来自站内运维规程[1]，现场处置以实际工况与最新规程为准。")
    elif not no_kb:
        # 命中语料但片段不是条目式结构，直接引用正文，不做二次加工
        raw = "\n".join(b.strip() for b in blocks[:2])
        body = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("#"))
        parts.append(f"依据《{doc}》[1]：\n{body[:300]}\n"
                     "\n以上结论来自站内运维规程[1]，现场处置以实际工况与最新规程为准。")
    elif tool_blob.strip():
        parts.append("该问题未命中规程条文，以上为实时数据查询结果，未做延伸推断。")
    else:
        parts.append("知识库中未检索到与该问题相关的内容，不做推测。\n"
                     "建议补充：设备编号、故障代码或对应规程章节，以便定位条文。")

    text = "".join(parts)
    for ch in text:
        time.sleep(0.003)
        yield ch
