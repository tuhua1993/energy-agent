"""多轮对话记忆：滑动窗口 + 字符预算。

不做无限拼接：历史越长，token 越贵、越容易冲淡当前问题。
工程上的取舍是"保留最近 N 轮 + 硬预算兜底"，需要长期记忆时再引入摘要或向量记忆。
"""
from . import config


class Memory:
    def __init__(self, max_turns=None, max_chars=None):
        self.max_turns = max_turns or config.MAX_HISTORY_TURNS
        self.max_chars = max_chars or config.MAX_CONTEXT_CHARS
        self.history = []  # [{"role":"user"/"assistant","content":...}]

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        # 按轮数裁剪（一问一答算两轮）
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get(self):
        """按字符预算从最新往回取，超预算直接丢弃更早的内容。"""
        out, total = [], 0
        for m in reversed(self.history):
            c = len(m["content"])
            if total + c > self.max_chars:
                break
            out.append(m)
            total += c
        return list(reversed(out))

    def clear(self):
        self.history = []

    def turns(self):
        return len(self.history) // 2
