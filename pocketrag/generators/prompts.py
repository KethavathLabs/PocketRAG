from __future__ import annotations

SYS = """You are a helpful assistant. Answer ONLY using the provided context.
If the context does not contain the answer, say: "I don't know." Keep answers concise.
"""

def build_prompt(question: str, context: str) -> str:
    return f"{SYS}\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"