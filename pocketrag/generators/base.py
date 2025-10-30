from __future__ import annotations
from typing import Protocol

class Generator(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.2) -> str:
        ...