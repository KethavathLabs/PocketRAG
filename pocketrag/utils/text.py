from __future__ import annotations
import re
from collections import Counter
from typing import List, Tuple

_ARTICLES = {"a","an","the"}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    tokens = [t for t in _WS_RE.sub(" ", s).split(" ") if t and t not in _ARTICLES]
    return " ".join(tokens)

def tokens(s: str) -> List[str]:
    s = normalize(s)
    return [t for t in s.split(" ") if t]

def f1_score(pred: str, ref: str) -> float:
    pt = Counter(tokens(pred)); rt = Counter(tokens(ref))
    overlap = sum((pt & rt).values())
    if overlap == 0: return 0.0
    p = overlap / max(1, sum(pt.values()))
    r = overlap / max(1, sum(rt.values()))
    return 2 * p * r / (p + r)

def em_score(pred: str, ref: str) -> float:
    return 1.0 if normalize(pred) == normalize(ref) else 0.0

def best_em_f1(pred: str, refs: List[str]) -> Tuple[float,float]:
    if not refs: return 0.0, 0.0
    em = max(em_score(pred, r) for r in refs)
    f1 = max(f1_score(pred, r) for r in refs)
    return float(em), float(f1)

def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return parts

_STOP = {"a","an","the","and","or","of","to","in","on","for","with","by","is","are","was","were","it","that","this","as","at","from","be"}
def content_tokens(s: str) -> List[str]:
    return [t for t in tokens(s) if t not in _STOP]