# pocketrag/eval/qa.py
from __future__ import annotations
from typing import Dict, List, Tuple
from ..utils.text import best_em_f1, split_sentences, content_tokens

def grounded_precision(answer: str, retrieved_texts: List[str], threshold: float = 0.6) -> float:
    """
    Fraction of answer sentences supported by at least one retrieved chunk,
    where support = token_overlap(answer_sentence, chunk) >= threshold.
    """
    if not answer.strip(): return 0.0
    sents = split_sentences(answer)
    if not sents: return 0.0

    def support(sent: str) -> bool:
        stoks = content_tokens(sent)
        if not stoks: return False
        needed = max(1, int(len(stoks) * threshold))
        for ctx in retrieved_texts:
            ctoks = content_tokens(ctx)
            overlap = len(set(stoks) & set(ctoks))
            if overlap >= needed:
                return True
        return False

    supported = sum(1 for s in sents if support(s))
    return supported / max(1, len(sents))

def score_example(pred: str, refs: List[str], retrieved_texts: List[str]) -> Dict[str, float]:
    em, f1 = best_em_f1(pred, refs) if refs else (0.0, 0.0)
    gp = grounded_precision(pred, retrieved_texts)
    return {"EM": em, "F1": f1, "grounded_precision": gp}