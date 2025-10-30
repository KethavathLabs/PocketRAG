from __future__ import annotations
from typing import Dict, List, Tuple
import math

# qrels: {qid: {doc_id: rel_grade(int)}}, hits: list[(doc_id, score)] (ranked)

def recall_at_k(qrels: Dict[str, int], hits: List[Tuple[str, float]], k: int) -> float:
    if not qrels:
        return 0.0
    topk_ids = {d for d, _ in hits[:k]}
    rel = set([d for d, g in qrels.items() if g > 0])
    return len(rel & topk_ids) / max(1, len(rel))

def mrr(qrels: Dict[str, int], hits: List[Tuple[str, float]], k: int) -> float:
    rel = set([d for d, g in qrels.items() if g > 0])
    for i, (d, _) in enumerate(hits[:k], start=1):
        if d in rel:
            return 1.0 / i
    return 0.0

def dcg_at_k(qrels: Dict[str, int], hits: List[Tuple[str, float]], k: int) -> float:
    dcg = 0.0
    for i, (d, _) in enumerate(hits[:k], start=1):
        rel = qrels.get(d, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def ndcg_at_k(qrels: Dict[str, int], hits: List[Tuple[str, float]], k: int) -> float:
    dcg = dcg_at_k(qrels, hits, k)
    # ideal DCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        if rel > 0:
            idcg += (2**rel - 1) / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0

def aggregate(qrels_map: Dict[str, Dict[str, int]],
              results_map: Dict[str, List[Tuple[str, float]]],
              ks = (1, 3, 5, 10)) -> Dict[str, float]:
    out = {}
    for k in ks:
        r = []; rr = []; nd = []
        for qid, qrels in qrels_map.items():
            hits = results_map.get(qid, [])
            r.append(recall_at_k(qrels, hits, k))
            rr.append(mrr(qrels, hits, k))
            nd.append(ndcg_at_k(qrels, hits, k))
        out[f"Recall@{k}"] = sum(r)/len(r) if r else 0.0
        out[f"MRR@{k}"] = sum(rr)/len(rr) if rr else 0.0
        out[f"nDCG@{k}"] = sum(nd)/len(nd) if nd else 0.0
    return out