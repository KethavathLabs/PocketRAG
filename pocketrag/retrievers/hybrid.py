from __future__ import annotations
from typing import List, Tuple, Dict
from .bm25_retriever import BM25Retriever
from .faiss_retriever import FaissRetriever

class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, faiss: FaissRetriever, k0: int = 60):
        self.bm25 = bm25
        self.faiss = faiss
        self.k0 = k0  # standard RRF constant

    def _rrf(self, ranked_lists: List[List[str]]) -> Dict[str, float]:
        # ranked_lists: each is [doc_id in rank order]
        scores: Dict[str, float] = {}
        for lst in ranked_lists:
            for rank, doc_id in enumerate(lst, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k0 + rank)
        return scores

    def search(self, query: str, k: int = 10, per_method_k: int = 100) -> List[Tuple[str, float]]:
        bm = self.bm25.search(query, k=per_method_k)
        fa = self.faiss.search(query, k=per_method_k)
        bm_ids = [d for d, _ in bm]
        fa_ids = [d for d, _ in fa]
        fused = self._rrf([bm_ids, fa_ids])
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked  # (doc_id, fused_score)