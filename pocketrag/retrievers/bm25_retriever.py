from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from ..index.bm25 import BM25Index, _tokenize
import numpy as np

class BM25Retriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.idx = BM25Index.load(index_dir)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        scores = self.idx.bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(self.idx.doc_ids[i], float(scores[i])) for i in order]