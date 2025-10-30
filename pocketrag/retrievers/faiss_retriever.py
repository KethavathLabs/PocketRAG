from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(self, index_dir: Path, model_name: str | None = None):
        self.index_dir = index_dir
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        meta = json.loads((index_dir / "meta.json").read_text())
        self.doc_ids = meta["doc_ids"]
        self.model_name = model_name or meta["model"]
        self.model = SentenceTransformer(self.model_name)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q, k)
        hits = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            hits.append((self.doc_ids[i], float(s)))
        return hits