from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def _normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / norms

@dataclass
class FaissArtifacts:
    index_path: Path
    meta_path: Path
    docstore_path: Path

def build_faiss(corpus: List[Dict], out_dir: Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FaissArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [(d["id"], (d.get("title", "") + " " + d.get("text", "")).strip()) for d in corpus]
    ids, contents = zip(*texts)
    model = SentenceTransformer(model_name)
    embs = model.encode(list(contents), batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype(np.float32))
    faiss.write_index(index, str(out_dir / "faiss.index"))

    (out_dir / "meta.json").write_text(json.dumps({"model": model_name, "dim": d, "doc_ids": list(ids)}, indent=2))

    with (out_dir / "docstore.jsonl").open("w") as f:
        for dct in corpus:
            f.write(json.dumps({"id": dct["id"], "title": dct.get("title", ""), "text": dct.get("text", "")}) + "\n")
    return FaissArtifacts(out_dir / "faiss.index", out_dir / "meta.json", out_dir / "docstore.jsonl")

def search_faiss(index_dir: Path, query: str, k: int = 5, model_name: str | None = None) -> List[Tuple[str, float]]:
    meta = json.loads((index_dir / "meta.json").read_text())
    if model_name is None:
        model_name = meta["model"]
    ids = meta["doc_ids"]
    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    scores, idxs = index.search(q, k)
    hits = []
    for i, s in zip(idxs[0], scores[0]):
        if i == -1:
            continue
        hits.append((ids[i], float(s)))
    return hits