from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import re, pickle, json
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+")

def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())

@dataclass
class BM25Index:
    doc_ids: List[str]
    bm25: BM25Okapi

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(out_dir: Path) -> "BM25Index":
        with (out_dir / "bm25.pkl").open("rb") as f:
            return pickle.load(f)

def build_bm25(corpus: List[Dict], out_dir: Path) -> BM25Index:
    tokenized = []
    doc_ids = []
    for d in corpus:
        text = (d.get("title", "") + " " + d.get("text", "")).strip()
        tokenized.append(_tokenize(text))
        doc_ids.append(d["id"])
    bm25 = BM25Okapi(tokenized)
    idx = BM25Index(doc_ids=doc_ids, bm25=bm25)
    idx.save(out_dir)

    with (out_dir / "docstore.jsonl").open("w") as f:
        for d in corpus:
            f.write(json.dumps({"id": d["id"], "title": d.get("title", ""), "text": d.get("text", "")}) + "\n")
    return idx

def search_bm25(index_dir: Path, query: str, k: int = 5) -> List[Tuple[str, float]]:
    idx = BM25Index.load(index_dir)
    scores = idx.bm25.get_scores(_tokenize(query))
    # top-k
    top = sorted(zip(idx.doc_ids, scores), key=lambda x: x[1], reverse=True)[:k]
    return top