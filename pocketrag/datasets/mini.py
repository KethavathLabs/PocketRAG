from __future__ import annotations
from pathlib import Path
import json

DOCS = [
    {"id": "D1", "title": "FAISS", "text": "FAISS is a library from Facebook AI Research for efficient similarity search and clustering of dense vectors."},
    {"id": "D2", "title": "BM25", "text": "BM25 is a bag-of-words retrieval function based on term frequency and inverse document frequency for ranking."},
    {"id": "D3", "title": "Transformers", "text": "Transformers is a library that provides state-of-the-art machine learning models including BERT and GPT."},
    {"id": "D4", "title": "Sentence-Transformers", "text": "Sentence-Transformers enables sentence embeddings for tasks like semantic search and clustering."},
    {"id": "D5", "title": "Apple Silicon MPS", "text": "Apple Silicon provides Metal Performance Shaders (MPS) acceleration for PyTorch on macOS."},
    {"id": "D6", "title": "Vector Databases", "text": "Vector databases store embeddings and support nearest neighbor search; examples include FAISS, Qdrant, and Milvus."},
]

QUERIES = [
    {"id": "Q1", "question": "Which library provides efficient similarity search for dense vectors?", "answers": ["FAISS"], "relevant_doc_ids": ["D1", "D6"]},
    {"id": "Q2", "question": "What is BM25 used for?", "answers": ["bag-of-words retrieval", "ranking"], "relevant_doc_ids": ["D2"]},
    {"id": "Q3", "question": "How can I compute sentence embeddings for semantic search?", "answers": ["Sentence-Transformers"], "relevant_doc_ids": ["D4", "D3"]},
    {"id": "Q4", "question": "What accelerates PyTorch on Apple Silicon?", "answers": ["MPS", "Metal Performance Shaders"], "relevant_doc_ids": ["D5"]},
]

QRELS = {  # relevance labels (1 = relevant)
    "Q1": {"D1": 1, "D6": 1},
    "Q2": {"D2": 1},
    "Q3": {"D4": 1, "D3": 1},
    "Q4": {"D5": 1}
}

def write_mini_dataset(root: Path) -> Path:
    """Create a tiny QA dataset at <root>/mini/{corpus.jsonl,queries.jsonl,qrels.json}."""
    ddir = root / "mini"
    ddir.mkdir(parents=True, exist_ok=True)
    (ddir / "corpus.jsonl").write_text("\n".join(json.dumps(x) for x in DOCS) + "\n")
    (ddir / "queries.jsonl").write_text("\n".join(json.dumps(x) for x in QUERIES) + "\n")
    (ddir / "qrels.json").write_text(json.dumps(QRELS, indent=2))
    return ddir