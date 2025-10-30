from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable
import json

def load_corpus_jsonl(path: Path) -> List[Dict]:
    docs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs

def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")