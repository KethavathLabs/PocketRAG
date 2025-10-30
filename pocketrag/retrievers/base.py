from __future__ import annotations
from typing import List, Tuple, Protocol

class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Return top-k (doc_id, score) for the query.
        Higher score means more relevant.
        """
        ...