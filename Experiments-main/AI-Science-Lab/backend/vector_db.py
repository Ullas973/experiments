from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import faiss
import numpy as np


@dataclass
class SearchHit:
    text: str
    score: float
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """
    Vector store backed by FAISS IndexFlatIP.
    Embeddings should be normalized for cosine-like similarity.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadata: List[Dict[str, Any]] | None = None,
    ) -> None:
        if not vectors:
            return
        if len(vectors) != len(texts):
            raise ValueError("vectors and texts length mismatch")

        meta = metadata or [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata and texts length mismatch")

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (n, {self.dim}), got {arr.shape}")

        self.index.add(arr)
        self.texts.extend(texts)
        self.metadata.extend(meta)

    def search(self, query_vector: List[float], top_k: int = 4) -> List[SearchHit]:
        if self.index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        scores, indices = self.index.search(q, top_k)

        hits: List[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(
                SearchHit(
                    text=self.texts[idx],
                    score=float(score),
                    metadata=self.metadata[idx],
                )
            )
        return hits

    def save(self, index_path: str, store_path: str) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(store_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        payload = {
            "dim": self.dim,
            "texts": self.texts,
            "metadata": self.metadata,
        }
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path: str, store_path: str) -> "FAISSVectorStore":
        with open(store_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        db = cls(dim=int(payload["dim"]))
        db.index = faiss.read_index(index_path)
        db.texts = payload["texts"]
        db.metadata = payload["metadata"]
        return db
