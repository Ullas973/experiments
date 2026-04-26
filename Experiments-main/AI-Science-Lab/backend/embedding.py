from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer


@dataclass
class ChunkConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150


def split_text_into_chunks(text: str, config: ChunkConfig | None = None) -> List[str]:
    """Split raw text into overlapping chunks suitable for retrieval."""
    if not text or not text.strip():
        return []

    cfg = config or ChunkConfig()
    normalized = " ".join(text.split())
    chunks: List[str] = []

    start = 0
    total = len(normalized)
    while start < total:
        end = min(start + cfg.chunk_size, total)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= total:
            break
        start = max(0, end - cfg.chunk_overlap)

    return chunks


class TextEmbedder:
    """Sentence-transformers wrapper used for document/query embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()
