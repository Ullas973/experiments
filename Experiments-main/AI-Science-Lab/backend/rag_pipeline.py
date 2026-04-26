from __future__ import annotations

import os
from typing import Any, Dict, List

import fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI

from embedding import ChunkConfig, TextEmbedder, split_text_into_chunks
from vector_db import FAISSVectorStore


class RAGPipeline:
    """Retrieval-augmented generation pipeline for textbook-grounded responses."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_mode: str = "openai",
        llm_model: str = "gpt-4o-mini",
        openai_api_key: str | None = None,
    ) -> None:
        self.embedder = TextEmbedder(model_name=embedding_model)
        self.vector_store: FAISSVectorStore | None = None
        self.llm_mode = llm_mode
        self.llm_model = llm_model
        self.openai_client = None
        if llm_mode == "openai":
            self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

    def _extract_text_pdfplumber(self, pdf_path: str) -> str:
        pages: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
        return "\n".join(pages)

    def _extract_text_pymupdf(self, pdf_path: str) -> str:
        pages: List[str] = []
        doc = fitz.open(pdf_path)
        try:
            for page in doc:
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append(text)
        finally:
            doc.close()
        return "\n".join(pages)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text = self._extract_text_pdfplumber(pdf_path)
        if text.strip():
            return text

        fallback = self._extract_text_pymupdf(pdf_path)
        if fallback.strip():
            return fallback

        raise ValueError("No readable text found in PDF.")

    def build_index_from_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> Dict[str, Any]:
        full_text = self.extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(
            full_text,
            ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        )
        if not chunks:
            raise ValueError("No chunks created from textbook text.")

        vectors = self.embedder.embed_texts(chunks)
        dim = len(vectors[0])
        self.vector_store = FAISSVectorStore(dim=dim)
        self.vector_store.add(
            vectors=vectors,
            texts=chunks,
            metadata=[
                {"source": os.path.basename(pdf_path), "chunk_id": i}
                for i in range(len(chunks))
            ],
        )
        return {"status": "ok", "chunks_indexed": len(chunks), "embedding_dim": dim}

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            raise ValueError("No vector index found. Upload and process a textbook first.")
        q_vec = self.embedder.embed_query(query)
        hits = self.vector_store.search(q_vec, top_k=top_k)
        return [
            {
                "text": h.text,
                "score": h.score,
                "metadata": h.metadata,
            }
            for h in hits
        ]

    def _answer_with_openai(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        context_block = "\n\n".join(
            f"[Context {i+1}] {item['text']}" for i, item in enumerate(contexts)
        )
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a curriculum-focused science tutor. "
                        "Answer only using retrieved textbook context. "
                        "If context is insufficient, clearly say so."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Retrieved Context:\n{context_block}\n\n"
                        "Write a clear, concise textbook-grounded answer."
                    ),
                },
            ],
        )
        return response.choices[0].message.content.strip()

    def answer_query(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Strict RAG behavior:
        1) Retrieve top-k context
        2) Generate answer from retrieved context only
        """
        contexts = self.retrieve(query=query, top_k=top_k)
        if not contexts:
            return {"answer": "No relevant textbook chunks were retrieved.", "sources": []}

        if self.llm_mode == "openai":
            if self.openai_client is None:
                raise ValueError("OPENAI_API_KEY is missing for openai mode.")
            answer = self._answer_with_openai(query, contexts)
        else:
            answer = (
                "Local LLM mode selected. Retrieved context is available, "
                "but local generation is not configured in this template."
            )

        sources = []
        for ctx in contexts:
            sources.append(
                {
                    "source": ctx["metadata"].get("source", "unknown"),
                    "chunk_id": ctx["metadata"].get("chunk_id"),
                    "score": ctx["score"],
                    "snippet": ctx["text"][:450],
                }
            )

        return {"answer": answer, "sources": sources, "retrieved_context": contexts}
