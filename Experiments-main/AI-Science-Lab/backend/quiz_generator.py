from __future__ import annotations

import json
from typing import Any, Dict, List

from rag_pipeline import RAGPipeline


class QuizGenerator:
    """Generates quiz questions grounded in retrieved textbook context."""

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag = rag_pipeline

    def _prompt(self, topic: str, contexts: List[Dict[str, Any]]) -> str:
        context_block = "\n\n".join(
            f"[Chunk {i+1}] {item['text']}" for i, item in enumerate(contexts)
        )
        return (
            "Create a quiz from the retrieved context only.\n"
            "Return strict JSON with keys:\n"
            "- mcqs: list of 5 objects with fields question, options (4 strings), answer\n"
            "- short_questions: list of 2 strings\n"
            "Do not use external facts outside retrieved text.\n\n"
            f"Topic: {topic}\n\n"
            f"Retrieved context:\n{context_block}"
        )

    def _fallback_quiz(self, topic: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        basis = contexts[0]["text"][:200] if contexts else topic
        mcqs = []
        for i in range(1, 6):
            mcqs.append(
                {
                    "question": f"Based on the textbook context, which statement best matches key idea {i} about {topic}?",
                    "options": [
                        f"Option A related to: {basis[:40]}",
                        "Option B (distractor)",
                        "Option C (distractor)",
                        "Option D (distractor)",
                    ],
                    "answer": "Option A related to context",
                }
            )
        return {
            "mcqs": mcqs,
            "short_questions": [
                f"Explain one core principle of {topic} from the retrieved textbook text.",
                f"How would you apply {topic} using the context-provided approach?",
            ],
        }

    def generate_quiz(self, topic: str, top_k: int = 4) -> Dict[str, Any]:
        contexts = self.rag.retrieve(query=topic, top_k=top_k)
        if not contexts:
            return {
                "error": "No retrieved context found. Upload and process a textbook first.",
                "sources": [],
            }

        if self.rag.llm_mode != "openai" or self.rag.openai_client is None:
            quiz = self._fallback_quiz(topic, contexts)
        else:
            completion = self.rag.openai_client.chat.completions.create(
                model=self.rag.llm_model,
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a science assessment creator. Use only provided context. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": self._prompt(topic, contexts)},
                ],
            )
            text = completion.choices[0].message.content.strip()
            try:
                quiz = json.loads(text)
            except json.JSONDecodeError:
                quiz = self._fallback_quiz(topic, contexts)

        return {
            "quiz": quiz,
            "sources": [
                {
                    "source": c["metadata"].get("source", "unknown"),
                    "chunk_id": c["metadata"].get("chunk_id"),
                    "snippet": c["text"][:300],
                }
                for c in contexts
            ],
        }
