from __future__ import annotations

import json
from typing import Any, Dict, List

from rag_pipeline import RAGPipeline


class LabGenerator:
    """Generates curriculum labs from retrieved textbook chunks."""

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag = rag_pipeline

    def _build_prompt(self, concept: str, contexts: List[Dict[str, Any]]) -> str:
        context_block = "\n\n".join(
            f"[Chunk {i+1}] {item['text']}" for i, item in enumerate(contexts)
        )
        return (
            "Create a practical coding lab for the concept using ONLY the retrieved context.\n"
            "Return strict JSON with keys: problem_statement, dataset, python_code, explanation.\n"
            "Rules:\n"
            "- dataset: brief description of data source (synthetic or sklearn)\n"
            "- python_code: runnable Python code\n"
            "- explanation: concise step-by-step explanation\n\n"
            f"Concept: {concept}\n\n"
            f"Retrieved context:\n{context_block}"
        )

    def _fallback_lab(self, concept: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        snippet = contexts[0]["text"][:220] if contexts else "No context available."
        code = """import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Synthetic dataset for concept practice
X, y = make_classification(
    n_samples=400, n_features=8, n_informative=5, n_redundant=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 3))
"""
        return {
            "problem_statement": f"Build a small experiment to explore: {concept}",
            "dataset": "Synthetic classification data via sklearn.make_classification",
            "python_code": code,
            "explanation": (
                "The dataset is generated synthetically, split into training and testing sets, "
                "and a logistic regression model is trained to demonstrate the concept. "
                f"Reference snippet: {snippet}"
            ),
        }

    def generate_lab(self, concept: str, top_k: int = 4) -> Dict[str, Any]:
        contexts = self.rag.retrieve(query=concept, top_k=top_k)
        if not contexts:
            return {
                "error": "No retrieved context found. Upload and process a textbook first.",
                "sources": [],
            }

        if self.rag.llm_mode != "openai" or self.rag.openai_client is None:
            lab = self._fallback_lab(concept, contexts)
        else:
            prompt = self._build_prompt(concept, contexts)
            completion = self.rag.openai_client.chat.completions.create(
                model=self.rag.llm_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a science curriculum lab designer. "
                            "Use only provided context. Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = completion.choices[0].message.content.strip()
            try:
                lab = json.loads(text)
            except json.JSONDecodeError:
                lab = self._fallback_lab(concept, contexts)

        return {
            "lab": lab,
            "sources": [
                {
                    "source": c["metadata"].get("source", "unknown"),
                    "chunk_id": c["metadata"].get("chunk_id"),
                    "snippet": c["text"][:300],
                }
                for c in contexts
            ],
        }
