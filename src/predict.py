"""
predict.py
----------
Exposes a prediction interface for the Campus Support Assistant.

This version keeps the original retrieval behaviour but allows dynamic model
selection (Auto, GloVe, or Word2Vec), which makes it easier to use from the
Campus Support Agent and the Gradio UI.
"""

from __future__ import annotations

import numpy as np

from data_processing import load_knowledge_base, build_vocabulary
from train import load_word2vec, ensure_glove_50d, load_glove_subset
from evaluate import (
    build_kb_vectors_glove,
    build_kb_vectors_w2v,
    retrieve_top_k,
    sentence_vector_glove,
    sentence_vector_w2v,
)

CONFIDENCE_THRESHOLD = 0.25


class CampusAssistant:
    """End-to-end inference wrapper for the Campus Support Assistant."""

    def __init__(self, use_glove: bool = True, top_k: int = 3):
        self.top_k = top_k
        self._load_artifacts(use_glove)

    def _load_artifacts(self, use_glove: bool) -> None:
        self.knowledge_base = load_knowledge_base()
        vocab = build_vocabulary(self.knowledge_base)

        self.w2v = load_word2vec()
        self.kb_vecs_w2v = build_kb_vectors_w2v(self.knowledge_base, self.w2v)

        self.glove_embeddings: dict[str, np.ndarray] | None = None
        self.kb_vecs_glove: np.ndarray | None = None

        if use_glove:
            try:
                glove_path = ensure_glove_50d()
                self.glove_embeddings = load_glove_subset(glove_path, vocab)
                self.kb_vecs_glove = build_kb_vectors_glove(
                    self.knowledge_base, self.glove_embeddings
                )
                print("Using GloVe for semantic retrieval.")
            except Exception as exc:
                print(f"GloVe unavailable ({exc}). Falling back to Word2Vec.")

    def _sentence_fn(self, model_choice: str):
        if model_choice == "GloVe" and self.glove_embeddings is not None:
            return lambda text: sentence_vector_glove(text, self.glove_embeddings)
        return lambda text: sentence_vector_w2v(text, self.w2v)

    def _kb_vectors(self, model_choice: str) -> np.ndarray:
        if model_choice == "GloVe" and self.kb_vecs_glove is not None:
            return self.kb_vecs_glove
        return self.kb_vecs_w2v

    def _resolve_model_choice(self, model_choice: str) -> str:
        choice = (model_choice or "Auto").strip()
        if choice == "Auto":
            return "GloVe" if self.glove_embeddings is not None else "Word2Vec"
        if choice == "GloVe" and self.glove_embeddings is None:
            return "Word2Vec"
        return choice

    def answer(
        self,
        query: str,
        model_choice: str = "Auto",
        top_k: int | None = None,
    ) -> dict:
        """Answer a campus support question using the selected embedding model."""
        model_used = self._resolve_model_choice(model_choice)
        candidates = retrieve_top_k(
            query,
            self.knowledge_base,
            self._kb_vectors(model_used),
            self._sentence_fn(model_used),
            top_k=top_k or self.top_k,
        )

        best = candidates[0] if candidates else {"answer": "No answer found.", "score": 0.0}
        return {
            "answer": best["answer"],
            "confidence": best["score"],
            "candidates": candidates,
            "uncertain": best["score"] < CONFIDENCE_THRESHOLD,
            "model_used": model_used,
        }



def interactive_session(assistant: CampusAssistant) -> None:
    """Run a simple question-answering REPL in the terminal."""
    print("\n🎓 Campus Support Assistant")
    print("   Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Assistant: Goodbye! 👋")
            break

        result = assistant.answer(query, model_choice="Auto", top_k=3)
        print(f"\nAssistant: {result['answer']}")
        if result["uncertain"]:
            print("  (Low confidence – please rephrase or contact reception.)")
        print(f"  [confidence: {result['confidence']:.3f} | model: {result['model_used']}]\n")


if __name__ == "__main__":
    print("Loading models …")
    assistant = CampusAssistant(use_glove=True, top_k=3)
    interactive_session(assistant)
