"""
agent.py
--------
Campus Support Agent built on top of the original semantic retriever.

The agent keeps the same campus-support domain, but adds:
  - intent detection
  - short-term memory for follow-up questions
  - clearer fallback handling when confidence is low
"""

from __future__ import annotations

from typing import Any

from predict import CampusAssistant
from tools import (
    detect_intent,
    extract_subject_from_candidates,
    looks_like_follow_up,
    resolve_follow_up,
    style_answer,
    uncertainty_note,
)


class CampusSupportAgent:
    """Simple agent wrapper around the existing CampusAssistant retriever."""

    def __init__(self, default_top_k: int = 3):
        self.assistant = CampusAssistant(use_glove=True, top_k=default_top_k)
        self.default_top_k = default_top_k

    def answer(
        self,
        query: str,
        session_state: dict[str, Any] | None = None,
        model_choice: str = "Auto",
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Answer a user query with agent-style routing and memory."""
        session_state = session_state or {}
        original_query = query.strip()
        last_subject = session_state.get("last_subject")

        resolved_query = original_query
        if looks_like_follow_up(original_query) and last_subject:
            resolved_query = resolve_follow_up(original_query, last_subject)

        intent = detect_intent(resolved_query)
        retrieval = self.assistant.answer(
            resolved_query,
            model_choice=model_choice,
            top_k=top_k or self.default_top_k,
        )

        subject = extract_subject_from_candidates(retrieval["candidates"]) or last_subject
        answer_text = style_answer(retrieval["answer"], intent)
        if retrieval["uncertain"]:
            answer_text += uncertainty_note(intent)

        if resolved_query != original_query:
            answer_text = f"Assuming you mean **{subject}**.\n\n" + answer_text

        return {
            "answer": answer_text,
            "intent": intent,
            "confidence": retrieval["confidence"],
            "candidates": retrieval["candidates"],
            "uncertain": retrieval["uncertain"],
            "model_used": retrieval["model_used"],
            "resolved_query": resolved_query,
            "original_query": original_query,
            "subject": subject,
        }

    @staticmethod
    def format_details(result: dict[str, Any]) -> str:
        """Create the right-hand details panel shown in the Gradio app."""
        lines = [
            f"**Intent:** {result['intent']}",
            f"**Model used:** {result['model_used']}",
            f"**Resolved query:** {result['resolved_query']}",
            f"**Confidence:** {result['confidence']:.3f}",
        ]

        if result.get("subject"):
            lines.append(f"**Current subject memory:** {result['subject']}")

        lines.append("\n**Top retrieved matches:**")
        for idx, item in enumerate(result["candidates"], start=1):
            lines.append(
                f"{idx}. [{item['score']:.3f}] **Q:** {item['question']}  \n"
                f"   **A:** {item['answer']}"
            )
        return "\n\n".join(lines)
