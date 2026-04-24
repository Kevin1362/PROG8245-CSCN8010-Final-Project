"""
tools.py
--------
Helper utilities for the Campus Support Agent.

These functions add light-weight agent behaviour on top of the existing
retrieval system:
  - intent detection
  - follow-up resolution using short-term memory
  - subject extraction
  - friendly fallback messages
"""

from __future__ import annotations

import re
from typing import Any

FOLLOW_UP_PRONOUNS = {
    "it",
    "that",
    "this",
    "there",
    "they",
    "them",
    "those",
    "these",
    "he",
    "she",
}

INTENT_KEYWORDS = {
    "hours": ["open", "close", "hours", "time", "schedule"],
    "location": ["where", "located", "room", "building", "map", "nearest"],
    "contact": ["contact", "phone", "email", "call", "reach"],
    "procedure": [
        "how do i",
        "how can i",
        "apply",
        "request",
        "reset",
        "update",
        "withdraw",
        "appeal",
        "book",
        "renew",
        "access",
        "join",
        "pay",
        "file",
        "change",
        "check",
    ],
    "availability": [
        "is there",
        "are there",
        "available",
        "offer",
        "provide",
        "can i",
        "does the campus",
    ],
    "emergency": ["emergency", "security", "urgent", "unsafe"],
}

LEADING_PATTERNS = [
    r"^what time does\s+",
    r"^when does\s+",
    r"^where is\s+",
    r"^where can i\s+",
    r"^where do i\s+",
    r"^how can i contact\s+",
    r"^how do i\s+",
    r"^how can i\s+",
    r"^is there\s+",
    r"^are there\s+",
    r"^can i\s+",
    r"^what happens if i\s+",
    r"^what happens if\s+",
    r"^how much does\s+",
    r"^what is\s+",
    r"^what are\s+",
    r"^can students\s+",
    r"^does the campus\s+",
]


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def detect_intent(query: str) -> str:
    """Classify the user query into a simple campus-support intent."""
    q = query.lower().strip()

    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q:
                return intent
    return "faq"



def looks_like_follow_up(query: str) -> bool:
    """Return True when the query depends on previous context."""
    q = query.lower().strip()
    if len(q.split()) <= 6 and any(f" {p} " in f" {q} " for p in FOLLOW_UP_PRONOUNS):
        return True
    if q.startswith(("and ", "what about ", "how about ", "where about ")):
        return True
    return False



def resolve_follow_up(query: str, last_subject: str | None) -> str:
    """Replace simple pronouns in a follow-up query using the last subject."""
    if not last_subject:
        return query

    resolved = query
    for pronoun in FOLLOW_UP_PRONOUNS:
        resolved = re.sub(
            rf"\b{re.escape(pronoun)}\b",
            last_subject,
            resolved,
            flags=re.IGNORECASE,
        )
    return normalise_whitespace(resolved)



def extract_subject_from_text(text: str) -> str | None:
    """Extract the main subject from a campus FAQ-style question."""
    q = text.lower().strip()
    q = re.sub(r"[?!.]+$", "", q)
    for pattern in LEADING_PATTERNS:
        q = re.sub(pattern, "", q)
    q = q.strip(" -:")
    q = re.sub(r"\b(on campus|through the campus|through campus)\b", "", q)
    q = normalise_whitespace(q)
    if not q:
        return None
    return q



def extract_subject_from_candidates(candidates: list[dict[str, Any]]) -> str | None:
    """Infer a conversation subject from the best retrieval candidate."""
    if not candidates:
        return None

    for candidate in candidates:
        question = candidate.get("question", "")
        subject = extract_subject_from_text(question)
        if subject:
            return subject
    return None



def uncertainty_note(intent: str) -> str:
    """Return a user-friendly fallback for uncertain answers."""
    if intent == "location":
        return (
            "\n\n⚠️ I am not fully confident about the exact location. "
            "Please confirm with student services, reception, or the campus map."
        )
    if intent == "hours":
        return (
            "\n\n⚠️ I am not fully confident about the hours. "
            "Please confirm with the office directly before visiting."
        )
    if intent == "contact":
        return (
            "\n\n⚠️ I am not fully confident about the contact details. "
            "Please check the official college website or student services."
        )
    return (
        "\n\n⚠️ I am not fully confident about this answer. "
        "Please try rephrasing your question or contact student services."
    )



def style_answer(answer: str, intent: str) -> str:
    """Lightly polish the retrieved answer without changing its meaning."""
    answer = answer.strip()
    if not answer:
        return "No answer found in the knowledge base."

    prefixes = {
        "hours": "Here are the hours I found: ",
        "location": "Here is the location information I found: ",
        "contact": "Here is the contact information I found: ",
        "procedure": "Here is the process I found: ",
        "availability": "Here is what I found: ",
        "emergency": "Here is the emergency-related information I found: ",
        "faq": "Here is the best answer I found: ",
    }
    return prefixes.get(intent, "Here is the best answer I found: ") + answer
