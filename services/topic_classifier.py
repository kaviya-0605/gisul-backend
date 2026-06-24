"""
Topic Classifier
================
Classifies a question into one of six predefined academic topics using
semantic similarity between the question and topic descriptor phrases.

- Topic embeddings are computed lazily on first call (not at import time).
- Safe fallback ("General Science") returned if classification fails.
- Uses the shared model from model_loader — model is loaded only once.
"""

import logging
from typing import Optional

import numpy as np

from services.model_loader import encode

logger = logging.getLogger(__name__)

# ── Topic descriptor vocabulary ───────────────────────────────────────────────
TOPICS: dict[str, str] = {
    "Biology":
        "photosynthesis dna cell plant gene organism enzyme protein",
    "Physics":
        "force gravity motion energy velocity acceleration wave",
    "Chemistry":
        "acid base molecule reaction element compound periodic",
    "Math":
        "algebra geometry equation calculus probability trigonometry",
    "Computer Science":
        "react node mongodb express javascript python java api backend "
        "frontend database programming software algorithm data structure",
    "General Science":
        "science experiment observation hypothesis research",
}

TOPIC_NAMES: list[str] = list(TOPICS.keys())
FALLBACK_TOPIC: str = "General Science"

# ── Cache ─────────────────────────────────────────────────────────────────────
_topic_embeddings: Optional[np.ndarray] = None  # shape (6, 384) — L2-normalised


def _get_topic_embeddings() -> np.ndarray:
    """
    Return cached, L2-normalised topic embeddings.
    Computed on first call; cached for all subsequent calls.
    """
    global _topic_embeddings

    if _topic_embeddings is not None:
        return _topic_embeddings

    try:
        descriptors = list(TOPICS.values())
        _topic_embeddings = encode(descriptors)   # shape (6, 384), already L2-normalised
        logger.info("Topic embeddings computed for %d topics.", len(TOPIC_NAMES))
    except Exception as exc:
        logger.error("Failed to compute topic embeddings: %s", exc)
        raise

    return _topic_embeddings


def get_topic(question: str) -> str:
    """
    Classify *question* into one of the predefined academic topics.

    Returns:
        Topic name string (e.g. "Biology", "Physics").
        Falls back to FALLBACK_TOPIC on any error.
    """
    try:
        topic_embeddings = _get_topic_embeddings()

        # encode() returns L2-normalised embedding
        q_emb = encode(question)   # shape (384,)

        # Cosine similarity = dot product (both sides are L2-normalised)
        scores = np.dot(topic_embeddings, q_emb)

        best_idx = int(np.argmax(scores))
        return TOPIC_NAMES[best_idx]

    except Exception as exc:
        logger.error("get_topic failed for %r: %s", question, exc)
        return FALLBACK_TOPIC   # safe fallback — never crashes the API