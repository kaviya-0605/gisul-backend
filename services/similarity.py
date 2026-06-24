"""
Similarity Service
==================
Finds the top-K most semantically similar questions from the pre-computed
dataset using cosine similarity (pure NumPy — no scikit-learn needed at
runtime, but scikit-learn is available if needed elsewhere).

Data is loaded lazily on first call — zero I/O at import time.
Pre-normalised embeddings are cached after the first load for speed.
"""

import json
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level caches (populated on first call) ─────────────────────────────
_embeddings: Optional[np.ndarray] = None   # shape (N, 384) — L2-normalised
_questions: Optional[list] = None


def _load_data() -> tuple[np.ndarray, list]:
    """
    Load and cache embeddings + questions from disk.
    Embeddings are L2-normalised at load time so cosine similarity
    reduces to a simple dot product (faster for repeated queries).
    """
    global _embeddings, _questions

    if _embeddings is None:
        try:
            raw = np.load("data/embeddings.npy")
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            _embeddings = (raw / norms).astype(np.float32)
            logger.info("Loaded embeddings: shape %s", _embeddings.shape)
        except FileNotFoundError:
            logger.error("data/embeddings.npy not found. Run prepare_dataset.py first.")
            raise
        except Exception as exc:
            logger.error("Failed to load embeddings: %s", exc)
            raise

    if _questions is None:
        try:
            with open("data/questions.json", "r", encoding="utf-8") as f:
                _questions = json.load(f)
            logger.info("Loaded %d questions.", len(_questions))
        except FileNotFoundError:
            logger.error("data/questions.json not found. Run prepare_dataset.py first.")
            raise
        except Exception as exc:
            logger.error("Failed to load questions: %s", exc)
            raise

    return _embeddings, _questions


def find_similar(question_embedding: np.ndarray, top_k: int = 5) -> list:
    """
    Return the top-K most similar questions to the given embedding.

    Args:
        question_embedding: 1-D numpy array of shape (384,) — already L2-normalised.
        top_k: number of results to return.

    Returns:
        List of dicts with keys: question, topic, score (0-100).
        Returns [] on error (safe fallback — prevents API crash).
    """
    try:
        embeddings, questions = _load_data()

        # Ensure query is float32 and L2-normalised
        q = question_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1e-9
        q = q / q_norm

        # Cosine similarity = dot product of L2-normalised vectors
        scores = np.dot(embeddings, q)

        # Descending sort
        indices = np.argsort(scores)[::-1]

        results = []
        for i in indices:
            score = float(scores[i])

            if score > 0.99:   # skip exact/near-exact duplicate
                continue
            if score < 0.25:   # below relevance threshold — stop early
                break

            results.append({
                "question": questions[i]["question"],
                "topic":    questions[i]["topic"],
                "score":    round(score * 100, 2),
            })

            if len(results) == top_k:
                break

        return results

    except Exception as exc:
        logger.error("find_similar failed: %s", exc)
        return []   # safe fallback — API returns empty list instead of 500