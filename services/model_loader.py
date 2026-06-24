"""
Model Loader — Lazy Singleton Pattern
======================================
Loads SentenceTransformer("all-MiniLM-L6-v2") locally on first use.
- Zero computation at import time → FastAPI starts instantly.
- Model cached after first load → subsequent requests are fast.
- Thread-safe via module-level lock.
"""

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def get_model():
    """
    Return the cached SentenceTransformer model, loading it on first call.
    Raises RuntimeError if the model cannot be loaded.
    """
    global _model

    if _model is not None:
        return _model

    with _lock:
        # Double-checked locking: another thread may have loaded it while we waited
        if _model is not None:
            return _model

        try:
            import os
            # Reduce PyTorch thread overhead — important on memory-constrained hosts
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

            import torch
            torch.set_num_threads(1)

            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load SentenceTransformer model: %s", exc)
            raise RuntimeError(f"Model loading failed: {exc}") from exc

    return _model


def encode(text: str | list) -> np.ndarray:
    """
    Encode one or more sentences into L2-normalised embeddings.

    Args:
        text: A single string or a list of strings.

    Returns:
        numpy array of shape (384,) for a single string,
        or (N, 384) for a list.

    Raises:
        RuntimeError: if the model cannot be loaded or encoding fails.
    """
    try:
        model = get_model()
        is_single = isinstance(text, str)
        inputs = [text] if is_single else text

        embeddings = model.encode(inputs, convert_to_numpy=True)

        # L2-normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        embeddings = embeddings / norms

        return embeddings[0] if is_single else embeddings

    except Exception as exc:
        logger.error("Encoding failed: %s", exc)
        raise RuntimeError(f"Encoding failed: {exc}") from exc
