import json
import numpy as np

# Load embeddings
embeddings = np.load("data/embeddings.npy")

# Precompute norms of backend embeddings for speed
embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_norm = np.where(embeddings_norm == 0, 1e-9, embeddings_norm)
normalized_embeddings = embeddings / embeddings_norm

# Load questions
with open(
    "data/questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


def find_similar(question_embedding, top_k=5):
    # Compute cosine similarity using pure numpy
    q_norm = np.linalg.norm(question_embedding)
    q_norm = 1e-9 if q_norm == 0 else q_norm
    normalized_q = question_embedding / q_norm

    # Dot product of normalized vectors yields cosine similarity
    scores = np.dot(normalized_embeddings, normalized_q)

    # Sort by similarity score descending
    indices = np.argsort(scores)[::-1]

    results = []

    for i in indices:

        # Skip exact same question
        if scores[i] > 0.99:
            continue

        # Skip very weak matches
        if scores[i] < 0.25:
            continue

        results.append({
            "question": questions[i]["question"],
            "topic": questions[i]["topic"],
            "score": round(float(scores[i]) * 100, 2)
        })

        if len(results) == top_k:
            break

    return results