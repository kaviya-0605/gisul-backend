import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
embeddings = np.load("data/embeddings.npy")

# Load questions
with open(
    "data/questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


def find_similar(question_embedding, top_k=5):

    scores = cosine_similarity(
        [question_embedding],
        embeddings
    )[0]

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