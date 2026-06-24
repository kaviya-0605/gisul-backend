import os
import json
import random
import numpy as np

from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# =====================================================
# CREATE DATA FOLDER
# =====================================================

os.makedirs("data", exist_ok=True)


# =====================================================
# SUBJECT DETECTOR
# =====================================================

def detect_subject(question):
    q = question.lower()

    if any(word in q for word in [
        "cell", "plant", "dna", "photosynthesis",
        "organism", "gene", "protein", "enzyme"
    ]):
        return "Biology"

    elif any(word in q for word in [
        "force", "gravity", "motion",
        "energy", "velocity", "acceleration"
    ]):
        return "Physics"

    elif any(word in q for word in [
        "acid", "base", "reaction",
        "molecule", "chemical", "element"
    ]):
        return "Chemistry"

    elif any(word in q for word in [
        "equation", "algebra", "geometry",
        "triangle", "fraction", "calculate"
    ]):
        return "Math"

    elif any(word in q for word in [
        "computer", "program", "algorithm",
        "database", "software", "internet"
    ]):
        return "Computer Science"

    else:
        return "General Science"


# =====================================================
# LOAD DATASETS
# =====================================================

all_questions = []
# =====================================================
# LOAD CUSTOM COMPUTER SCIENCE QUESTIONS
# =====================================================

try:

    with open(
        "data/cs_questions.json",
        "r",
        encoding="utf-8"
    ) as f:

        cs_questions = json.load(f)

    all_questions.extend(cs_questions)

    print(
        f"Custom CS Questions Loaded: {len(cs_questions)}"
    )

except Exception as e:

    print(
        "Custom CS Dataset Error:",
        e
    )

print("\nLoading SciQ Dataset...")

try:
    sciq = load_dataset("sciq")

    for split in ["train", "validation", "test"]:

        for q in sciq[split]["question"]:

            if q and len(q.strip()) > 10:

                all_questions.append({
                    "question": q.strip(),
                    "topic": detect_subject(q)
                })

    print("SciQ Loaded")

except Exception as e:
    print("SciQ Error:", e)


print("\nLoading OpenBookQA Dataset...")

try:
    openbook = load_dataset("openbookqa")

    for split in ["train", "validation", "test"]:

        for q in openbook[split]["question_stem"]:

            if q and len(q.strip()) > 10:

                all_questions.append({
                    "question": q.strip(),
                    "topic": detect_subject(q)
                })

    print("OpenBookQA Loaded")

except Exception as e:
    print("OpenBookQA Error:", e)


print("\nLoading ARC Easy Dataset...")

try:
    arc_easy = load_dataset("ai2_arc", "ARC-Easy")

    for split in ["train", "validation", "test"]:

        for q in arc_easy[split]["question"]:

            if q and len(q.strip()) > 10:

                all_questions.append({
                    "question": q.strip(),
                    "topic": detect_subject(q)
                })

    print("ARC Easy Loaded")

except Exception as e:
    print("ARC Easy Error:", e)


print("\nLoading ARC Challenge Dataset...")

try:
    arc_hard = load_dataset("ai2_arc", "ARC-Challenge")

    for split in ["train", "validation", "test"]:

        for q in arc_hard[split]["question"]:

            if q and len(q.strip()) > 10:

                all_questions.append({
                    "question": q.strip(),
                    "topic": detect_subject(q)
                })

    print("ARC Challenge Loaded")

except Exception as e:
    print("ARC Challenge Error:", e)


# =====================================================
# REMOVE DUPLICATES
# =====================================================

print("\nRemoving Duplicates...")

seen = set()
unique_questions = []

for item in all_questions:

    q = item["question"].lower().strip()

    if q not in seen:

        seen.add(q)
        unique_questions.append(item)

print("Unique Questions:", len(unique_questions))


# =====================================================
# TAKE 5000 QUESTIONS
# =====================================================

random.seed(42)
random.shuffle(unique_questions)

final_questions = unique_questions[:10000]

print("Final Question Count:", len(final_questions))


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

print("\nLoading Embedding Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model Loaded")


# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

questions_text = [
    q["question"]
    for q in final_questions
]

print("\nGenerating Embeddings...")

embeddings = model.encode(
    questions_text,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embedding Shape:", embeddings.shape)


# =====================================================
# SAVE QUESTIONS
# =====================================================

with open(
    "data/questions.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_questions,
        f,
        ensure_ascii=False,
        indent=2
    )

print("questions.json saved")


# =====================================================
# SAVE EMBEDDINGS
# =====================================================

np.save(
    "data/embeddings.npy",
    embeddings
)

print("embeddings.npy saved")


# =====================================================
# SAVE TOPICS
# =====================================================

topics = sorted(
    list(
        set(
            q["topic"]
            for q in final_questions
        )
    )
)

with open(
    "data/topics.json",
    "w"
) as f:

    json.dump(
        topics,
        f,
        indent=2
    )

print("topics.json saved")


# =====================================================
# FINISHED
# =====================================================

print("\n====================================")
print("DATASET PREPARATION COMPLETED")
print("====================================")

print("\nGenerated Files:")

print("data/questions.json")
print("data/embeddings.npy")
print("data/topics.json")

print("\nReady for FastAPI Backend")