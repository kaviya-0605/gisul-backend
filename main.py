


# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

# from sentence_transformers import SentenceTransformer

# from services.similarity import find_similar
# from services.topic_classifier import get_topic

# from database.mongodb import search_collection

# from datetime import datetime


# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model = SentenceTransformer(
#     "all-MiniLM-L6-v2"
# )


# class QuestionRequest(BaseModel):
#     question: str


# @app.post("/ask")
# def ask_question(data: QuestionRequest):

#     embedding = model.encode(
#         data.question
#     )

#     topic = get_topic(
#         data.question
#     )

#     similar_questions = find_similar(
#         embedding
#     )

#     document = {
#         "question": data.question,
#         "topic": topic,
#         "similarQuestions": similar_questions,
#         "createdAt": datetime.utcnow()
#     }

#     search_collection.insert_one(
#         document
#     )

#     return {
#         "topic": topic,
#         "similarQuestions": similar_questions
#     }


# @app.get("/history")
# def get_history():

#     records = list(
#         search_collection.find(
#             {},
#             {"_id": 0}
#         ).sort("createdAt", -1)
#     )

#     return records


# @app.get("/stats")
# def get_stats():

#     total_questions = search_collection.count_documents({})

#     records = list(
#         search_collection.find(
#             {},
#             {"_id": 0}
#         )
#     )

#     topic_counts = {}

#     for item in records:

#         topic = item.get(
#             "topic",
#             "Unknown"
#         )

#         topic_counts[topic] = (
#             topic_counts.get(topic, 0)
#             + 1
#         )

#     return {
#         "totalQuestions": total_questions,
#         "topics": topic_counts
#     }







# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from datetime import datetime
# from collections import Counter

# from sentence_transformers import SentenceTransformer

# from services.similarity import find_similar
# from services.topic_classifier import get_topic

# from database.mongodb import collection

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://127.0.0.1:5173"
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # =====================================================
# # LOAD MODEL ONCE
# # =====================================================

# model = SentenceTransformer(
#     "all-MiniLM-L6-v2"
# )

# # =====================================================
# # REQUEST MODEL
# # =====================================================

# class QuestionRequest(BaseModel):
#     question: str

# # =====================================================
# # ASK QUESTION API
# # =====================================================

# @app.post("/ask")
# def ask_question(data: QuestionRequest):

#     embedding = model.encode(
#         data.question
#     )

#     topic = get_topic(
#         data.question
#     )

#     similar_questions = find_similar(
#         embedding
#     )

#     document = {
#         "question": data.question,
#         "topic": topic,
#         "similarQuestions": similar_questions,
#         "createdAt": datetime.utcnow()
#     }

#     collection.insert_one(
#         document
#     )

#     return {
#         "topic": topic,
#         "similarQuestions": similar_questions
#     }

# # =====================================================
# # HISTORY API
# # =====================================================

# @app.get("/history")
# def get_history():

#     questions = list(
#         collection.find().sort(
#             "createdAt",
#             -1
#         )
#     )

#     results = []

#     for item in questions:

#         results.append({
#             "id": str(item["_id"]),
#             "question": item["question"],
#             "topic": item["topic"],
#             "date": item["createdAt"].strftime(
#                 "%d-%m-%Y"
#             ),
#             "score":
#             item["similarQuestions"][0]["score"]
#             if len(item["similarQuestions"]) > 0
#             else 0
#         })

#     return results

# # =====================================================
# # DASHBOARD API
# # =====================================================

# @app.get("/dashboard")
# def dashboard():

#     data = list(
#         collection.find()
#     )

#     total_questions = len(data)

#     topics = set()

#     total_matches = 0

#     for item in data:

#         topics.add(
#             item["topic"]
#         )

#         total_matches += len(
#             item["similarQuestions"]
#         )

#     progress = min(
#         total_questions * 2,
#         100
#     )

#     return {
#         "totalQuestions":
#         total_questions,

#         "topicsLearned":
#         len(topics),

#         "similarMatches":
#         total_matches,

#         "progress":
#         progress
#     }

# # =====================================================
# # PROFILE API
# # =====================================================

# @app.get("/profile")
# def profile():

#     data = list(
#         collection.find()
#     )

#     topic_counter = Counter()

#     for item in data:

#         topic_counter[
#             item["topic"]
#         ] += 1

#     return {
#         "totalQuestions":
#         len(data),

#         "topics":
#         dict(topic_counter)
#     }

# # =====================================================
# # ROOT
# # =====================================================

# @app.get("/")
# def root():
#     return {
#         "message":
#         "StudySync AI Backend Running"
#     }








import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from datetime import datetime
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer

from services.similarity import find_similar
from services.topic_classifier import get_topic

from database.mongodb import collection

app = FastAPI()

# =====================================================
# CORS
# =====================================================

# Set ALLOWED_ORIGINS in .env as a comma-separated list of frontend URLs.
# e.g. ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:5173
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
)
origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MODEL
# =====================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================================
# REQUEST MODEL
# =====================================================

class QuestionRequest(BaseModel):
    question: str

# =====================================================
# ASK QUESTION
# =====================================================

@app.post("/ask")
def ask_question(data: QuestionRequest):

    embedding = model.encode(
        data.question
    )

    topic = get_topic(
        data.question
    )

    similar_questions = find_similar(
        embedding
    )

    document = {
        "question": data.question,
        "topic": topic,
        "similarQuestions": similar_questions,
        "createdAt": datetime.utcnow()
    }

    collection.insert_one(
        document
    )

    return {
        "topic": topic,
        "similarQuestions": similar_questions
    }

# =====================================================
# HISTORY
# =====================================================

@app.get("/history")
def get_history():

    questions = list(
        collection.find().sort(
            "createdAt",
            -1
        )
    )

    results = []

    for item in questions:

        score = 0

        if len(item["similarQuestions"]) > 0:
            score = item["similarQuestions"][0]["score"]

        results.append({
            "id": str(item["_id"]),
            "question": item["question"],
            "topic": item["topic"],
            "date": item["createdAt"].strftime(
                "%d-%m-%Y"
            ),
            "score": round(score)
        })

    return results

# =====================================================
# DASHBOARD
# =====================================================

@app.get("/dashboard")
def dashboard():

    data = list(
        collection.find()
    )

    total_questions = len(data)

    topics = set()

    total_matches = 0

    total_score = 0
    score_count = 0

    recent_activity = []

    for item in data:

        topics.add(
            item["topic"]
        )

        total_matches += len(
            item["similarQuestions"]
        )

        if len(item["similarQuestions"]) > 0:

            score = item["similarQuestions"][0]["score"]

            total_score += score
            score_count += 1

        recent_activity.append({
            "question": item["question"],
            "topic": item["topic"],
            "date": item["createdAt"].strftime("%d-%m-%Y"),
            "score": round(
                item["similarQuestions"][0]["score"]
            ) if len(item["similarQuestions"]) > 0 else 0
        })

    avg_score = (
        total_score / score_count
        if score_count > 0
        else 0
    )

    progress = min(
        total_questions * 10,
        100
    )

    return {
        "totalQuestions": total_questions,
        "topicsLearned": len(topics),
        "similarMatches": total_matches,
        "averageScore": round(avg_score, 2),
        "progress": progress,
        "recentActivity": recent_activity[-5:]
    }

# =====================================================
# PROFILE
# =====================================================

@app.get("/profile")
def profile():

    records = list(
        collection.find()
    )

    total_questions = len(records)

    topic_counter = Counter()

    monthly_counter = defaultdict(int)

    total_score = 0
    score_count = 0

    for item in records:

        topic_counter[
            item["topic"]
        ] += 1

        month = item["createdAt"].strftime(
            "%b"
        )

        monthly_counter[
            month
        ] += 1

        if len(item["similarQuestions"]) > 0:

            total_score += item[
                "similarQuestions"
            ][0]["score"]

            score_count += 1

    average_similarity = (
        total_score / score_count
        if score_count > 0
        else 0
    )

    monthly_data = []

    running_total = 0

    month_order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
    ]

    for month in month_order:

        running_total += monthly_counter.get(
            month,
            0
        )

        monthly_data.append({
            "month": month,
            "questions": running_total
        })

    return {
        "name": "StudySync User",
        "email": "student@example.com",
        "joined": "2026",

        "totalQuestions": total_questions,

        "topics": dict(
            topic_counter
        ),

        "monthlyData": monthly_data,

        "averageSimilarity": round(
            average_similarity,
            2
        ),

        "learningProgress": min(
            total_questions * 10,
            100
        )
    }

# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {
        "message":
        "StudySync AI Backend Running"
    }