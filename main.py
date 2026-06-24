"""
StudySync AI — FastAPI Backend
================================
Endpoints:
  POST /ask          — find semantically similar questions
  GET  /history      — all past questions (newest first)
  GET  /dashboard    — aggregate stats
  GET  /profile      — per-user stats + monthly activity
  GET  /             — health check

Design decisions:
  • Model is loaded LAZILY on the first /ask request — FastAPI starts
    in < 1 second regardless of model size.
  • All ML operations are wrapped in try/except — the API never crashes
    due to embedding failures; it returns a 500 with a clear message.
  • MongoDB _id (ObjectId) is serialised to str for JSON compatibility.
"""

import logging
import os
from collections import Counter, defaultdict
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Service imports (no heavy work happens here — all lazy) ───────────────────
from services.similarity import find_similar
from services.topic_classifier import get_topic
from database.mongodb import collection

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="StudySync AI",
    description="Semantic question similarity search backend",
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS in your .env / Render environment variables as a
# comma-separated list:  https://gisul-frontend-eight.vercel.app,http://localhost:5173
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS origins: %s", origins)


# ── Request schema ─────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe_score(item: dict) -> float:
    """Return the best similarity score from a history record, or 0."""
    qs = item.get("similarQuestions", [])
    return qs[0]["score"] if qs else 0


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ask")
def ask_question(data: QuestionRequest):
    """
    Accept a study question, classify its topic, find semantically similar
    questions from the dataset, persist the record, and return results.
    """
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # ── Embed the question (model loaded lazily here on first call) ────────
        from services.model_loader import encode
        embedding = encode(data.question)
    except RuntimeError as exc:
        logger.error("/ask — model encode failed: %s", exc)
        raise HTTPException(status_code=503, detail="Embedding model unavailable. Try again shortly.")

    try:
        topic = get_topic(data.question)
    except Exception as exc:
        logger.warning("/ask — topic classification failed (%s), using fallback.", exc)
        topic = "General Science"

    similar_questions = find_similar(embedding)   # returns [] on error — never raises

    # ── Persist to MongoDB ────────────────────────────────────────────────────
    document = {
        "question":        data.question,
        "topic":           topic,
        "similarQuestions": similar_questions,
        "createdAt":       datetime.utcnow(),
    }
    try:
        collection.insert_one(document)
    except Exception as exc:
        logger.error("/ask — MongoDB insert failed: %s", exc)
        # Still return results even if persistence fails

    return {
        "topic":            topic,
        "similarQuestions": similar_questions,
    }


@app.get("/history")
def get_history():
    """Return all past questions sorted newest-first."""
    try:
        records = list(collection.find().sort("createdAt", -1))
    except Exception as exc:
        logger.error("/history — DB query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable.")

    results = []
    for item in records:
        try:
            results.append({
                "id":       str(item["_id"]),
                "question": item["question"],
                "topic":    item.get("topic", "Unknown"),
                "date":     item["createdAt"].strftime("%d-%m-%Y"),
                "score":    round(_safe_score(item)),
                "similarQuestions": item.get("similarQuestions", []),
                "createdAt": item["createdAt"].isoformat(),
            })
        except Exception as exc:
            logger.warning("Skipping malformed history record: %s", exc)

    return results


@app.get("/dashboard")
def dashboard():
    """Return aggregate learning statistics."""
    try:
        data = list(collection.find())
    except Exception as exc:
        logger.error("/dashboard — DB query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable.")

    total_questions = len(data)
    topics: set = set()
    total_matches = 0
    total_score = 0.0
    score_count = 0
    recent_activity = []

    for item in data:
        topics.add(item.get("topic", "Unknown"))
        qs = item.get("similarQuestions", [])
        total_matches += len(qs)

        if qs:
            total_score += qs[0]["score"]
            score_count += 1

        recent_activity.append({
            "question": item["question"],
            "topic":    item.get("topic", "Unknown"),
            "date":     item["createdAt"].strftime("%d-%m-%Y"),
            "score":    round(qs[0]["score"]) if qs else 0,
        })

    avg_score  = round(total_score / score_count, 2) if score_count else 0
    progress   = min(total_questions * 10, 100)

    return {
        "totalQuestions": total_questions,
        "topicsLearned":  len(topics),
        "similarMatches": total_matches,
        "averageScore":   avg_score,
        "progress":       progress,
        "recentActivity": recent_activity[-5:],
    }


@app.get("/profile")
def profile():
    """Return per-user stats with monthly activity and topic breakdown."""
    try:
        records = list(collection.find())
    except Exception as exc:
        logger.error("/profile — DB query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable.")

    total_questions = len(records)
    topic_counter: Counter  = Counter()
    monthly_counter: defaultdict = defaultdict(int)
    total_score = 0.0
    score_count = 0

    for item in records:
        topic_counter[item.get("topic", "Unknown")] += 1
        monthly_counter[item["createdAt"].strftime("%b")] += 1

        qs = item.get("similarQuestions", [])
        if qs:
            total_score += qs[0]["score"]
            score_count += 1

    average_similarity = round(total_score / score_count, 2) if score_count else 0

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    running_total = 0
    monthly_data  = []
    for month in month_order:
        running_total += monthly_counter.get(month, 0)
        monthly_data.append({"month": month, "questions": running_total})

    return {
        "name":              "StudySync User",
        "email":             "student@example.com",
        "joined":            "2026",
        "totalQuestions":    total_questions,
        "topics":            dict(topic_counter),
        "monthlyData":       monthly_data,
        "averageSimilarity": average_similarity,
        "learningProgress":  min(total_questions * 10, 100),
    }


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status":  "ok",
        "message": "StudySync AI Backend Running",
    }