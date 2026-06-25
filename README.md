# Gisul Backend
StudySync AI Backend is developed using FastAPI and provides REST APIs for AI-powered semantic question search, topic classification, user-specific history management, and learning analytics. The backend leverages Sentence Transformers to generate vector embeddings, enabling semantic similarity search instead of traditional keyword matching. User data and search history are securely stored in MongoDB Atlas, with access controlled through JWT authentication.

###AI Workflow

The backend follows the workflow below:
User submits a study question.
FastAPI receives the request.
Sentence Transformer converts the question into a vector embedding.
Topic classifier predicts the subject.
Cosine similarity is computed with the stored question embeddings.
Top similar questions are retrieved.
Search history is stored in MongoDB.
Results are returned to the frontend.

Each search is linked to the authenticated user's account.so each search is stored to particular use account.it ensure data privacy,user personalization.

Semantic Search
StudySync AI uses Sentence Transformers to perform semantic similarity search.

Model used:
all-MiniLM-L6-v2

Advantages:
Understands sentence meaning
Handles paraphrased questions
Better than keyword matching
Fast inference
Lightweight transformer model

Dataset
The backend uses a hybrid dataset consisting of:

SciQ Dataset
OpenBookQA Dataset
Custom Computer Science Dataset

## Tech Stack

- **FastAPI** — API framework
- **Uvicorn** — ASGI server
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — semantic embeddings
- **Scikit-learn** — cosine similarity
- **PyMongo** — MongoDB Atlas connection
- **Python-dotenv** — environment variable management

## Getting Started

### 1. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```
Edit `.env` with your MongoDB URI and allowed frontend origins.

### 4. Run development server
```bash
uvicorn main:app --reload --port 8000
```

### 5. For production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Description |
|---|---|
| `MONGO_URI` | MongoDB Atlas connection string |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed frontend URLs |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Submit a question, returns similar questions + topic |
| `GET` | `/history` | All past questions (newest first) |
| `GET` | `/dashboard` | Stats: total questions, topics, matches, progress |
| `GET` | `/profile` | User profile with monthly data & topic breakdown |
| `GET` | `/` | Health check |

## Project Structure

```
backend/
├── main.py                   # FastAPI app & all routes
├── requirements.txt
├── .env.example
├── database/
│   └── mongodb.py            # MongoDB connection
├── services/
│   ├── similarity.py         # Cosine similarity search
│   └── topic_classifier.py   # Sentence-embedding topic detection
├── data/
│   ├── questions.json        # Question dataset
│   ├── embeddings.npy        # Pre-computed embeddings (not in git)
│   └── topics.json
└── prepare_dataset.py        # Script to build embeddings.npy
```

> **Note:** `data/embeddings.npy` is excluded from git (15 MB binary).  
> Run `python prepare_dataset.py` after cloning to regenerate it.
