# Gisul Backend

FastAPI + Python backend for **StudySync AI** — semantic question similarity search using sentence embeddings and MongoDB.

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
