import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is not set")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
    socketTimeoutMS=5000
)

try:
    client.admin.command("ping")
    print("MongoDB Connected Successfully")
except Exception as e:
    print("MongoDB Connection Failed:", e)

db = client["gisul"]

collection = db["search"]
users_collection = db["users"]