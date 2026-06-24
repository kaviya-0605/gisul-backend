import os
import time
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class HFInferenceModel:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def encode(self, sentences, **kwargs):
        is_single = isinstance(sentences, str)
        inputs = [sentences] if is_single else sentences
        
        # Try Hugging Face Inference API with 3 retries (to handle model cold start)
        for attempt in range(3):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json={"inputs": inputs, "options": {"wait_for_model": True}},
                    timeout=15
                )
                if response.status_code == 200:
                    res_data = response.json()
                    # Return as numpy array matching SentenceTransformer behavior
                    return np.array(res_data[0] if is_single else res_data, dtype=np.float32)
                elif response.status_code == 503:
                    # Model loading on Hugging Face, wait and retry
                    time.sleep(4)
                    continue
                else:
                    raise Exception(f"Hugging Face API returned status {response.status_code}: {response.text}")
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(2)
        
        raise Exception("Failed to fetch embeddings from Hugging Face Inference API")

print("Initializing API-based SentenceTransformer...")
model = HFInferenceModel()
print("API Model Ready")
