import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "bayhaqieee/qwen3.5-9b-nlaw-gguf")
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

def query_hf_api(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "512")),
            "temperature": float(os.getenv("TEMPERATURE", "0.1")),
            "repetition_penalty": 1.15,
            "return_full_text": False
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)
    except Exception as e:
        return f"Error communicating with HF Inference API: {str(e)}"
