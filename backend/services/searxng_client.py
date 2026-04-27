import os
import requests
from dotenv import load_dotenv

load_dotenv()

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")

def search_web(query: str, num_results: int = 3) -> list[dict]:
    url = f"{SEARXNG_URL}/search"
    params = {
        "q": query,
        "format": "json",
        "language": "id"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = []
        for res in data.get("results", [])[:num_results]:
            results.append({
                "snippet": res.get("content", ""),
                "url": res.get("url", "")
            })
        return results
    except Exception as e:
        print(f"SearXNG error: {e}")
        return []
