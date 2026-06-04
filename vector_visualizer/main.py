import os
import time
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymilvus import MilvusClient
from sklearn.decomposition import PCA
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NusantaraLaw 3D Vector Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nusantara_law")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

# Global states
_pca_model = None
_loaded_embeddings = None  # raw embeddings of loaded vectors: shape (N, 4096)
_loaded_metadata = None    # list of loaded metadata dicts

def get_milvus_client() -> MilvusClient:
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    return MilvusClient(uri=uri)

def get_ollama_embedding(text: str) -> list:
    url = f"{OLLAMA_HOST}/api/embed"
    payload = {"model": EMBEDDING_MODEL, "input": text}
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        embs = resp.json().get("embeddings", [])
        return embs[0] if embs else []
    except Exception as e:
        print(f"Ollama embed error: {e}")
        return []

class QueryPayload(BaseModel):
    query: str
    top_k: int = 5

@app.get("/api/vectors")
def get_vectors(limit: int = 500):
    global _pca_model, _loaded_embeddings, _loaded_metadata
    try:
        client = get_milvus_client()
        
        if not client.has_collection(COLLECTION_NAME):
            return {"status": "error", "message": f"Collection '{COLLECTION_NAME}' not found."}
            
        # Get row count info
        stats = {}
        try:
            stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            
        total_rows = stats.get("row_count", 0)

        # Retrieve vectors
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter="id >= 0",
            output_fields=["id", "doc_name", "category", "chunk_text", "page_no", "embedding"],
            limit=limit
        )
        
        if not results:
            return {
                "status": "success",
                "total_rows": total_rows,
                "loaded_rows": 0,
                "vectors": []
            }

        # Separate embeddings and metadata
        embeddings = []
        metadata = []
        for r in results:
            if "embedding" in r and r["embedding"]:
                embeddings.append(r["embedding"])
                metadata.append({
                    "id": r.get("id"),
                    "doc_name": r.get("doc_name", "Unknown"),
                    "category": r.get("category", "Unknown"),
                    "chunk_text": r.get("chunk_text", ""),
                    "page_no": r.get("page_no", 0)
                })

        if not embeddings:
            return {
                "status": "success",
                "total_rows": total_rows,
                "loaded_rows": 0,
                "vectors": []
            }

        embeddings_np = np.array(embeddings, dtype=np.float32)  # shape (N, 4096)
        
        # Fit PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings_np)  # shape (N, 3)

        # Save to global
        _pca_model = pca
        _loaded_embeddings = embeddings_np
        _loaded_metadata = metadata

        # Prepare response
        resp_vectors = []
        for i, meta in enumerate(metadata):
            resp_vectors.append({
                **meta,
                "x": float(embeddings_3d[i, 0]),
                "y": float(embeddings_3d[i, 1]),
                "z": float(embeddings_3d[i, 2])
            })

        # Calculate explained variance ratio
        var_ratio = [float(v) for v in pca.explained_variance_ratio_]

        return {
            "status": "success",
            "total_rows": total_rows,
            "loaded_rows": len(resp_vectors),
            "explained_variance_ratio": var_ratio,
            "vectors": resp_vectors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project_query")
def project_query(payload: QueryPayload):
    global _pca_model, _loaded_embeddings, _loaded_metadata
    if _pca_model is None or _loaded_embeddings is None:
        raise HTTPException(status_code=400, detail="PCA model is not fitted yet. Please load vectors first.")

    query_emb = get_ollama_embedding(payload.query)
    if not query_emb:
        raise HTTPException(status_code=500, detail="Failed to get embedding from Ollama.")

    query_emb_np = np.array(query_emb, dtype=np.float32).reshape(1, -1)  # shape (1, 4096)

    # Project to 3D space
    query_3d = _pca_model.transform(query_emb_np)[0]  # [x, y, z]

    # Calculate cosine distances to loaded embeddings
    norms_loaded = np.linalg.norm(_loaded_embeddings, axis=1)
    norm_query = np.linalg.norm(query_emb_np)
    
    # Avoid divide by zero
    norms_loaded[norms_loaded == 0] = 1e-10
    nq = norm_query if norm_query > 0 else 1e-10

    dot_products = np.dot(_loaded_embeddings, query_emb_np.T).flatten()
    cosine_similarities = dot_products / (norms_loaded * nq)
    cosine_distances = 1.0 - cosine_similarities

    # Find top_k nearest neighbors
    top_k = min(payload.top_k, len(_loaded_metadata))
    nearest_indices = np.argsort(cosine_distances)[:top_k]

    neighbors = []
    for idx in nearest_indices:
        neighbors.append({
            "id": _loaded_metadata[int(idx)]["id"],
            "distance": float(cosine_distances[idx]),
            "similarity": float(cosine_similarities[idx])
        })

    return {
        "status": "success",
        "query_3d": {
            "x": float(query_3d[0]),
            "y": float(query_3d[1]),
            "z": float(query_3d[2])
        },
        "neighbors": neighbors
    }

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
