import os
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nusantara_law")

_model = None
_client = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _model

def get_client() -> MilvusClient:
    global _client
    if _client is None:
        uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        _client = MilvusClient(uri=uri)
    return _client

def search_milvus(query: str, top_k: int = 5) -> tuple[list[dict], float]:
    client = get_client()
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["chunk_text", "doc_name", "page_no", "category"],
        search_params={"metric_type": "L2", "params": {"ef": 64}}
    )

    chunks = []
    max_score = 0.0
    for hits in results:
        for hit in hits:
            score = hit.get("distance", 0.0)
            max_score = max(max_score, score)
            entity = hit.get("entity", {})
            chunks.append({
                "chunk_text": entity.get("chunk_text"),
                "doc_name":   entity.get("doc_name"),
                "page_no":    entity.get("page_no"),
                "category":   entity.get("category"),
                "score":      score,
            })
    return chunks, max_score

def insert_chunks(chunks: list[dict]):
    client = get_client()
    client.insert(collection_name=COLLECTION_NAME, data=chunks)
