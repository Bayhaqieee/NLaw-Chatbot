import os
import time
from pymilvus import MilvusClient
from services.ollama_client import get_embeddings_local
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nusantara_law")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")

_client = None
_collection_loaded = False


def _ensure_collection_loaded(client: MilvusClient) -> None:
    """Load the Milvus collection into memory if not already loaded.
    Checks state to avoid calling load_collection if it is already loaded or in progress.
    """
    global _collection_loaded
    if _collection_loaded:
        return

    try:
        if not client.has_collection(COLLECTION_NAME):
            print(f"[milvus] Collection '{COLLECTION_NAME}' does not exist yet.")
            return

        state = client.get_load_state(COLLECTION_NAME)
        print(f"[milvus] Collection '{COLLECTION_NAME}' load state: {state}")

        state_str = str(state).lower()

        # If it is already loaded, we are done
        if "loaded" in state_str and "loading" not in state_str and "not" not in state_str:
            print(f"[milvus] Collection already loaded — skipping load_collection().")
            _collection_loaded = True
            return

        # If it is currently loading, wait for it to finish rather than double-triggering
        if "loading" in state_str:
            print(f"[milvus] Collection is currently loading in background. Waiting...")
            for i in range(30):
                time.sleep(1)
                new_state = client.get_load_state(COLLECTION_NAME)
                new_state_str = str(new_state).lower()
                if "loaded" in new_state_str and "loading" not in new_state_str and "not" not in new_state_str:
                    print(f"[milvus] Collection finished loading after {i+1}s.")
                    _collection_loaded = True
                    return
            print("[milvus] Collection still loading after timeout. Proceeding...")
            return

        # If not loaded at all, initiate load
        print(f"[milvus] Collection not loaded. Calling load_collection()...")
        client.load_collection(COLLECTION_NAME)
        
        # Poll for completion
        for i in range(30):
            time.sleep(1)
            new_state = client.get_load_state(COLLECTION_NAME)
            new_state_str = str(new_state).lower()
            if "loaded" in new_state_str and "loading" not in new_state_str and "not" not in new_state_str:
                print(f"[milvus] Collection became ready after {i+1}s.")
                _collection_loaded = True
                return
        print("[milvus] Collection load initiated, but not fully ready yet.")

    except Exception as e:
        print(f"[milvus] WARNING: load_collection failed: {e}")
        print(f"[milvus] Will retry on next request.")


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        print(f"[milvus] Connecting to Milvus at {uri}...")
        _client = MilvusClient(uri=uri)
        print(f"[milvus] Connected.")
        _ensure_collection_loaded(_client)
    return _client


def search_milvus(query: str, top_k: int = 5) -> tuple[list[dict], float]:
    client = get_client()

    # Ensure loaded (no-op if already loaded; retries if previous load failed)
    _ensure_collection_loaded(client)

    print(f"[milvus] Generating embedding for query: '{query[:50]}...'")
    query_vector = get_embeddings_local(query, model=EMBEDDING_MODEL)

    if not query_vector:
        print("[milvus] ERROR: Embedding returned empty — cannot search.")
        return [], 0.0

    print(f"[milvus] Searching collection '{COLLECTION_NAME}' (top_k={top_k})...")
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k,
            output_fields=["chunk_text", "doc_name", "page_no", "category"],
            search_params={"metric_type": "L2", "params": {"ef": 64}}
        )
    except Exception as e:
        print(f"[milvus] Search error: {e}")
        # If collection not loaded, reset flag and retry once
        if "not loaded" in str(e).lower():
            global _collection_loaded
            _collection_loaded = False
            print("[milvus] Attempting to reload collection and retry...")
            _ensure_collection_loaded(client)
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vector],
                limit=top_k,
                output_fields=["chunk_text", "doc_name", "page_no", "category"],
                search_params={"metric_type": "L2", "params": {"ef": 64}}
            )
        else:
            raise

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
    print(f"[milvus] Search returned {len(chunks)} chunks (max_score={max_score:.4f})")
    return chunks, max_score


def insert_chunks(chunks: list[dict]):
    client = get_client()
    client.insert(collection_name=COLLECTION_NAME, data=chunks)
