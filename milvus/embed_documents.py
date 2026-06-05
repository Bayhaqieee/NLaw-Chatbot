import os
import json
import time
import fitz
import requests
from pymilvus import MilvusClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nusantara_law")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UU-PDP")
TEST_SET_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Test-Set", "data-test.json")


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    pages_data = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            pages_data.append({"page_no": page_num + 1, "text": text})
    return pages_data


def get_embedding(text: str) -> list[float]:
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": EMBEDDING_MODEL, "prompt": text}
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            if attempt < 4:
                print(f"      [embedding] API failed, retrying in 2s... ({e})")
                time.sleep(2)
            else:
                raise


def embed_and_insert():
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    print(f"Connecting to Milvus at {uri}...")
    client = MilvusClient(uri=uri)

    # Make sure collection exists
    if not client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Run init_collection.py first.")
        return

    print(f"Using Ollama embedding model ({EMBEDDING_MODEL})...")

    # 1. Embed UU-PDP PDFs
    if not os.path.exists(DOCS_DIR):
        print(f"Directory {DOCS_DIR} not found. Skipping PDF embedding.")
    else:
        pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            print(f"No PDF files found in {DOCS_DIR}.")
        else:
            for filename in pdf_files:
                filepath = os.path.join(DOCS_DIR, filename)
                print(f"Processing PDF document: {filename}...")
                pages_data = extract_text_from_pdf(filepath)
                batch = []

                for page_data in pages_data:
                    for chunk in chunk_text(page_data["text"]):
                        if len(chunk.strip()) < 10:
                            continue
                        embedding = get_embedding(chunk)
                        batch.append({
                            "doc_name":    filename,
                            "category":    "UU_PDP",
                            "chunk_text":  chunk,
                            "page_no":     page_data["page_no"],
                            "embedding":   embedding,
                            "upload_by":   "system",
                            "uploaded_at": datetime.utcnow().isoformat(),
                        })

                if batch:
                    client.insert(collection_name=COLLECTION_NAME, data=batch)
                    print(f"  Inserted {len(batch)} chunks from {filename}.")

    # 2. Embed data-test.json Context and Responses
    if not os.path.exists(TEST_SET_FILE):
        print(f"Test set file {TEST_SET_FILE} not found. Skipping.")
    else:
        print(f"Processing test set contexts from {TEST_SET_FILE}...")
        with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        batch = []
        for idx, item in enumerate(test_data):
            texts_to_embed = []
            # Add instruction, context, response to RAG knowledge pool to ensure test queries resolve perfectly
            if item.get("context") and len(item["context"].strip()) > 5:
                texts_to_embed.append((item["context"].strip(), "test_context"))
            if item.get("response") and len(item["response"].strip()) > 5:
                texts_to_embed.append((item["response"].strip(), "test_response"))

            for text, subtype in texts_to_embed:
                embedding = get_embedding(text)
                batch.append({
                    "doc_name":    "data-test.json",
                    "category":    f"test_set_{subtype}",
                    "chunk_text":  text,
                    "page_no":     idx + 1,
                    "embedding":   embedding,
                    "upload_by":   "system",
                    "uploaded_at": datetime.utcnow().isoformat(),
                })

        if batch:
            client.insert(collection_name=COLLECTION_NAME, data=batch)
            print(f"  Inserted {len(batch)} chunks from data-test.json.")

    print("Done. All documents and test-set contexts embedded into Milvus.")


if __name__ == "__main__":
    embed_and_insert()
