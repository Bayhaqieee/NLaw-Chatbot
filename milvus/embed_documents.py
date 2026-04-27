import os
import fitz
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nusantara_law")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UU-PDP")

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

def embed_and_insert():
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    print(f"Connecting to Milvus at {uri}...")
    client = MilvusClient(uri=uri)

    print("Loading embedding model (paraphrase-multilingual-mpnet-base-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    if not os.path.exists(DOCS_DIR):
        print(f"Directory {DOCS_DIR} not found. Place PDFs in UU-PDP/ folder.")
        return

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {DOCS_DIR}.")
        return

    for filename in pdf_files:
        filepath = os.path.join(DOCS_DIR, filename)
        print(f"Processing {filename}...")
        pages_data = extract_text_from_pdf(filepath)
        batch = []

        for page_data in pages_data:
            for chunk in chunk_text(page_data["text"]):
                if len(chunk) < 10:
                    continue
                embedding = model.encode(chunk).tolist()
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

    print("Done. All documents embedded into Milvus.")

if __name__ == "__main__":
    embed_and_insert()
