# NusantaraLaw Chatbot — Idea & Execution Blueprint

> **Status**: Planning → Ready to Build
> **Model**: `bayhaqieee/qwen3.5-9b-nlaw-gguf` (Fine-Tuned, Cloud Inference via HuggingFace)
> **Scope**: Included in BAB IV Skripsi (Pengembangan Perangkat Lunak)

---

## 1. Vision

A fully self-hosted legal AI chatbot that answers Indonesian law questions with zero
hallucination by combining:
- A fine-tuned LLM that *understands* Indonesian legal language
- RAG grounded in verified law documents (UU PDP, UU ITE, etc.)
- Real-time web augmentation via SearXNG for up-to-date legal references
- Token-efficient prompt engineering using TOON Format
- Containerized deployment for reproducibility (Docker Compose)

---

## 2. Tech Stack

| Layer              | Technology                                | Purpose                                  |
|--------------------|-------------------------------------------|------------------------------------------|
| **Frontend**       | HTML + Vanilla JS (or Next.js optional)   | Chat UI, document upload, session mgmt   |
| **Backend API**    | FastAPI (Python 3.11)                     | RAG pipeline, LLM routing, file handling |
| **Vector DB**      | Milvus v2.4 (Docker)                      | Store + search document embeddings       |
| **Embedding Model**| `paraphrase-multilingual-mpnet-base-v2`   | Convert text to vectors (same as eval)   |
| **LLM Inference**  | HuggingFace Inference API (Cloud)         | Run `bayhaqieee/qwen3.5-9b-nlaw-gguf`   |
| **Web Search**     | SearXNG (Docker, self-hosted)             | Augment answers with real-time web info  |
| **Prompt Format**  | TOON (Token-Oriented Object Notation)     | Efficient context serialization to LLM   |
| **Orchestration**  | Docker Compose                            | Single-command deployment                |
| **PDF Parsing**    | PyMuPDF (fitz)                            | Extract text from law documents          |

---

## 3. System Architecture

```
USER BROWSER
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│                   DOCKER COMPOSE NETWORK                    │
│                                                            │
│  ┌────────────┐     ┌──────────────────────────────────┐  │
│  │  Frontend  │────▶│        FastAPI Backend            │  │
│  │  :3000     │     │        :8000                      │  │
│  └────────────┘     └───┬──────────┬──────────┬────────┘  │
│                         │          │          │            │
│                    ┌────▼───┐ ┌────▼───┐ ┌───▼────────┐  │
│                    │ Milvus │ │SearXNG │ │   HF API   │  │
│                    │ :19530 │ │ :8080  │ │  (Cloud)   │  │
│                    └────────┘ └────────┘ └────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 4. RAG Pipeline — Step by Step

```
User Question
     │
     ▼
1. EMBED QUESTION
   └─ paraphrase-multilingual-mpnet-base-v2 → query_vector

     │
     ▼
2. MILVUS SEARCH  (Top-K = 5 chunks)
   └─ Search nusantara_law collection
   └─ Returns: [pasal_text, sumber_dokumen, page_no]

     │
     ▼
3. SEARXNG AUGMENT  (if Milvus score < threshold OR user toggles web)
   └─ Query: "{user_question} site:hukumonline.com OR site:jdih.go.id"
   └─ Returns: top 3 web snippets with URLs

     │
     ▼
4. TOON FORMAT BUILDER
   └─ Serialize RAG context into compact TOON format:
      legal_context[5]{no,pasal,isi,sumber}:
      1,Pasal 1 Ayat 1,Perlindungan Data...,UU 27/2022
      2,Pasal 65,Setiap Orang...,UU 27/2022
      ...
   └─ web_context[3]{no,snippet,url}:
      1,Berdasarkan putusan MA...,hukumonline.com/...

     │
     ▼
5. PROMPT ASSEMBLY
   └─ System: "Anda adalah pakar hukum Indonesia yang presisi..."
   └─ Context: [TOON formatted chunks]
   └─ Question: user input

     │
     ▼
6. HF INFERENCE API  →  bayhaqieee/qwen3.5-9b-nlaw-gguf
   └─ max_new_tokens: 512
   └─ temperature: 0.1 (low for factual legal answers)
   └─ repetition_penalty: 1.15

     │
     ▼
7. RESPONSE  →  streamed back to Frontend
   └─ Sources cited (Pasal, UU, URL)
   └─ Confidence indicator based on retrieval score
```

---

## 5. Knowledge Base Design (Milvus)

### Collection: `nusantara_law`

| Field        | Type         | Description                          |
|--------------|--------------|--------------------------------------|
| `id`         | INT64 (PK)   | Auto-generated unique ID             |
| `doc_name`   | VARCHAR(256) | Source document filename             |
| `category`   | VARCHAR(64)  | e.g., "UU_PDP", "UU_ITE", "PERPRES" |
| `chunk_text` | VARCHAR(4096)| Actual text chunk (≤500 words)       |
| `page_no`    | INT32        | Original page number in PDF          |
| `embedding`  | FLOAT_VECTOR | 768-dim from mpnet-base-v2           |
| `upload_by`  | VARCHAR(64)  | "system" or user identifier          |
| `uploaded_at`| VARCHAR(32)  | ISO timestamp                        |

### Pre-loaded Documents (System Default):
```
UU-PDP/
├── UU Nomor 27 Tahun 2022.pdf     ← Primary KnowledgeBase
├── UU Nomor 11 Tahun 2008.pdf     ← UU ITE original
├── UU Nomor 1 Tahun 2024.pdf      ← UU ITE amendment
├── UU Nomor 19 Tahun 2016.pdf     ← UU ITE amendment 2016
├── perpres-82-2023.pdf            ← Supporting regulation
└── perpres-83-2025.pdf            ← Supporting regulation
```

### Real-Time User Upload Feature:
- Users can upload additional PDF documents via the chat UI
- Backend: PyMuPDF extracts text → chunked (500 words, 50-word overlap)
- Each chunk embedded → inserted into Milvus `nusantara_law` collection
- Upload tagged with `upload_by: "user"` + timestamp
- Documents appear immediately in the next RAG query
- Admin panel (optional): view/delete uploaded documents

---

## 6. TOON Format — Implementation Detail

### Why TOON Here?
Each RAG context chunk in standard JSON would look like:
```json
[
  {"no": 1, "pasal": "Pasal 1 Ayat 1", "isi": "...", "sumber": "UU 27/2022"},
  {"no": 2, "pasal": "Pasal 65", "isi": "...", "sumber": "UU 27/2022"}
]
```
With 5 chunks, that's ~400–600 extra tokens just in JSON syntax overhead.

TOON equivalent:
```
legal_context[5]{no,pasal,isi,sumber}:
1,Pasal 1 Ayat 1,...,UU 27/2022
2,Pasal 65,...,UU 27/2022
```
**Savings: ~30–50% fewer tokens**, critical when using free-tier HF Inference API with token limits.

### Python Implementation (`toon_formatter.py`):
```python
def format_legal_context_toon(chunks: list[dict]) -> str:
    header = f"legal_context[{len(chunks)}]{{no,pasal,isi,sumber}}:"
    rows = [
        f"{i+1},{c.get('pasal','-')},{c['chunk_text'][:200].replace(',','；')},{c['category']}"
        for i, c in enumerate(chunks)
    ]
    return header + "\n" + "\n".join(rows)

def format_web_context_toon(web_results: list[dict]) -> str:
    if not web_results:
        return ""
    header = f"web_context[{len(web_results)}]{{no,snippet,url}}:"
    rows = [
        f"{i+1},{r['snippet'][:150].replace(',','；')},{r['url']}"
        for i, r in enumerate(web_results)
    ]
    return header + "\n" + "\n".join(rows)
```

---

## 7. SearXNG Integration

### Configuration (`searxng/settings.yml`):
```yaml
search:
  safe_search: 0
  default_lang: "id"
engines:
  - name: google
    engine: google
    language: id
  - name: hukumonline
    engine: xpath
    url: https://www.hukumonline.com/search/?q={query}
```

### When SearXNG is Triggered:
1. Milvus similarity score < 0.70 (low confidence in local KB)
2. User explicitly toggles "Search Web" button
3. Question contains keywords like "terbaru", "2024", "2025", "revisi"

### Query Strategy for Legal Domain:
```python
search_query = f"{user_question} site:hukumonline.com OR site:jdih.go.id OR site:peraturan.go.id"
```

---

## 8. Project Structure

```
nusantara-law-chatbot/
│
├── docker-compose.yml              ← Single-command deployment
├── .env                            ← HF_API_TOKEN, config vars
│
├── backend/
│   ├── Dockerfile
│   ├── main.py                     ← FastAPI app entry
│   ├── routes/
│   │   ├── chat.py                 ← POST /api/chat
│   │   ├── upload.py               ← POST /api/upload
│   │   └── documents.py            ← GET /api/documents
│   ├── services/
│   │   ├── rag_pipeline.py         ← Main RAG orchestrator
│   │   ├── milvus_client.py        ← Milvus operations
│   │   ├── searxng_client.py       ← SearXNG search
│   │   ├── hf_inference.py         ← HuggingFace API calls
│   │   ├── pdf_parser.py           ← PyMuPDF text extraction
│   │   └── toon_formatter.py       ← TOON serialization
│   ├── models/
│   │   └── schemas.py              ← Pydantic request/response models
│   └── requirements.txt
│
├── frontend/
│   ├── Dockerfile
│   ├── index.html                  ← Chat interface
│   ├── style.css
│   └── app.js                      ← Fetch API calls, streaming UI
│
├── milvus/
│   ├── init_collection.py          ← Create collection + load default docs
│   └── embed_documents.py          ← Batch embed PDFs
│
└── searxng/
    └── settings.yml                ← SearXNG engine config
```

---

## 9. docker-compose.yml (Full)

```yaml
version: '3.9'

services:
  # ─── Frontend ─────────────────────────────────────────────
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  # ─── Backend API ──────────────────────────────────────────
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - HF_API_TOKEN=${HF_API_TOKEN}
      - HF_MODEL_ID=bayhaqieee/qwen3.5-9b-nlaw-gguf
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
      - SEARXNG_URL=http://searxng:8080
      - COLLECTION_NAME=nusantara_law
    volumes:
      - ./UU-PDP:/app/documents:ro   # Mount law documents read-only
    depends_on:
      - milvus-standalone
      - searxng

  # ─── Milvus Vector Database ───────────────────────────────
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus-standalone:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio

  # ─── SearXNG (Web Search) ──────────────────────────────────
  searxng:
    image: searxng/searxng:latest
    ports:
      - "8080:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

---

## 10. Environment Variables (`.env`)

```env
# HuggingFace
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_MODEL_ID=bayhaqieee/qwen3.5-9b-nlaw-gguf

# Milvus
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530
COLLECTION_NAME=nusantara_law
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

# SearXNG
SEARXNG_URL=http://searxng:8080
SEARXNG_THRESHOLD=0.70

# RAG Config
TOP_K_CHUNKS=5
MAX_NEW_TOKENS=512
TEMPERATURE=0.1
```

---

## 11. API Endpoints (FastAPI)

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| POST   | `/api/chat`        | Main chat endpoint (RAG + LLM)           |
| POST   | `/api/upload`      | Upload PDF → embed → insert to Milvus   |
| GET    | `/api/documents`   | List all embedded documents              |
| DELETE | `/api/documents/{id}` | Remove document from Milvus           |
| GET    | `/api/health`      | Health check for all services            |

### Request / Response Schema:

```python
# POST /api/chat
class ChatRequest(BaseModel):
    question: str
    use_web_search: bool = False
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    web_results: list[WebResult]
    retrieval_score: float
    toon_tokens_saved: int  # show token savings

# POST /api/upload
class UploadResponse(BaseModel):
    filename: str
    chunks_embedded: int
    status: str
```

---

## 12. Build Order (Step-by-Step Execution)

```
Phase 1: Infrastructure Setup
  [1] Write docker-compose.yml
  [2] Write searxng/settings.yml
  [3] docker-compose up etcd minio milvus-standalone searxng
  [4] Verify: Milvus health → http://localhost:9091/healthz
  [5] Verify: SearXNG → http://localhost:8080

Phase 2: Document Pipeline
  [6] Write milvus/init_collection.py → create schema
  [7] Write milvus/embed_documents.py → embed UU-PDP PDFs
  [8] Run: python milvus/embed_documents.py
  [9] Verify: collection has N chunks in Milvus

Phase 3: Backend
  [10] Write backend/services/toon_formatter.py
  [11] Write backend/services/milvus_client.py
  [12] Write backend/services/searxng_client.py
  [13] Write backend/services/hf_inference.py
  [14] Write backend/services/rag_pipeline.py
  [15] Write backend/routes/chat.py, upload.py, documents.py
  [16] Write backend/main.py
  [17] docker-compose up backend
  [18] Test: curl POST /api/chat with sample legal question

Phase 4: Frontend
  [19] Build chat UI (index.html + app.js)
  [20] Add document upload panel
  [21] Add source citation display
  [22] docker-compose up frontend
  [23] End-to-end test via browser

Phase 5: Integration Testing
  [24] Test full pipeline: Question → Milvus → SearXNG → TOON → LLM → Answer
  [25] Test real-time upload: Upload new PDF → query immediately
  [26] Verify TOON token savings vs JSON baseline
  [27] Screenshot/record for BAB IV thesis documentation
```

---

## 13. Key Design Decisions & Rationale

| Decision | Reason |
|---|---|
| **HuggingFace Inference API** | Fine-tuned model already pushed; no local GPU required; free tier sufficient for demo/thesis |
| **Milvus over FAISS** | Milvus is production-grade, persistent, and supports real-time insert (FAISS is in-memory only) |
| **TOON Format** | Free-tier HF API has token limits; TOON reduces context tokens by 30–50%, fitting more legal context |
| **SearXNG self-hosted** | Privacy-preserving; no API key needed; can configure Indonesian legal site prioritization |
| **paraphrase-multilingual-mpnet-base-v2** | Same embedding model used in the eval notebook → semantic consistency |
| **PyMuPDF** | Best-in-class PDF parsing for Indonesian law documents (handles complex layouts) |
| **500-word chunks, 50-word overlap** | Standard RAG chunking; overlap prevents context loss at boundaries |

---

## 14. Thesis Integration (BAB IV)

This chatbot system is documented in **BAB IV: Pengembangan Perangkat Lunak**:

- **4.1** Arsitektur Sistem Chatbot Hukum Lokal
- **4.2** Konfigurasi Docker Compose dan Infrastruktur
- **4.3** Implementasi Milvus Vector Database dan Pipeline Embedding Dokumen
- **4.4** Implementasi Augmentasi Pencarian Web via SearXNG
- **4.5** Optimasi Prompt dengan Format TOON
- **4.6** Integrasi Model Fine-Tuned via HuggingFace Inference API
- **4.7** Antarmuka Pengguna dan Fitur Upload Dokumen Real-Time
- **4.8** Pengujian End-to-End

---

## 15. Future Extensions (BAB VI — Saran)

- [ ] Implement TurboQuant KV-cache compression when switching to local vLLM
- [ ] Add hybrid search (BM25 sparse + Milvus dense) like Pratama et al. (2025)
- [ ] Extend Knowledge Base to court decisions (Putusan Mahkamah Agung)
- [ ] Add multi-turn conversation memory (Redis-based)
- [ ] Deploy to VPS with Nginx reverse proxy for public access
