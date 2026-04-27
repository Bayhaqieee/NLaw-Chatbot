# NusantaraLaw Chatbot

A self-hosted legal AI chatbot that answers Indonesian law questions without hallucination, built specifically for integration into BAB IV of the software development thesis.

## Architecture & Tech Stack

- **Frontend**: HTML + Vanilla JS + CSS (Lightweight, responsive)
- **Backend API**: FastAPI (Python 3.11)
- **Vector Database**: Milvus v2.4 (Docker)
- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2`
- **LLM Inference**: HuggingFace Inference API (`bayhaqieee/qwen3.5-9b-nlaw-gguf`)
- **Web Search**: SearXNG (Docker, self-hosted)
- **Prompt Format**: TOON (Token-Oriented Object Notation)

---

## Step-by-Step Execution Guide

### Prerequisites
- Docker and Docker Compose installed
- Python 3.11 installed locally (for testing ingestion)
- HuggingFace API Token (currently left empty in `.env` for privacy, but will be needed to query the LLM)

### Step 1: Infrastructure Setup
1. Open terminal and navigate to the project root `NusantaraLaw-Chatbot`.
2. Ensure you have the required environment variables in the `.env` file. You can leave `HF_API_TOKEN` empty for now, but you **must** add a valid token before asking the chatbot questions.
3. Start the Docker containers:
   ```bash
   docker-compose up -d --build
   ```
4. This command will start:
   - Frontend (`localhost:3000`)
   - Backend API (`localhost:8000`)
   - Milvus Vector DB (`localhost:19530`) & health interface (`localhost:9091`)
   - SearXNG (`localhost:8080`)

### Step 2: Document Ingestion (Local Setup for Ingestion Scripts)
By default, the database is empty. You need to ingest your Indonesian legal PDF documents into Milvus.

1. Place your legal PDF files inside the `UU-PDP/` folder.
2. Open a **new terminal** at the project root and create a Python virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```bash
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # Windows CMD
   .\.venv\Scripts\activate.bat
   ```
4. Install the ingestion script dependencies:
   ```bash
   pip install -r milvus/requirements.txt
   ```
5. Make sure Docker containers are already running (`docker-compose up -d`), then initialize the Milvus collection schema:
   ```bash
   python milvus/init_collection.py
   ```
6. Run the embedding script to chunk and embed all PDFs from the `UU-PDP/` folder into Milvus:
   ```bash
   python milvus/embed_documents.py
   ```
   *(Alternatively, skip steps 5 & 6 and use the **Ingest Document** button on the web UI instead!)*

### Step 3: API Verification
Before using the frontend, ensure the APIs are healthy:
- Check Backend: `curl http://localhost:8000/api/health`
- Check SearXNG: `curl http://localhost:8080`
- Check Milvus: `curl http://localhost:9091/healthz`

### Step 4: Interact with the Chatbot
1. Open your browser and navigate to `http://localhost:3000`.
2. **Uploading**: On the left sidebar, select a PDF document (like UU 27/2022) and click "Ingest Document". Wait for the success message.
3. **Chatting**: Ask a legal question in the main chat area (e.g., *"Apa definisi data pribadi menurut UU PDP?"*).
4. **Web Search**: Toggle "Gunakan Web Search" if you want the chatbot to cross-reference *hukumonline.com* for the latest articles.

---

## Verification Plan

*(As requested, this plan helps to test and validate the entire integration)*

### Automated Testing
- [ ] Run `docker-compose ps` to ensure all containers (`frontend`, `backend`, `milvus-standalone`, `etcd`, `minio`, `searxng`) are running.
- [ ] Run health check: `curl http://localhost:8000/api/health` -> Expect `{"status": "ok"}`

### Manual End-to-End Verification
1. **Document Ingestion Test**:
   - Navigate to `localhost:3000`.
   - Upload a test PDF file using the left sidebar.
   - Verify that the status turns green indicating chunks were successfully embedded.
2. **RAG Flow Test (Local Vector Search)**:
   - Ask a question derived from the uploaded PDF.
   - Verify the chatbot answers correctly, citing the source document and page number.
   - Verify the system notes that TOON format saved tokens.
3. **SearXNG Web Fallback Test**:
   - Turn ON the "Gunakan Web Search" toggle.
   - Ask about a very recent legal development (e.g., "Berita terbaru revisi UU ITE 2024").
   - Verify the chatbot response includes URLs from `hukumonline.com` or `jdih.go.id` in the source citations.
4. **Error Handling Test**:
   - Disconnect internet or provide an invalid HF Token.
   - Verify the UI gracefully catches the error and displays a clear message to the user.
