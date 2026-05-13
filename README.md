# NusantaraLaw Chatbot

A self-hosted Indonesian legal AI chatbot combining QLoRA fine-tuning, RAG over verified law documents, and a 9-metric evaluation dashboard — built for BAB IV of an undergraduate thesis on Indonesian Legal AI.

> **Fine-Tuned model wins 8/9 evaluation metrics vs the vanilla baseline.**

---

## Architecture

| Component          | Technology                                      |
|--------------------|-------------------------------------------------|
| Frontend           | HTML5 + Vanilla JS + CSS3 (dark glassmorphism)  |
| Backend API        | FastAPI 0.111 + Uvicorn (Python 3.11)           |
| Vector DB          | Milvus v2.4 (Docker)                            |
| Embedding          | `qwen3-embedding:8b` via Ollama (4096-dim)      |
| LLM — Local        | Ollama → `qwen3.5:9b` + `qwen3.5-9b-nlaw`      |
| LLM — Cloud        | HuggingFace Inference API (optional toggle)     |
| Web Search         | SearXNG self-hosted (5 s hard timeout)          |
| NLI / BERTScore / PPPL | `nli-deberta-v3-base` + `roberta-large` (local) |

📖 **Full documentation**: [`Idea/idea.md`](./Idea/idea.md) | [`Idea/Technical_Specs.md`](./Idea/Technical_Specs.md)

---

## Prerequisites

- **Docker + Docker Compose**
- **Ollama** with models pulled:
  ```bash
  ollama pull qwen3.5:9b
  ollama pull qwen3.5-9b-nlaw
  ollama pull qwen3-embedding:8b
  ollama pull paraphrase-multilingual:278m-mpnet-base-v2-fp16
  ```
- **Python 3.11** (for ingestion scripts)

---

## Quick Start

### 1. Clone & configure
```bash
git clone <repo-url>
cd NusantaraLaw-Chatbot
cp .env.example .env
# Edit .env — set OLLAMA_HOST, HF_API_TOKEN (optional)
```

### 2. Download local model weights
```bash
python download_nli.py
# Downloads nli-deberta-v3-base and roberta-large into ./Model/
```

### 3. Start all services
```bash
docker-compose up -d --build
```

| Service        | URL                              |
|----------------|----------------------------------|
| Chat UI        | http://localhost:3000            |
| Backend API    | http://localhost:8000            |
| Milvus health  | http://localhost:9091/healthz    |
| SearXNG        | http://localhost:8080            |

### 4. Ingest law documents
Place PDF files in `UU-PDP/`, then:
```bash
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r milvus/requirements.txt
python milvus/init_collection.py      # create 4096-dim collection
python milvus/embed_documents.py      # embed all PDFs
```

### 5. Verify
```bash
curl http://localhost:8000/api/health
# → {"status":"ok","milvus":"ok","ollama":"ok","searxng":"ok"}
```

---

## Using the Application

### Chat (`localhost:3000`)

**Settings sidebar:**
| Control | Function |
|---|---|
| **Aktifkan Web Search** | SearXNG real-time augmentation (5 s timeout, non-blocking) |
| **Gunakan HF Inference API** | Toggle between `● Lokal` (Ollama, default) and `● HuggingFace` (cloud API) |
| **Model LLM** | Switch between Fine-Tuned (`qwen3.5-9b-nlaw`) and Vanilla (`qwen3.5:9b`) |

Responses show a model badge: **Fine-Tuned** (gold) · **Vanilla** (grey) · **HuggingFace** (blue).

### Evaluation Dashboard (`localhost:3000/evaluation.html`)

1. Select test cases (or "Pilih Semua")
2. Click **Mulai Evaluasi**
3. Results include:
   - **Ringkasan Evaluasi** — average metric comparison, winner highlighted
   - **Kompilasi Latent Space** — all embeddings in one shared PCA/t-SNE (○ Vanilla, △ FT, □ GT)
   - **Per-Question Cards** — individual metrics + per-question latent space chart

---

## Maintenance

```bash
# Rebuild after code changes
docker-compose up -d --build backend frontend

# Check logs
.\check_logs.ps1
# or: curl http://localhost:8000/api/logs

# Reset Milvus (wipe + recreate collection)
python milvus/init_collection.py

# Free up Docker storage
docker system prune -f
```

---

## Evaluation Results

| Metric          | Vanilla | Fine-Tuned | Winner         |
|-----------------|---------|------------|----------------|
| SacreBLEU       | 5.27    | **7.31**   | ✅ Fine-Tuned   |
| ROUGE-L         | 0.155   | **0.178**  | ✅ Fine-Tuned   |
| METEOR          | 0.286   | **0.300**  | ✅ Fine-Tuned   |
| BERTScore (F1)  | **0.970** | 0.969   | Vanilla        |
| Sentence Sim    | 0.546   | **0.753**  | ✅ Fine-Tuned   |
| NLI Entailment  | -0.082  | **+2.765** | ✅ Fine-Tuned   |
| Perplexity      | 1.340   | **1.120**  | ✅ Fine-Tuned   |
| NLaw Score      | 0.665   | **0.736**  | ✅ Fine-Tuned   |
| L2 Distance     | 0.817   | **0.707**  | ✅ Fine-Tuned   |
| **Overall**     |         | **8 / 9**  | **Fine-Tuned** |

---

## Project Structure

```
NusantaraLaw-Chatbot/
├── docker-compose.yml
├── .env / .env.example
├── README.md
├── download_nli.py
├── check_logs.ps1
├── Idea/
│   ├── idea.md                  ← Master technical document
│   ├── Requirement_Specs.md
│   ├── Design_Specs.md
│   ├── Implementation_Specs.md
│   └── Technical_Specs.md
├── backend/
│   ├── routes/   (chat, upload, evaluation, documents, health)
│   └── services/ (ollama_client, milvus_client, searxng_client,
│                   rag_pipeline, hf_inference, evaluator, pdf_parser)
├── frontend/     (index.html, style.css, app.js, evaluation.html/js)
├── milvus/       (init_collection.py, embed_documents.py)
├── Model/        (nli-deberta-v3-base/, roberta-large/)
├── UU-PDP/       (Indonesian law PDFs)
└── Test-Set/     (data-test.json)
```
