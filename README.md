# NusantaraLaw Chatbot

A self-hosted Indonesian legal AI chatbot that answers law questions with minimal hallucination. Combines a QLoRA fine-tuned LLM (`qwen3.5-9b-nlaw`) with Retrieval-Augmented Generation (RAG) over verified Indonesian law documents, evaluated against an 8-metric benchmark.

> Built for BAB IV (Pengembangan Perangkat Lunak) of the undergraduate thesis on Indonesian Legal AI.

---

## Architecture & Tech Stack

| Component         | Technology                                      |
|-------------------|------------------------------------------------|
| **Frontend**      | HTML5 + Vanilla JS + CSS3 (dark glassmorphism) |
| **Backend API**   | FastAPI 0.111 + Uvicorn (Python 3.11)          |
| **Vector DB**     | Milvus v2.4 (Docker)                           |
| **Embedding**     | `qwen3-embedding:8b` via Ollama (4096-dim)     |
| **LLM Inference** | Ollama local — `qwen3.5:9b` + `qwen3.5-9b-nlaw` |
| **Web Search**    | SearXNG (self-hosted Docker)                   |
| **NLI Model**     | `nli-deberta-v3-base` (local weights)          |
| **BERTScore**     | `roberta-large` (local weights)                |
| **Sentence Sim**  | `paraphrase-multilingual:278m-mpnet-base-v2`   |

For full system documentation, architecture diagrams, hyperparameters, and design decisions see [`idea.md`](./idea.md).

---

## Prerequisites

- **Docker + Docker Compose** installed
- **Ollama** installed locally with these models pulled:
  ```bash
  ollama pull qwen3.5:9b
  ollama pull qwen3.5-9b-nlaw        # your fine-tuned model
  ollama pull qwen3-embedding:8b
  ollama pull paraphrase-multilingual:278m-mpnet-base-v2-fp16
  ```
- **Python 3.11** (for running ingestion scripts locally)

---

## Step-by-Step Execution Guide

### Step 1: Clone & Configure

```bash
git clone <repo-url>
cd NusantaraLaw-Chatbot
```

Verify `.env` contains:
```env
OLLAMA_HOST=http://host.docker.internal:11434
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530
COLLECTION_NAME=nusantara_law
SEARXNG_URL=http://searxng:8080
```

### Step 2: Download Local Model Weights

Run this **before** building Docker to pre-download NLI + RoBERTa weights into `./Model/`:

```bash
python download_nli.py
```

This places `nli-deberta-v3-base` and `roberta-large` into `./Model/`, which are bind-mounted into the container at `/app/Model/`.

### Step 3: Start All Services

```bash
docker-compose up -d --build
```

This starts:
| Service             | URL                        |
|---------------------|----------------------------|
| Frontend (nginx)    | `http://localhost:3000`    |
| Backend (FastAPI)   | `http://localhost:8000`    |
| Milvus              | `localhost:19530`          |
| Milvus health       | `http://localhost:9091/healthz` |
| SearXNG             | `http://localhost:8080`    |

### Step 4: Ingest Law Documents

Place your Indonesian law PDFs in `UU-PDP/`, then:

```bash
# Create Python virtual environment (first time only)
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate           # macOS/Linux

# Install ingestion dependencies
pip install -r milvus/requirements.txt

# Initialize Milvus collection (4096-dim IVF_FLAT index)
python milvus/init_collection.py

# Embed all PDFs in UU-PDP/ into Milvus
python milvus/embed_documents.py
```

### Step 5: Verify Health

```bash
curl http://localhost:8000/api/health
# Expected: {"status":"ok","milvus":"ok","ollama":"ok","searxng":"ok"}
```

### Step 6: Use the Application

1. Open `http://localhost:3000`
2. **Chat**: Ask a legal question (e.g., *"Apa kewajiban pengendali data menurut UU 27/2022?"*)
3. **Web Search**: Toggle "Gunakan Web Search" for real-time augmentation
4. **Upload**: Upload additional law PDFs via the left sidebar
5. **Evaluate**: Click "Evaluasi" in the navbar to open the evaluation dashboard

---

## Evaluation Dashboard

Located at `http://localhost:3000/evaluation.html`.

### How it works:
1. Select test cases from the left sidebar (or "Pilih Semua")
2. Click "Mulai Evaluasi"
3. Both `qwen3.5:9b` (Vanilla) and `qwen3.5-9b-nlaw` (Fine-Tuned) generate answers using live Milvus RAG context
4. Results are evaluated across **8 metrics** in 3 layers:

| Layer        | Metrics                                         |
|--------------|--------------------------------------------------|
| Semantic     | SacreBLEU, ROUGE-L, METEOR                      |
| Sequential   | BERTScore (F1), Sentence Similarity, NLI Entailment |
| Latent Space | NLaw Score (Cosine), L2 Distance                |

### Output:
- **Ringkasan Evaluasi**: Side-by-side average scores for all metrics, winner highlighted in gold
- **Kompilasi Latent Space**: Global PCA + t-SNE chart of ALL embeddings projected in one shared space
- **Per-Question Cards**: Individual metric scores + per-question latent space scatter plot (Vanilla ○, Fine-Tuned △, Ground Truth □)
- **Timing**: Per-question generation time + total evaluation duration

---

## Maintenance

### Check logs
```powershell
.\check_logs.ps1
```
Or via API: `http://localhost:8000/api/logs`

### Rebuild after code changes
```bash
docker-compose up -d --build backend    # backend only
docker-compose up -d --build frontend   # frontend only
docker-compose up -d --build            # everything
```

### Clear vector database (start fresh)
```bash
python milvus/init_collection.py   # drops + recreates collection
```

### Free up Docker storage
```bash
docker system prune -f              # remove dangling images/containers
docker system prune -a --volumes -f # WARNING: also removes volumes
```

---

## Project Structure

```
NusantaraLaw-Chatbot/
├── docker-compose.yml
├── .env
├── idea.md                      ← Full technical documentation
├── README.md                    ← This file
├── download_nli.py              ← Pre-downloads model weights
├── check_logs.ps1               ← Log monitoring utility
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   ├── routes/
│   │   ├── chat.py
│   │   ├── upload.py
│   │   ├── evaluation.py        ← /api/evaluate with RAG + 8-metric eval
│   │   ├── documents.py
│   │   └── health.py
│   └── services/
│       ├── ollama_client.py     ← LLM generation + embeddings via Ollama
│       ├── milvus_client.py     ← Vector search + CRUD
│       ├── searxng_client.py    ← Web search augmentation
│       ├── rag_pipeline.py      ← RAG orchestration
│       ├── evaluator.py         ← 8-metric evaluation (singleton models)
│       └── pdf_parser.py        ← PyMuPDF chunking
├── frontend/
│   ├── Dockerfile
│   ├── index.html               ← Chat UI
│   ├── style.css
│   ├── app.js
│   ├── evaluation.html          ← Evaluation dashboard
│   └── evaluation.js            ← Charts + summary + global latent viz
├── milvus/
│   ├── init_collection.py
│   └── embed_documents.py
├── Model/
│   ├── nli-deberta-v3-base/     ← Local NLI weights
│   └── roberta-large/           ← Local BERTScore weights
├── UU-PDP/                      ← Law document PDFs
└── Test-Set/
    └── data-test.json           ← Golden evaluation test cases
```

---

## Validated Evaluation Results

Fine-Tuned model (`qwen3.5-9b-nlaw`) vs Vanilla (`qwen3.5:9b`):

| Metric          | Vanilla | Fine-Tuned | Winner         |
|-----------------|---------|------------|----------------|
| SacreBLEU       | 5.27    | **7.31**   | ✅ Fine-Tuned   |
| ROUGE-L         | 0.155   | **0.178**  | ✅ Fine-Tuned   |
| METEOR          | 0.286   | **0.300**  | ✅ Fine-Tuned   |
| BERTScore (F1)  | **0.970**| 0.969     | Vanilla        |
| Sentence Sim    | 0.546   | **0.753**  | ✅ Fine-Tuned   |
| NLI Entailment  | -0.082  | **+2.765** | ✅ Fine-Tuned   |
| NLaw Score      | 0.665   | **0.736**  | ✅ Fine-Tuned   |
| L2 Distance     | 0.817   | **0.707**  | ✅ Fine-Tuned   |
| **Overall**     |         | **7/8**    | **Fine-Tuned** |
