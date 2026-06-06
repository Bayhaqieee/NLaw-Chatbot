# NusantaraLaw Chatbot

A self-hosted Indonesian legal AI chatbot combining QLoRA fine-tuning, RAG over verified law documents, and a 9-metric evaluation dashboard — built for BAB IV of an undergraduate thesis on Indonesian Legal AI.

> **Fine-Tuned model wins 7–8 out of 9 evaluation metrics across all 4 hyperparameter scenarios vs the vanilla baseline.**

---

## Architecture

| Component          | Technology                                      |
|--------------------|-------------------------------------------------|
| Frontend           | HTML5 + Vanilla JS + CSS3 (dual-theme glassmorphism) |
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

> [!TIP]
> **Production Hyperparameters for Chatbot**:
> Both the Vanilla and Fine-Tuned local models run via Ollama's native `/api/generate` endpoint, aligning perfectly with their instruction-response training templates. The chatbot uses a calibrated `repeat_penalty` of `1.10` and `repeat_last_n` of `128` (`CHAT_FT_OPTIONS`) to maintain grammatical fluency and eliminate word salad or repetition loops.

### Evaluation Dashboard (`localhost:3000/evaluation.html`)

1. Select test cases (or "Pilih Semua")
2. Choose a **Skenario Hyperparameter** from the dropdown:
   - **EXP-01** — Conservative (temp 0.05) — Maximum accuracy
   - **EXP-02** — Balanced (temp 0.10) — Production default
   - **EXP-03** — Explorative (temp 0.25) — Enhanced diversity
   - **EXP-04** — Creative (temp 0.40) — Maximum expressiveness
3. Click **Mulai Evaluasi**
4. Results include:
   - **Ringkasan Evaluasi** — average metric comparison, winner highlighted
   - **Kompilasi Latent Space** — all embeddings in one shared PCA/t-SNE (○ Vanilla, △ FT, □ GT)
   - **Per-Question Cards** — individual metrics + per-question latent space chart
5. If the connection drops during a long evaluation, click **Muat Hasil Terakhir** to load cached results from `eval_cache/`

> **Note**: Hyperparameter scenarios only affect the Fine-Tuned model during evaluation. The chatbot always uses fixed production parameters.

### Showcase (`localhost:3000/showcase.html`)

Static evaluation reports for thesis documentation:
- `showcase-xp1.html` through `showcase-xp4.html`
- Print-optimized with `@media print` CSS
- Mirrors the evaluation dashboard layout

---

## Evaluation Results

All experiments evaluate **50 test cases** across both Vanilla (`qwen3.5:9b`) and Fine-Tuned (`qwen3.5-9b-nlaw`) models. Metrics follow the 0–100 scale for percentage-based metrics (matching the Kaggle evaluation notebook). Lower is better for Perplexity (↓) and L2 Distance (↓).

### EXP-01 — Conservative (temp 0.05)

*Fine-Tuned won 7/9 metrics · 50 cases · Server: 6691s*

| Metric          | Vanilla  | Fine-Tuned | Winner         |
|-----------------|----------|------------|----------------|
| SacreBLEU       | 6.7901   | **9.3025** | ✅ Fine-Tuned  |
| ROUGE-L         | **0.2429** | 0.2515   | ✅ Fine-Tuned  |
| METEOR          | **0.4001** | 0.3444   | ❌ Vanilla     |
| BERTScore (F1)  | **0.9742** | 0.9728   | ❌ Vanilla     |
| Sentence Sim    | 0.6810   | **0.7618** | ✅ Fine-Tuned  |
| NLI Entailment  | 0.1129   | **0.3848** | ✅ Fine-Tuned  |
| Perplexity ↓    | 0.34507  | **0.30072** | ✅ Fine-Tuned |
| NLaw Score      | 0.6549   | **0.6993** | ✅ Fine-Tuned  |
| L2 Distance ↓   | 0.8207   | **0.7515** | ✅ Fine-Tuned  |

---

### EXP-02 — Balanced (temp 0.10)

*Fine-Tuned won 8/9 metrics · 50 cases · Server: 6157s*

| Metric          | Vanilla  | Fine-Tuned  | Winner         |
|-----------------|----------|-------------|----------------|
| SacreBLEU       | 6.1879   | **10.1060** | ✅ Fine-Tuned  |
| ROUGE-L         | 0.2319   | **0.2789**  | ✅ Fine-Tuned  |
| METEOR          | **0.3856** | 0.3710    | ❌ Vanilla     |
| BERTScore (F1)  | 0.9739   | **0.9742**  | ✅ Fine-Tuned  |
| Sentence Sim    | 0.6742   | **0.7745**  | ✅ Fine-Tuned  |
| NLI Entailment  | 0.1362   | **0.2643**  | ✅ Fine-Tuned  |
| Perplexity ↓    | 0.35067  | **0.29195** | ✅ Fine-Tuned  |
| NLaw Score      | 0.6392   | **0.7081**  | ✅ Fine-Tuned  |
| L2 Distance ↓   | 0.8386   | **0.7433**  | ✅ Fine-Tuned  |

---

### EXP-03 — Explorative (temp 0.25)

*Fine-Tuned won 7/9 metrics · 50 cases · Server: 6634s*

| Metric          | Vanilla  | Fine-Tuned  | Winner         |
|-----------------|----------|-------------|----------------|
| SacreBLEU       | 6.2635   | **9.4924**  | ✅ Fine-Tuned  |
| ROUGE-L         | 0.2392   | **0.2576**  | ✅ Fine-Tuned  |
| METEOR          | **0.3825** | 0.3542    | ❌ Vanilla     |
| BERTScore (F1)  | **0.9738** | 0.9733    | ❌ Vanilla     |
| Sentence Sim    | 0.6767   | **0.7550**  | ✅ Fine-Tuned  |
| NLI Entailment  | 0.1470   | **0.3470**  | ✅ Fine-Tuned  |
| Perplexity ↓    | 0.36185  | **0.29701** | ✅ Fine-Tuned  |
| NLaw Score      | 0.6381   | **0.7030**  | ✅ Fine-Tuned  |
| L2 Distance ↓   | 0.8402   | **0.7470**  | ✅ Fine-Tuned  |

---

### EXP-04 — Creative (temp 0.40)

*Fine-Tuned won 8/9 metrics · 50 cases · Server: 5857s*

| Metric          | Vanilla  | Fine-Tuned  | Winner         |
|-----------------|----------|-------------|----------------|
| SacreBLEU       | 6.2652   | **11.3469** | ✅ Fine-Tuned  |
| ROUGE-L         | 0.2363   | **0.2828**  | ✅ Fine-Tuned  |
| METEOR          | **0.3885** | 0.3592    | ❌ Vanilla     |
| BERTScore (F1)  | 0.9738   | **0.9740**  | ✅ Fine-Tuned  |
| Sentence Sim    | 0.6854   | **0.7220**  | ✅ Fine-Tuned  |
| NLI Entailment  | 0.1270   | **0.3835**  | ✅ Fine-Tuned  |
| Perplexity ↓    | 0.36004  | **0.29240** | ✅ Fine-Tuned  |
| NLaw Score      | 0.6400   | **0.6871**  | ✅ Fine-Tuned  |
| L2 Distance ↓   | 0.8385   | **0.7593**  | ✅ Fine-Tuned  |

---

### Cross-Scenario Summary

| Scenario       | FT Wins | Vanilla Wins | Best SacreBLEU | Best NLI Entail |
|----------------|---------|--------------|----------------|-----------------|
| EXP-01 Conservative | 7/9 | 2/9 | 9.30 | 0.3848 |
| EXP-02 Balanced     | 8/9 | 1/9 | 10.11 | 0.2643 |
| EXP-03 Explorative  | 7/9 | 2/9 | 9.49 | 0.3470 |
| EXP-04 Creative     | 8/9 | 1/9 | 11.35 | 0.3835 |

> Full per-question results in [`Idea/Result/Website/`](./Idea/Result/Website/). Hyperparameter rationale in [`Idea/rag_hyperparameter.md`](./Idea/rag_hyperparameter.md).

---

## Maintenance

```bash
# Rebuild and start main services (excluding vector-visualizer)
docker-compose up -d --build backend frontend

# Start the optional 3D Vector Visualizer (runs separately to save resources)
docker-compose --profile visualizer up -d --build

# Check live backend logs
.\check_logs.ps1
# or: curl http://localhost:8000/api/logs

# Reset Milvus collection (run using the virtual environment to avoid ModuleNotFoundError)
.venv\Scripts\python.exe milvus/init_collection.py

# Free up Docker storage
docker system prune -f
```

---

## Project Structure

```
NusantaraLaw-Chatbot/
├── docker-compose.yml
├── .env / .env.example
├── README.md
├── download_nli.py
├── check_logs.ps1
├── generate_showcase.py
├── eval_cache/                  ← Persisted evaluation results (host-mounted)
├── Idea/
│   ├── idea.md                  ← Master technical document
│   ├── Requirement_Specs.md
│   ├── Design_Specs.md
│   ├── Implementation_Specs.md
│   ├── Technical_Specs.md
│   ├── rag_hyperparameter.md    ← Hyperparameter scenario rationale
│   └── Result/Website/          ← EXP-01 to EXP-04 full results (.txt/.md)
├── backend/
│   ├── routes/   (chat, upload, evaluation, documents, health)
│   └── services/ (ollama_client, milvus_client, searxng_client,
│                   rag_pipeline, hf_inference, evaluator, pdf_parser)
├── frontend/     (index.html, style.css, app.js, evaluation.html/js,
│                   showcase.html, showcase-xp[1-4].html)
├── milvus/       (init_collection.py, embed_documents.py)
├── Model/        (nli-deberta-v3-base/, roberta-large/)
├── UU-PDP/       (Indonesian law PDFs)
└── Test-Set/     (data-test.json — 50 golden test cases)
```
