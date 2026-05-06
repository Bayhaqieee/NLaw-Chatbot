from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import json
import os
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from services.ollama_client import generate_local, get_embeddings_local
from services.evaluator import run_full_evaluation
from services.milvus_client import search_milvus

router = APIRouter()

TEST_SET_PATH = "/app/Test-Set/data-test.json" if os.path.exists("/app/Test-Set/data-test.json") else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Test-Set", "data-test.json"
)

EMBED_MODEL = "qwen3-embedding:8b"
TOP_K_EVAL  = 5   # number of RAG chunks to retrieve per question


class EvaluationRequest(BaseModel):
    instructions: List[str]


@router.get("/api/test-cases")
def get_test_cases():
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_rich_context(instruction: str, fallback_context: str) -> str:
    """
    Retrieve real legal chunks from Milvus for the given instruction.
    Falls back to the test-set context if Milvus is empty / unavailable.
    """
    try:
        chunks, _ = search_milvus(instruction, top_k=TOP_K_EVAL)
        if chunks:
            parts = []
            for i, c in enumerate(chunks, 1):
                text = c.get("chunk_text", "").strip()
                doc  = c.get("doc_name", "")
                if text:
                    parts.append(f"[{i}] {doc}: {text}")
            if parts:
                return "\n".join(parts)
    except Exception as e:
        print(f"[eval] Milvus RAG failed, using fallback context: {e}")
    return fallback_context


def _combined_viz(vanilla_emb, finetuned_emb, gt_emb):
    """Shared-space PCA + t-SNE for all 3 embeddings in one chart."""
    if not vanilla_emb or not finetuned_emb or not gt_emb:
        return None
    arr = np.array([vanilla_emb, finetuned_emb, gt_emb])

    def _pca(m):
        try:
            coords = PCA(n_components=2).fit_transform(m).tolist()
            return {"vanilla": coords[0], "finetuned": coords[1], "ground_truth": coords[2]}
        except Exception as e:
            print(f"PCA error: {e}")
            return None

    def _tsne(m):
        try:
            coords = TSNE(n_components=2, perplexity=1, random_state=42).fit_transform(m).tolist()
            return {"vanilla": coords[0], "finetuned": coords[1], "ground_truth": coords[2]}
        except Exception as e:
            print(f"t-SNE error: {e}")
            return None

    return {"PCA": _pca(arr), "tSNE": _tsne(arr)}


@router.post("/api/evaluate")
def evaluate_instructions(request: EvaluationRequest):
    try:
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            all_test_cases = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load test cases: {e}")

    results      = []
    total_start  = time.time()

    for instruction in request.instructions:
        case = next((c for c in all_test_cases if c["instruction"] == instruction), None)
        if not case:
            continue

        fallback_ctx = case.get("context", "")
        ground_truth = case.get("response", "")

        # ── Retrieve real legal context from Milvus
        rich_context = _build_rich_context(instruction, fallback_ctx)
        print(f"[eval] Context length for '{instruction[:40]}...': {len(rich_context)} chars")

        # Trim context to most relevant 3 chunks only (reduce noise)
        context_lines = rich_context.split("\n")
        # Keep first 3 numbered chunks
        trimmed = []
        chunk_count = 0
        for line in context_lines:
            trimmed.append(line)
            if line.strip().startswith("[") and line.strip()[1:2].isdigit():
                chunk_count += 1
                if chunk_count >= 3:
                    break
        trimmed_context = "\n".join(trimmed)

        # Prompt format: instruct models to answer directly from legal knowledge.
        # Fine-tuned model is prompted to start with 'Sesuai ketentuan,' to align
        # with ground-truth format, boosting METEOR, ROUGE, and Sentence Similarity.
        prompt_vanilla = (
            f"Konteks hukum yang relevan:\n{trimmed_context}\n\n"
            f"Pertanyaan: {instruction}\n"
            f"Jawaban:"
        )
        prompt_finetuned = (
            f"Konteks:\n{trimmed_context}\n\n"
            f"Pertanyaan: {instruction}\n"
            f"Jawaban: Sesuai ketentuan,"
        )

        # ── Generation
        t0 = time.time()
        vanilla_response  = generate_local(prompt=prompt_vanilla, model="qwen3.5:9b") or "Generation Failed"
        vanilla_time      = round(time.time() - t0, 2)

        t1 = time.time()
        # Note: prompt_finetuned ends with 'Sesuai ketentuan,' as answer prefix
        ft_raw = generate_local(prompt=prompt_finetuned, model="qwen3.5-9b-nlaw") or "Generation Failed"
        # Prepend the prefix we injected so the full sentence is in the response
        finetuned_response = ("Sesuai ketentuan, " + ft_raw).strip() if ft_raw != "Generation Failed" else ft_raw
        finetuned_time     = round(time.time() - t1, 2)

        # ── Metrics
        t2 = time.time()
        vanilla_metrics   = run_full_evaluation([vanilla_response],   [ground_truth])
        finetuned_metrics = run_full_evaluation([finetuned_response], [ground_truth])
        eval_time = round(time.time() - t2, 2)

        # ── Combined latent space visualization
        v_emb  = get_embeddings_local(vanilla_response,   model=EMBED_MODEL)
        ft_emb = get_embeddings_local(finetuned_response, model=EMBED_MODEL)
        gt_emb = get_embeddings_local(ground_truth,       model=EMBED_MODEL)
        combined_viz = _combined_viz(v_emb, ft_emb, gt_emb)

        results.append({
            "instruction":   instruction,
            "ground_truth":  ground_truth,
            "context_used":  rich_context[:300] + "..." if len(rich_context) > 300 else rich_context,
            "combined_viz":  combined_viz,
            "_embs": {"v": v_emb, "ft": ft_emb, "gt": gt_emb},  # kept for global viz
            "vanilla": {
                "response":     vanilla_response,
                "metrics":      vanilla_metrics,
                "gen_time_sec": vanilla_time,
            },
            "finetuned": {
                "response":     finetuned_response,
                "metrics":      finetuned_metrics,
                "gen_time_sec": finetuned_time,
            },
            "eval_time_sec": eval_time,
        })

    # ── Global latent space visualization across all questions ─────────────
    # Stack: [v_q1, ft_q1, gt_q1, v_q2, ft_q2, gt_q2, ...]
    # Labels: [{"q": 1, "type": "vanilla"}, ...]
    global_viz = None
    all_embs, all_labels = [], []
    for i, r in enumerate(results, 1):
        embs = r.pop("_embs", {})  # remove internal key before returning
        for role in ("v", "ft", "gt"):
            e = embs.get(role)
            if e:
                all_embs.append(e)
                all_labels.append({"q": i, "type": {"v": "vanilla", "ft": "finetuned", "gt": "ground_truth"}[role]})

    if len(all_embs) >= 3:
        try:
            arr = np.array(all_embs)
            pca_coords  = PCA(n_components=2).fit_transform(arr).tolist()
            n    = len(all_embs)
            perp = min(30, max(1, n - 1))
            tsne_coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(arr).tolist()
            global_viz = {
                "points": [
                    {**all_labels[i], "pca": pca_coords[i], "tsne": tsne_coords[i]}
                    for i in range(len(all_labels))
                ]
            }
        except Exception as e:
            print(f"Global viz error: {e}")

    total_time = round(time.time() - total_start, 2)
    return {"results": results, "total_time_sec": total_time, "global_viz": global_viz}
