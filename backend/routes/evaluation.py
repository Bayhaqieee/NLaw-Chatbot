from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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

EVAL_CACHE_PATH = "/app/eval_results_cache.json"

TEST_SET_PATH = "/app/Test-Set/data-test.json" if os.path.exists("/app/Test-Set/data-test.json") else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Test-Set", "data-test.json"
)

EMBED_MODEL      = "qwen3-embedding:8b"
TOP_K_EVAL       = 5    # number of RAG chunks to retrieve per question
MAX_CONTEXT_CHARS = 1800  # ~450 tokens; keeps total prompt well under num_ctx=4096


class EvaluationRequest(BaseModel):
    instructions: List[str]
    scenario: Optional[str] = "balanced"


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

    total_start  = time.time()
    scenario = request.scenario or "balanced"
    print(f"[eval] Using hyperparameter scenario: {scenario}")

    # ── Phase 1: Prepare all cases (context retrieval + prompt assembly) ───────
    cases = []
    for instruction in request.instructions:
        case = next((c for c in all_test_cases if c["instruction"] == instruction), None)
        if not case:
            continue

        fallback_ctx = case.get("context", "")
        ground_truth = case.get("response", "")

        # ── Build evaluation context ──────────────────────────────────────────
        # Strategy: use full Milvus RAG (all top-5 chunks, no char limit) as primary.
        # The test-set context field is a short keyword stub; include it as a hint.
        # Both are combined so the model has maximum signal to generate correctly.
        rich_context = _build_rich_context(instruction, fallback_ctx)
        print(f"[eval] RAG context: {len(rich_context)} chars for '{instruction[:40]}...'")

        # Combine RAG with the test-set's own context label (short keyword hint)
        if fallback_ctx and fallback_ctx not in rich_context:
            combined_context = f"{rich_context}\n\nKonteks tambahan: {fallback_ctx}"
        else:
            combined_context = rich_context

        cases.append({
            "instruction":  instruction,
            "ground_truth": ground_truth,
            "rich_context": rich_context,
            "prompt": (
                f"Konteks:\n{combined_context}\n\n"
                f"Pertanyaan: {instruction}\n"
                f"Jawaban:"
            ),
        })


    # ── Phase 2: ALL vanilla (qwen3.5:9b) — one model load, no switching ─────
    print(f"[eval] === Vanilla batch: {len(cases)} questions ===")
    for c in cases:
        t0 = time.time()
        c["vanilla_response"] = generate_local(c["prompt"], model="qwen3.5:9b") or "Generation Failed"
        c["vanilla_time"]     = round(time.time() - t0, 2)
        print(f"[eval] Vanilla {c['vanilla_time']}s — {c['instruction'][:40]}")

    # ── Phase 3: ALL fine-tuned (qwen3.5-9b-nlaw) — one model load ───────────
    print(f"[eval] === Fine-tuned batch: {len(cases)} questions ===")
    for c in cases:
        t1 = time.time()
        c["finetuned_response"] = generate_local(c["prompt"], model="qwen3.5-9b-nlaw",
                                                   eval_scenario=scenario) or "Generation Failed"
        c["finetuned_time"]     = round(time.time() - t1, 2)
        print(f"[eval] FT {c['finetuned_time']}s — {c['instruction'][:40]}")

    # ── Phase 4: Metrics + visualization ──────────────────────────────────────
    results = []
    for c in cases:
        t2 = time.time()
        vanilla_metrics   = run_full_evaluation([c["vanilla_response"]],   [c["ground_truth"]])
        finetuned_metrics = run_full_evaluation([c["finetuned_response"]], [c["ground_truth"]])
        eval_time = round(time.time() - t2, 2)

        v_emb  = get_embeddings_local(c["vanilla_response"],   model=EMBED_MODEL)
        ft_emb = get_embeddings_local(c["finetuned_response"], model=EMBED_MODEL)
        gt_emb = get_embeddings_local(c["ground_truth"],       model=EMBED_MODEL)
        combined_viz = _combined_viz(v_emb, ft_emb, gt_emb)

        results.append({
            "instruction":  c["instruction"],
            "ground_truth": c["ground_truth"],
            "context_used": c["rich_context"][:300] + "..." if len(c["rich_context"]) > 300 else c["rich_context"],
            "combined_viz": combined_viz,
            "_embs": {"v": v_emb, "ft": ft_emb, "gt": gt_emb},
            "vanilla": {
                "response":     c["vanilla_response"],
                "metrics":      vanilla_metrics,
                "gen_time_sec": c["vanilla_time"],
            },
            "finetuned": {
                "response":     c["finetuned_response"],
                "metrics":      finetuned_metrics,
                "gen_time_sec": c["finetuned_time"],
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
    response_data = {
        "results": results,
        "total_time_sec": total_time,
        "global_viz": global_viz,
        "scenario_used": scenario,
    }

    # Cache results to disk so they survive frontend disconnects
    try:
        with open(EVAL_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"[eval] Results cached to {EVAL_CACHE_PATH}")
    except Exception as e:
        print(f"[eval] Failed to cache results: {e}")

    return response_data


@router.get("/api/evaluate/last")
def get_last_evaluation():
    """Retrieve the last cached evaluation results."""
    if not os.path.exists(EVAL_CACHE_PATH):
        raise HTTPException(status_code=404, detail="No cached evaluation results found.")
    try:
        with open(EVAL_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read cache: {e}")
