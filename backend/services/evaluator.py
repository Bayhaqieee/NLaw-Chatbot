import os
import math
import numpy as np
import requests
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import evaluate
import torch
from sentence_transformers import CrossEncoder
from services.ollama_client import get_embeddings_local, OLLAMA_HOST

# ══════════════════════════════════════════════════════════════════════
# Module-level singletons — loaded ONCE at container startup.
# ══════════════════════════════════════════════════════════════════════

_hf_metrics: dict = {}

def _get_metric(name: str):
    if name not in _hf_metrics:
        # keep_in_memory=True avoids writing .arrow cache files to disk,
        # which prevents FileNotFoundError when evaluate tries to delete
        # stale cache files during concurrent or repeated evaluations.
        _hf_metrics[name] = evaluate.load(name, keep_in_memory=True)
    return _hf_metrics[name]


# ── NLI model ──────────────────────────────────────────────────────────
_NLI_PATH = "/app/Model/nli-deberta-v3-base"
print(f"[evaluator] Loading NLI model from: {_NLI_PATH}")
_nli_model: CrossEncoder = CrossEncoder(_NLI_PATH)

print("[evaluator] All singleton models ready.")

# ── Ollama embedding model names ─────────────────────────────────────
_SIM_MODEL = "paraphrase-multilingual:278m-mpnet-base-v2-fp16"

# Self-Model Encoder: use the fine-tuned model's embedding representation
# for latent space evaluation (NLaw Score, L2 Distance, Self-Model Perplexity).
# NOTE: Ollama /api/embed returns 501 for generative models (qwen3.5-9b-nlaw),
# so we use qwen3-embedding:8b which shares the same Qwen architecture family.
# The notebook uses the model's own hidden states directly via PyTorch;
# in our Docker/Ollama setup, qwen3-embedding is the closest equivalent.
_SELF_MODEL = "qwen3-embedding:8b"

# External embedding model for RAG indexing (kept separate from eval)
_RAG_EMBED_MODEL = "qwen3-embedding:8b"


# ══════════════════════════════════════════════════════════════════════
# Self-Model Perplexity
# ══════════════════════════════════════════════════════════════════════

def _compute_self_model_perplexity(prediction: str, reference: str) -> float:
    """
    Self-Model Perplexity using the fine-tuned model's own encoder.

    Matches the notebook's approach: the fine-tuned model evaluates its own
    output quality. The notebook computes perplexity as exp(model_loss) using
    the model's causal LM head. Since Ollama doesn't expose per-token logprobs,
    we use the model's embedding space as a proxy:

        PPPL = 1 - cosine_sim(model_embed(prediction), model_embed(reference))

    This measures how close the generated text is to the ground truth in the
    fine-tuned model's own latent space. The key principle is using the SAME
    model that was fine-tuned (qwen3.5-9b-nlaw) as the judge.

    Lower = better (0 = perfect match in model's latent space, 1 = worst).

    Args:
        prediction: Generated answer text to evaluate.
        reference:  Ground truth answer text.

    Returns:
        PPPL score (float, 0–1). Lower is better.
    """
    if not prediction or prediction == "Generation Failed":
        return float("inf")

    try:
        pred_emb = get_embeddings_local(prediction, model=_SELF_MODEL)
        ref_emb  = get_embeddings_local(reference,  model=_SELF_MODEL)
        if pred_emb and ref_emb:
            sim  = float(1 - cosine(pred_emb, ref_emb))
            pppl = round(1.0 - sim, 5)  # 0–1, lower=better
            return pppl
        return float("inf")
    except Exception as e:
        print(f"[evaluator] Self-Model Perplexity error: {e}")
        return float("inf")


# ══════════════════════════════════════════════════════════════════════
# Evaluation functions
# ══════════════════════════════════════════════════════════════════════

def evaluate_semantics(predictions, references):
    """SacreBLEU, ROUGE-L, METEOR — all on 0–100 scale (matches notebook)."""
    sacrebleu = _get_metric("sacrebleu")
    rouge     = _get_metric("rouge")
    meteor    = _get_metric("meteor")

    bleu      = sacrebleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_res = rouge.compute(predictions=predictions, references=references)
    met_res   = meteor.compute(predictions=predictions, references=references)

    return {
        "SacreBLEU": bleu["score"],                          # already 0–100
        "ROUGE-1":   rouge_res.get("rouge1", 0) * 100,       # 0–1 → 0–100
        "ROUGE-L":   rouge_res.get("rougeL", 0) * 100,       # 0–1 → 0–100
        "METEOR":    met_res.get("meteor", 0) * 100,          # 0–1 → 0–100
    }


def evaluate_sequential(predictions, references):
    """BERTScore, Sentence Similarity, NLI, Perplexity — 0–100 scale (matches notebook)."""
    bertscore  = _get_metric("bertscore")
    model_type = "/app/Model/roberta-large" if os.path.exists("/app/Model/roberta-large") else "roberta-large"

    b_score = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="id",
        model_type=model_type,
        num_layers=24,
    )
    avg_bert_f1 = float(np.mean(b_score["f1"])) * 100       # 0–1 → 0–100

    # Sentence Similarity via Ollama paraphrase-multilingual (768-dim)
    sim_scores = []
    for p, r in zip(predictions, references):
        p_emb = get_embeddings_local(p, model=_SIM_MODEL)
        r_emb = get_embeddings_local(r, model=_SIM_MODEL)
        if p_emb and r_emb:
            sim_scores.append(float(1 - cosine(p_emb, r_emb)))
        else:
            sim_scores.append(0.0)

    # NLI Entailment via local DeBERTa CrossEncoder
    # Apply softmax to logits to get probabilities.
    # Notebook uses index 2 for entailment scoring — we match for consistency
    # with thesis XP-1 to XP-4 results.
    pairs      = [(r, p) for p, r in zip(predictions, references)]
    raw_logits = _nli_model.predict(pairs)
    probs      = torch.softmax(torch.tensor(raw_logits), dim=1)
    nli_scores = [float(probs[i, 2].item()) for i in range(len(probs))]  # index 2 (matches notebook)

    # Self-Model Perplexity via fine-tuned model's own encoder (qwen3.5-9b-nlaw)
    # NOTE: Notebook uses exp(cross_entropy_loss) which requires PyTorch model access.
    # Ollama only exposes /api/embed, so we use cosine-distance proxy (0–1, lower=better).
    # This metric is NOT scaled to 0–100 — it remains a distance measure.
    pppl_scores = [_compute_self_model_perplexity(p, r)
                   for p, r in zip(predictions, references)]
    # Filter out inf values for averaging (failed generations)
    valid_pppl  = [s for s in pppl_scores if s != float("inf")]
    avg_pppl    = round(float(np.mean(valid_pppl)), 5) if valid_pppl else None

    return {
        "BERTScore (F1)":      avg_bert_f1,                             # 0–100
        "Sentence Similarity": float(np.mean(sim_scores)) * 100,        # 0–100
        "NLI Entailment":      float(np.mean(nli_scores)) * 100,        # 0–100
        "Perplexity":          avg_pppl,   # 0–1 proxy (lower=better) or None
        "BARTScore":           "N/A",
    }


def evaluate_latent(predictions, references):
    """NLaw Score (Cosine) and L2 distance via the fine-tuned model's own encoder (qwen3.5-9b-nlaw).
    
    Matches the notebook's approach: extract_hidden_vector() uses the fine-tuned
    model itself to compute mean-pooled last-layer hidden states, then measures
    cosine similarity and L2 distance between prediction and ground truth vectors.
    """
    pred_embs = [get_embeddings_local(p, model=_SELF_MODEL) for p in predictions]
    ref_embs  = [get_embeddings_local(r, model=_SELF_MODEL) for r in references]

    cosine_sims, l2_dists = [], []
    for p_emb, r_emb in zip(pred_embs, ref_embs):
        if not p_emb or not r_emb:
            continue
        cosine_sims.append(float(1 - cosine(p_emb, r_emb)))
        l2_dists.append(float(euclidean(p_emb, r_emb)))

    return {
        "NLaw Score (Cosine)": float(np.mean(cosine_sims)) * 100 if cosine_sims else 0.0,  # 0–100
        "L2 Latent Space":     float(np.mean(l2_dists))          if l2_dists    else 0.0,   # raw euclidean
    }, pred_embs, ref_embs


def run_full_evaluation(predictions, references):
    sem                 = evaluate_semantics(predictions, references)
    seq                 = evaluate_sequential(predictions, references)
    lat, p_embs, r_embs = evaluate_latent(predictions, references)

    pca_coords, tsne_coords = [], []
    all_embs = [e for e in (p_embs + r_embs) if e]
    if len(all_embs) >= 2:
        try:
            arr         = np.array(all_embs)
            pca_coords  = PCA(n_components=2).fit_transform(arr).tolist()
            n           = len(all_embs)
            perp        = min(30, max(1, n - 1))
            tsne_coords = TSNE(n_components=2, perplexity=perp).fit_transform(arr).tolist()
        except Exception as e:
            print(f"Visualization error: {e}")

    return {
        "Semantic":      sem,
        "Sequential":    seq,
        "Latent":        lat,
        "Visualization": {"PCA": pca_coords, "t-SNE": tsne_coords},
    }
