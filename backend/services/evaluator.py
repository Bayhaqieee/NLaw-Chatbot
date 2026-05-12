import os
import math
import numpy as np
import requests
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import evaluate
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from sentence_transformers import CrossEncoder
from services.ollama_client import get_embeddings_local, OLLAMA_HOST

# ══════════════════════════════════════════════════════════════════════
# Module-level singletons — loaded ONCE at container startup.
# ══════════════════════════════════════════════════════════════════════

_hf_metrics: dict = {}

def _get_metric(name: str):
    if name not in _hf_metrics:
        _hf_metrics[name] = evaluate.load(name)
    return _hf_metrics[name]


# ── NLI model ──────────────────────────────────────────────────────────
_NLI_PATH = "/app/Model/nli-deberta-v3-base"
print(f"[evaluator] Loading NLI model from: {_NLI_PATH}")
_nli_model: CrossEncoder = CrossEncoder(_NLI_PATH)

# ── RoBERTa MLM — used for Pseudo-Perplexity (PPPL) ──────────────────
_ROBERTA_PATH = "/app/Model/roberta-large"
print(f"[evaluator] Loading RoBERTa tokenizer for PPPL from: {_ROBERTA_PATH}")
_roberta_tokenizer = RobertaTokenizer.from_pretrained(_ROBERTA_PATH)
_roberta_model     = RobertaForMaskedLM.from_pretrained(_ROBERTA_PATH)
_roberta_model.eval()

print("[evaluator] All singleton models ready.")

# Ollama embedding model names
_SIM_MODEL = "paraphrase-multilingual:278m-mpnet-base-v2-fp16"
_LAT_MODEL = "qwen3-embedding:8b"

# ── PPPL token cap — prevents excessive CPU time on long answers ───────
# Decision (logged in Technical_Specs.md): Approximate PPPL uses batch masking
# (10% of tokens per pass ≈ 10 passes) instead of N passes, capped at 50 tokens.
_PPPL_MAX_TOKENS = 50


# ══════════════════════════════════════════════════════════════════════
# Pseudo-Perplexity (PPPL)
# ══════════════════════════════════════════════════════════════════════

def _compute_pppl(text: str) -> float:
    """
    Approximate Pseudo-Perplexity using batch masking (Salazar et al., 2020).

    Standard PPPL masks 1 token at a time → N forward passes.
    This implementation masks ~10% of tokens per batch → ~10 passes (10× faster).
    Input is capped at _PPPL_MAX_TOKENS (50) for CPU feasibility.

    Lower PPPL = better: the model assigns higher probability to the text,
    indicating more fluent and natural language generation.

    Args:
        text: Generated answer text to evaluate.

    Returns:
        PPPL score (float). Lower is better.
    """
    if not text or text == "Generation Failed":
        return float("inf")

    try:
        enc = _roberta_tokenizer(
            text,
            return_tensors="pt",
            max_length=_PPPL_MAX_TOKENS,
            truncation=True,
            padding=False,
        )
        input_ids = enc["input_ids"][0]   # shape (N,)
        N = input_ids.size(0)

        # Ignore special tokens [CLS]=0, [SEP]=-1
        token_indices = list(range(1, N - 1))
        if not token_indices:
            return float("inf")

        # Batch size = 10% of tokens, minimum 1
        batch_size = max(1, len(token_indices) // 10)
        log_probs  = []

        for start in range(0, len(token_indices), batch_size):
            batch_idx = token_indices[start : start + batch_size]

            # Clone and mask the batch positions
            masked_ids = input_ids.clone()
            for idx in batch_idx:
                masked_ids[idx] = _roberta_tokenizer.mask_token_id

            with torch.no_grad():
                logits = _roberta_model(masked_ids.unsqueeze(0)).logits[0]  # (N, V)

            for idx in batch_idx:
                true_tok  = input_ids[idx].item()
                log_p     = torch.log_softmax(logits[idx], dim=-1)[true_tok].item()
                log_probs.append(log_p)

        if not log_probs:
            return float("inf")

        pppl = math.exp(-sum(log_probs) / len(log_probs))
        return round(pppl, 4)

    except Exception as e:
        print(f"[evaluator] PPPL error: {e}")
        return float("inf")


# ══════════════════════════════════════════════════════════════════════
# Evaluation functions
# ══════════════════════════════════════════════════════════════════════

def evaluate_semantics(predictions, references):
    """SacreBLEU, ROUGE-L, METEOR"""
    sacrebleu = _get_metric("sacrebleu")
    rouge     = _get_metric("rouge")
    meteor    = _get_metric("meteor")

    bleu      = sacrebleu.compute(predictions=predictions, references=[[r] for r in references])
    rouge_res = rouge.compute(predictions=predictions, references=references)
    met_res   = meteor.compute(predictions=predictions, references=references)

    return {
        "SacreBLEU": bleu["score"],
        "ROUGE-1":   rouge_res.get("rouge1", 0),
        "ROUGE-L":   rouge_res.get("rougeL", 0),
        "METEOR":    met_res.get("meteor", 0),
    }


def evaluate_sequential(predictions, references):
    """BERTScore (local RoBERTa), Sentence Similarity (Ollama), NLI (local DeBERTa), PPPL"""
    bertscore  = _get_metric("bertscore")
    model_type = "/app/Model/roberta-large" if os.path.exists("/app/Model/roberta-large") else "roberta-large"

    b_score = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="id",
        model_type=model_type,
        num_layers=24,
    )
    avg_bert_f1 = float(np.mean(b_score["f1"]))

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
    pairs      = [(r, p) for p, r in zip(predictions, references)]
    logits     = _nli_model.predict(pairs)
    nli_scores = [float(row[1]) for row in logits]

    # Pseudo-Perplexity via batch-masked RoBERTa (approximate, 10 passes)
    pppl_scores = [_compute_pppl(p) for p in predictions]
    # Filter out inf values for averaging (failed generations)
    valid_pppl  = [s for s in pppl_scores if s != float("inf")]
    avg_pppl    = round(float(np.mean(valid_pppl)), 4) if valid_pppl else None

    return {
        "BERTScore (F1)":      avg_bert_f1,
        "Sentence Similarity": float(np.mean(sim_scores)),
        "NLI Entailment":      float(np.mean(nli_scores)),
        "Perplexity":          avg_pppl,   # float (lower=better) or None
        "BARTScore":           "N/A",
    }


def evaluate_latent(predictions, references):
    """NLaw Score (Cosine) and L2 distance via qwen3-embedding:8b (Ollama)."""
    pred_embs = [get_embeddings_local(p, model=_LAT_MODEL) for p in predictions]
    ref_embs  = [get_embeddings_local(r, model=_LAT_MODEL) for r in references]

    cosine_sims, l2_dists = [], []
    for p_emb, r_emb in zip(pred_embs, ref_embs):
        if not p_emb or not r_emb:
            continue
        cosine_sims.append(float(1 - cosine(p_emb, r_emb)))
        l2_dists.append(float(euclidean(p_emb, r_emb)))

    return {
        "NLaw Score (Cosine)": float(np.mean(cosine_sims)) if cosine_sims else 0.0,
        "L2 Latent Space":     float(np.mean(l2_dists))    if l2_dists    else 0.0,
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
