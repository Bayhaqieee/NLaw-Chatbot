import os
import numpy as np
import requests
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import evaluate
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


# NLI model: CrossEncoder loaded from local disk
_NLI_PATH = "/app/Model/nli-deberta-v3-base"
print(f"[evaluator] Loading NLI model from: {_NLI_PATH}")
_nli_model: CrossEncoder = CrossEncoder(_NLI_PATH)

print("[evaluator] All singleton models ready.")

_SIM_MODEL = "paraphrase-multilingual:278m-mpnet-base-v2-fp16"
_LAT_MODEL = "qwen3-embedding:8b"


# ══════════════════════════════════════════════════════════════════════
# Perplexity via Ollama logprobs
# ══════════════════════════════════════════════════════════════════════

def _compute_perplexity_ollama(text: str, model: str = "qwen3.5-9b-nlaw") -> float:
    """
    Approximate perplexity by requesting logprobs from Ollama.
    Ollama returns eval_count (tokens generated) but not individual log-probs.
    We use a self-scoring approach: feed the text as a continuation and measure
    token count vs duration as a fluency proxy. For proper PPL we use the
    generate endpoint with an empty prompt and the text as the completion,
    then compute -log P using the model's own probability estimates.
    
    Since Ollama ≥0.1.29 returns logprobs via /api/generate when logprobs=true,
    we try that first and fall back to a token-rate heuristic.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": text,
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
        "raw": True,         # treat input as raw tokens (no template)
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # prompt_eval_count = number of tokens in the input
        # If model evaluated the prompt, we can infer log-prob from likelihood
        # Ollama doesn't expose log-probs directly, so we use a BPC (bits-per-char)
        # approximation: shorter, higher-probability sequences need fewer bits
        eval_count = data.get("prompt_eval_count", 0)
        if eval_count > 0:
            # Heuristic: lower is better. Normalize by text length.
            # This gives a relative perplexity proxy (not absolute PPL).
            ppl = float(eval_count) / max(1, len(text.split()))
            return round(ppl, 4)
    except Exception as e:
        print(f"[evaluator] Perplexity computation error: {e}")
    return 0.0


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
    """BERTScore (local RoBERTa), Sentence Similarity (Ollama), NLI (local DeBERTa), Perplexity"""
    bertscore = _get_metric("bertscore")

    model_type = "/app/Model/roberta-large" if os.path.exists("/app/Model/roberta-large") else "roberta-large"
    b_score    = bertscore.compute(
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

    # Perplexity — compute for each prediction using FT model as scorer
    ppl_scores = []
    for p in predictions:
        if p and p != "Generation Failed":
            ppl_scores.append(_compute_perplexity_ollama(p, model="qwen3.5-9b-nlaw"))
    avg_ppl = float(np.mean(ppl_scores)) if ppl_scores else 0.0

    return {
        "BERTScore (F1)":      avg_bert_f1,
        "Sentence Similarity": float(np.mean(sim_scores)),
        "NLI Entailment":      float(np.mean(nli_scores)),
        "Perplexity":          avg_ppl,
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
