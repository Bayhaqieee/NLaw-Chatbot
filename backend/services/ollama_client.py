import re
import requests
import os

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
FINETUNED_MODEL = "qwen3.5-9b-nlaw"

# ── Common stop tokens — shared by both models ─────────────────────────────
_COMMON_STOP = [
    "Pertanyaan:",       # prevents re-generating the Q&A template
    "Konteks:",          # prevents re-generating context header
    "###",
    "<|im_end|>",
    "<|im_start|>",      # Qwen chat-template token must be stopped
    "<|endoftext|>",     # EOS token leaking through GGUF
    "\n\nJawaban:",      # stops repetition loops (second Jawaban: block)
    "\n\n\n",            # stops excessive blank lines
]

# ── Vanilla model (qwen3.5:9b) ─────────────────────────────────────────────
VANILLA_OPTIONS = {
    "num_predict":    600,
    "temperature":    0.1,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.2,
    "repeat_last_n":  256,
    "stop":           _COMMON_STOP,
}

VANILLA_SYSTEM = (
    "Anda adalah pakar hukum Indonesia yang berpengalaman. "
    "Jawab setiap pertanyaan secara LANGSUNG dan AKURAT berdasarkan konteks hukum yang diberikan "
    "dan pengetahuan hukum Indonesia Anda. "
    "PENTING: Jangan pernah mengatakan 'dokumen tidak tersedia', 'tidak dapat dijawab', atau "
    "'UU X tidak ada dalam dokumen'. Selalu berikan jawaban substantif berdasarkan hukum Indonesia."
)

# ── Fine-tuned model (qwen3.5-9b-nlaw) ────────────────────────────────────
# High repeat_penalty is ESSENTIAL to prevent looping.
# repeat_last_n 512 covers full context window for dedup detection.
FINETUNED_OPTIONS = {
    "num_predict":    500,
    "temperature":    0.12,
    "top_p":          0.90,
    "top_k":          30,
    "repeat_penalty": 1.55,     # must be high — model loops without this
    "repeat_last_n":  512,
    "stop":           _COMMON_STOP,
}

FINETUNED_SYSTEM = (
    "Anda adalah pakar hukum Indonesia. "
    "Jawab pertanyaan secara LANGSUNG dan AKURAT mulai dengan 'Sesuai ketentuan, ...'. "
    "Berikan SATU jawaban singkat dan tepat — jangan mengulangi atau merevisi jawaban Anda. "
    "JANGAN pernah mengatakan 'dokumen tidak tersedia' atau 'tidak dapat dijawab'. "
    "Jika konteks tidak lengkap, gunakan pengetahuan hukum Indonesia Anda untuk menjawab."
)

_THINK_RE    = re.compile(r"<think>.*?</think>", re.DOTALL)
_JUNK_TOKENS = re.compile(r"(<\|[a-z_]+\|>)", re.MULTILINE)

# Patterns that signal the model is starting a second pass / self-correction.
# We keep only everything BEFORE the first occurrence of any of these.
_SECOND_PASS = [
    "\n\nJawaban:",
    "\n\nBerdasarkan dokumen",
    "\n\nDokumen yang tersedia",
    "\nJawaban yang benar",
    "\nJawaban yang lebih",
    "\nJawaban yang tepat",
    "\nJawaban: Berd",
    "\nApakah ini kesalahan",
    "\nKoreksi:",
    "\nCatatan tambahan:",
    "\nNamun, jika merujuk",
    "\nNamun, berdasarkan",
]


def _clean(text: str) -> str:
    """Strip <think> blocks, leaked tokens, and all second-pass repetitions."""
    text = _THINK_RE.sub("", text)
    text = _JUNK_TOKENS.sub("", text)
    # Truncate at the earliest second-pass marker
    for pat in _SECOND_PASS:
        if pat in text:
            text = text.split(pat)[0]
    return text.strip()


def generate_local(prompt: str, model: str = "qwen3.5:9b",
                   system_prompt: str = None) -> str:
    """Generate text using a local Ollama model."""
    url  = f"{OLLAMA_HOST}/api/generate"
    opts = (FINETUNED_OPTIONS if model == FINETUNED_MODEL else VANILLA_OPTIONS).copy()

    # Inject default system prompts if caller didn't supply one
    if system_prompt is None:
        system_prompt = FINETUNED_SYSTEM if model == FINETUNED_MODEL else VANILLA_SYSTEM

    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "think":   False,
        "options": opts,
        "system":  system_prompt,
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return _clean(raw)
    except Exception as e:
        print(f"Ollama generation error for model {model}: {e}")
        return None


def get_embeddings_local(text: str, model: str = "qwen3-embedding:8b") -> list:
    """Get embeddings via Ollama /api/embed (newer endpoint)."""
    url     = f"{OLLAMA_HOST}/api/embed"
    payload = {"model": model, "input": text}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        embs = resp.json().get("embeddings", [])
        return embs[0] if embs else []
    except Exception as e:
        print(f"Ollama embed error for model {model}: {e}")
        return []
