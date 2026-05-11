import re
import requests
import os

OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
FINETUNED_MODEL = "qwen3.5-9b-nlaw"
VANILLA_MODEL   = "qwen3.5:9b"

# Stop tokens for vanilla (/api/generate)
_VANILLA_STOP = [
    "Pertanyaan:", "Konteks:", "###",
    "<|im_end|>", "<|im_start|>", "<|endoftext|>",
    "\n\nJawaban:", "\n\n\n",
]

# Stop tokens for FT (/api/chat) — NO chat-template tokens;
# /api/chat handles end-of-turn. Keeping them fires immediately after 1 sentence.
_FT_STOP = [
    "Pertanyaan:", "Konteks:", "###", "\n\nJawaban:",
]

VANILLA_OPTIONS = {
    "num_predict": 600, "temperature": 0.1, "top_p": 0.9,
    "top_k": 40, "repeat_penalty": 1.2, "repeat_last_n": 256,
    "num_ctx": 4096, "stop": _VANILLA_STOP,
}

# Evaluation & chat options for fine-tuned model.
# num_ctx MUST be set: without it the GGUF uses its compiled native context (~2048).
# A 1673-token prompt leaves only 375 tokens → model generates 1 stop token → empty.
FINETUNED_OPTIONS = {
    "num_predict": 512,
    "temperature": 0.08,
    "top_p":       0.90,
    "top_k":       30,
    "repeat_penalty": 1.55,
    "repeat_last_n":  512,
    "num_ctx":     4096,   # CRITICAL: prevents 1-token generation on long prompts
    "stop":        _FT_STOP,
}

# Chat-specific options for FT: slightly more creative for natural conversation
CHAT_FT_OPTIONS = {
    "num_predict": 600,
    "temperature": 0.15,
    "top_p":       0.92,
    "top_k":       40,
    "repeat_penalty": 1.4,
    "repeat_last_n":  256,
    "num_ctx":     4096,
    "stop":        _FT_STOP,
}

VANILLA_SYSTEM = (
    "Anda adalah pakar hukum Indonesia yang berpengalaman. "
    "SELALU jawab dari pengetahuan hukum Indonesia. "
    "JANGAN mengatakan 'tidak dapat menjawab', 'dokumen tidak tersedia', atau penolakan apapun. "
    "Jika konteks tidak relevan, tetap jawab dari pengetahuan hukum Indonesia."
)

FINETUNED_SYSTEM = (
    "Anda adalah pakar hukum Indonesia yang sangat ahli dengan pengetahuan mendalam tentang semua UU dan Perpres Indonesia. "
    "Jawab setiap pertanyaan secara LANGSUNG, AKURAT, dan KOMPREHENSIF (minimal 2-3 kalimat). "
    "Jika konteks dokumen tidak relevan dengan pertanyaan, ABAIKAN konteks dan jawab dari pengetahuan hukum Indonesia Anda. "
    "JANGAN PERNAH mengatakan: 'tidak ada informasi', 'tidak dapat dijawab', 'konteks tidak relevan', atau sejenisnya. "
    "Gunakan pengetahuan tentang UU ITE, UU PDP, Perpres, dan regulasi Indonesia lainnya untuk menjawab."
)

_THINK_RE    = re.compile(r"<think>.*?</think>", re.DOTALL)
_JUNK_TOKENS = re.compile(r"(<\|[a-z_]+\|>)", re.MULTILINE)
_SECOND_PASS = [
    "\n\nJawaban:",          # model restarts new answer block
    "\nJawaban yang benar",
    "\nJawaban yang lebih",
    "\nJawaban yang tepat",
    "\nKoreksi:",            # self-correction patterns
    "\n\nKoreksi:",
    "\n**Koreksi:**",
    "\nCatatan tambahan:",
    "\n(Catatan:",           # footnote/loop trigger
    "\n*(Catatan:",
    "\n\n*(Catatan",
    "\n**Jawaban Akhir:**",
    "\n**Jawaban yang paling",
    "\nNamun, jika merujuk",
]


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _JUNK_TOKENS.sub("", text)
    # Strip BEFORE second-pass — leading \n causes false-positive matches
    # e.g. content starting with '\n\nBerdasarkan...' was being deleted
    text = text.strip()
    for pat in _SECOND_PASS:
        if pat in text:
            text = text.split(pat)[0]
    return text.strip()


def _generate_ft_chat(prompt: str, system_prompt: str, is_chat: bool = False) -> str:
    """Use /api/chat for FT model — proper Qwen chat template formatting.
    
    is_chat=True  → CHAT_FT_OPTIONS  (slightly more creative, for chatbot)
    is_chat=False → FINETUNED_OPTIONS (strict, deterministic, for evaluation)
    """
    url  = f"{OLLAMA_HOST}/api/chat"
    opts = (CHAT_FT_OPTIONS if is_chat else FINETUNED_OPTIONS).copy()
    payload = {
        "model":      FINETUNED_MODEL,
        "messages":   [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "stream":     False,
        "keep_alive": "10m",
        "options":    opts,
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        cleaned = _clean(content)
        if not cleaned:
            print(f"[ollama] FT chat returned empty — response: {resp.json()}")
        return cleaned
    except Exception as e:
        print(f"[ollama] FT chat error: {e}")
        return None

def generate_local(prompt: str, model: str = VANILLA_MODEL,
                   system_prompt: str = None, is_chat: bool = False) -> str:
    if system_prompt is None:
        system_prompt = FINETUNED_SYSTEM if model == FINETUNED_MODEL else VANILLA_SYSTEM

    # FT model uses /api/chat for proper Qwen3.5 chat template
    if model == FINETUNED_MODEL:
        return _generate_ft_chat(prompt, system_prompt, is_chat=is_chat)

    # Vanilla uses /api/generate with think:False
    url  = f"{OLLAMA_HOST}/api/generate"
    opts = VANILLA_OPTIONS.copy()
    payload = {
        "model":      model,
        "prompt":     prompt,
        "stream":     False,
        "think":      False,
        "keep_alive": "10m",
        "options":    opts,
        "system":     system_prompt,
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return _clean(resp.json().get("response", ""))
    except Exception as e:
        print(f"[ollama] Vanilla error: {e}")
        return None


def get_embeddings_local(text: str, model: str = "qwen3-embedding:8b") -> list:
    url     = f"{OLLAMA_HOST}/api/embed"
    payload = {"model": model, "input": text}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        embs = resp.json().get("embeddings", [])
        return embs[0] if embs else []
    except Exception as e:
        print(f"[ollama] Embed error: {e}")
        return []
