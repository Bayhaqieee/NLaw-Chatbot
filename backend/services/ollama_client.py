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
# Evaluation & chat options for fine-tuned model.
# num_ctx MUST be set: without it the GGUF uses its compiled native context (~2048).
# A 1673-token prompt leaves only 375 tokens → model generates 1 stop token → empty.
FINETUNED_OPTIONS = {
    "num_predict": 512,
    "temperature": 0.08,
    "top_p":       0.90,
    "top_k":       30,
    "repeat_penalty": 1.10,
    "repeat_last_n":  128,
    "num_ctx":     4096,   # CRITICAL: prevents 1-token generation on long prompts
    "stop":        _FT_STOP,
}

# Chat-specific options for FT: slightly more creative for natural conversation
CHAT_FT_OPTIONS = {
    "num_predict": 600,
    "temperature": 0.10,
    "top_p":       0.90,
    "top_k":       40,
    "repeat_penalty": 1.10,
    "repeat_last_n":  128,
    "num_ctx":     4096,
    "stop":        _FT_STOP,
}

# ── Evaluation Hyperparameter Scenarios ──────────────────────────────────
# These ONLY affect the Fine-Tuned model during evaluation runs.
# The Chatbot page always uses CHAT_FT_OPTIONS; Vanilla always uses VANILLA_OPTIONS.
EVAL_SCENARIOS = {
    "conservative": {
        "num_predict": 512,  "temperature": 0.05, "top_p": 0.85,
        "top_k": 20, "repeat_penalty": 1.3, "repeat_last_n": 512,
        "num_ctx": 4096, "stop": _FT_STOP,
    },
    "balanced": {
        "num_predict": 512,  "temperature": 0.10, "top_p": 0.90,
        "top_k": 40, "repeat_penalty": 1.55, "repeat_last_n": 512,
        "num_ctx": 4096, "stop": _FT_STOP,
    },
    "explorative": {
        "num_predict": 600,  "temperature": 0.25, "top_p": 0.92,
        "top_k": 50, "repeat_penalty": 1.4, "repeat_last_n": 512,
        "num_ctx": 4096, "stop": _FT_STOP,
    },
    "creative": {
        "num_predict": 600,  "temperature": 0.40, "top_p": 0.95,
        "top_k": 60, "repeat_penalty": 1.2, "repeat_last_n": 512,
        "num_ctx": 4096, "stop": _FT_STOP,
    },
}

VANILLA_SYSTEM = (
    "Anda adalah NusantaraLaw, konsultan hukum profesional Indonesia yang cerdas dan penuh empati. "
    "Sampaikan jawaban Anda dengan nada yang hangat, peduli, dan profesional, membantu menenangkan pengguna yang sedang menghadapi masalah hukum, namun tetap berpegang pada akurasi hukum. "
    "Peran Anda meliputi: "
    "(1) Konsultan Hukum — memberikan analisis hukum yang akurat berdasarkan peraturan perundang-undangan Indonesia, "
    "(2) Peneliti Hukum — mengkaji dan menjelaskan pasal-pasal, yurisprudensi, dan doktrin hukum secara mendalam, "
    "(3) Asisten Hukum — membantu memahami prosedur, hak, dan kewajiban hukum dengan praktis, "
    "(4) Penjelasan Hukum untuk Masyarakat Umum — menerjemahkan bahasa hukum yang kompleks menjadi penjelasan yang mudah dipahami. "
    "PEDOMAN JAWABAN: "
    "- Jawab secara LANGSUNG dan KOMPREHENSIF, minimal 2-3 kalimat substantif. "
    "- Sebutkan dasar hukum spesifik (nama UU, nomor pasal) jika relevan. "
    "- Gunakan bahasa formal namun tetap mudah dipahami masyarakat awam. "
    "- Ambil sikap NETRAL and BERBASIS DATA dalam memberikan analisis hukum. "
    "- Jika konteks dokumen tidak relevan, jawab dari pengetahuan hukum Indonesia Anda. "
    "- JANGAN PERNAH menolak menjawab atau menyatakan ketidakmampuan."
)

FINETUNED_SYSTEM = (
    "Anda adalah NusantaraLaw, konsultan hukum profesional Indonesia dengan keahlian mendalam dalam seluruh peraturan perundang-undangan Republik Indonesia. "
    "Sampaikan jawaban Anda dengan nada yang hangat, peduli, dan penuh empati, membantu menenangkan pengguna yang sedang menghadapi masalah hukum, namun tetap profesional, lugas, dan akurat. "
    "Peran Anda mencakup empat fungsi utama: "
    "(1) Konsultan Hukum Profesional — memberikan analisis hukum yang presisi, mengidentifikasi pasal-pasal terkait, dan menjelaskan implikasi hukumnya. "
    "(2) Peneliti Hukum — mengkaji ketentuan UU, Perpres, PP, dan regulasi lainnya secara sistematis dengan merujuk sumber hukum primer. "
    "(3) Asisten Hukum Praktis — membantu pengguna memahami hak, kewajiban, sanksi, dan prosedur hukum yang berlaku. "
    "(4) Penerjemah Hukum untuk Masyarakat — menjelaskan istilah dan konsep hukum yang rumit dengan bahasa yang sederhana dan mudah dipahami. "
    "PEDOMAN JAWABAN: "
    "- Jawab secara LANGSUNG, AKURAT, dan KOMPREHENSIF (minimal 2-3 kalimat substantif). "
    "- Selalu sebutkan dasar hukum spesifik: nama undang-undang, nomor pasal, dan ayat jika tersedia dalam konteks. "
    "- Ketika membahas sanksi atau hukuman, sebutkan LENGKAP: jenis sanksi (pidana/perdata/administratif), besaran denda, dan durasi hukuman. "
    "- Ambil sikap NETRAL dan BERBASIS DATA — hindari opini politik, fokus pada fakta hukum. "
    "- Jika konteks dokumen tidak relevan dengan pertanyaan, ABAIKAN konteks dan jawab dari pengetahuan hukum Indonesia. "
    "- JANGAN PERNAH mengatakan: 'tidak ada informasi', 'tidak dapat dijawab', 'konteks tidak relevan', atau penolakan serupa. "
    "- Gunakan format terstruktur jika jawaban kompleks: poin-poin bernomor untuk langkah prosedural, bullet points untuk daftar ketentuan."
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


def generate_local(prompt: str, model: str = VANILLA_MODEL,
                   system_prompt: str = None, is_chat: bool = False,
                   eval_scenario: str = None) -> str:
    if system_prompt is None:
        system_prompt = FINETUNED_SYSTEM if model == FINETUNED_MODEL else VANILLA_SYSTEM

    # Use /api/generate for all models to match training template and prevent chat template nesting bugs
    url = f"{OLLAMA_HOST}/api/generate"

    if model == FINETUNED_MODEL:
        if eval_scenario and eval_scenario in EVAL_SCENARIOS:
            opts = EVAL_SCENARIOS[eval_scenario].copy()
            print(f"[ollama] FT using eval scenario: {eval_scenario}")
        elif is_chat:
            opts = CHAT_FT_OPTIONS.copy()
        else:
            opts = FINETUNED_OPTIONS.copy()
    else:
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
        print(f"[ollama] Error generating for model {model}: {e}")
        return None


def get_embeddings_local(text: str, model: str = "qwen3-embedding:8b") -> list:
    """Get embeddings from Ollama with retry logic for GPU contention.
    
    During evaluation, the GPU alternates between LLM generation and embedding.
    If the model is still unloading/loading, the embed call can time out.
    We retry up to 3 times with increasing backoff to handle this.
    """
    import time
    url     = f"{OLLAMA_HOST}/api/embed"
    payload = {"model": model, "input": text}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            embs = resp.json().get("embeddings", [])
            return embs[0] if embs else []
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"[ollama] Embed attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"[ollama] Embed error after {max_retries} attempts: {e}")
                return []
