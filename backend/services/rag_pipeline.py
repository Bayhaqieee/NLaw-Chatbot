import os
from services.milvus_client import search_milvus
from services.searxng_client import search_web
from services.toon_formatter import format_legal_context_toon, format_web_context_toon
from services.hf_inference import query_hf_api
from services.ollama_client import generate_local, VANILLA_SYSTEM, FINETUNED_SYSTEM, FINETUNED_MODEL

# System prompt for the chat endpoint.
# Prioritizes web search results when available, falls back to RAG + general knowledge.
# Does NOT force document-only answers so general questions (e.g. current president) work.
CHAT_SYSTEM = (
    "Anda adalah asisten hukum Indonesia yang cerdas dan informatif. "
    "Jika terdapat hasil pencarian web, gunakan informasi tersebut sebagai sumber utama. "
    "Jika terdapat konteks dokumen hukum, gunakan untuk pertanyaan hukum. "
    "Untuk pertanyaan umum (bukan hukum), jawab dari pengetahuan umum Anda tanpa merujuk dokumen. "
    "JANGAN mengatakan 'informasi tidak tersedia dalam dokumen' untuk pertanyaan umum. "
    "Untuk topik atau opini politik, selalu ambil sikap yang netral, objektif, dan berbasis data tanpa memihak pihak manapun. "
    "Berikan jawaban yang informatif, akurat, dan langsung."
)


def run_rag(question: str, use_web_search: bool,
            model: str = "qwen3.5-9b-nlaw",
            use_hf: bool = False) -> dict:

    top_k = int(os.getenv("TOP_K_CHUNKS", "5"))

    # ── Vector retrieval ───────────────────────────────────────────────────
    chunks, max_score = search_milvus(question, top_k)

    # ── Optional web augmentation ──────────────────────────────────────────
    web_results = []
    if use_web_search:
        # Plain query — no site: restriction so general questions (e.g. current president)
        # also work. SearXNG will search across all configured engines.
        web_results = search_web(question)


    # ── Prompt assembly — ZERO-SHOT, identical for all models ─────────────────
    # Same structure as the evaluation prompt so chat and eval are consistent.
    # System prompt (per model) handles persona; no format injection here.
    legal_toon = format_legal_context_toon(chunks)
    web_toon   = format_web_context_toon(web_results)

    prompt = (
        f"Konteks:\n{legal_toon}\n{web_toon}\n\n"
        f"Pertanyaan: {question}\n"
        f"Jawaban:"
    )

    # ── Generation — route based on use_hf flag ────────────────────────────
    answer = None

    if use_hf:
        # HuggingFace Inference API path (explicit user choice)
        print(f"[rag] Using HuggingFace Inference API for model: {os.getenv('HF_MODEL_ID')}")
        answer = query_hf_api(prompt)
        # Surface HF errors clearly instead of silently falling through
        if answer and answer.startswith("Error communicating"):
            answer = f"⚠️ HuggingFace API Error: {answer}"
    else:
        # Local Ollama path (default)
        system = CHAT_SYSTEM  # unified chat persona
        print(f"[rag] Using Ollama model: {model}")
        try:
            answer = generate_local(prompt=prompt, model=model,
                                    system_prompt=system, is_chat=True)
        except Exception as e:
            print(f"[rag] Ollama error: {e}")
            answer = f"⚠️ Ollama tidak dapat dijangkau. Pastikan Ollama berjalan di host. Error: {e}"

    # ── Build response ─────────────────────────────────────────────────────
    json_equivalent = str(chunks) + str(web_results)
    toon_combined   = legal_toon + web_toon
    tokens_saved    = max(0, len(json_equivalent) - len(toon_combined)) // 4

    sources = [{
        "pasal":   "-",
        "isi":     c.get("chunk_text", "")[:80] + "...",
        "sumber":  c.get("doc_name", ""),
        "page_no": c.get("page_no", 0)
    } for c in chunks]

    return {
        "answer":            (answer or "Gagal menghasilkan jawaban.").strip(),
        "sources":           sources,
        "web_results":       web_results,
        "retrieval_score":   max_score,
        "toon_tokens_saved": tokens_saved,
    }
