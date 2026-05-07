import os
from services.milvus_client import search_milvus
from services.searxng_client import search_web
from services.toon_formatter import format_legal_context_toon, format_web_context_toon
from services.hf_inference import query_hf_api
from services.ollama_client import generate_local, VANILLA_SYSTEM, FINETUNED_SYSTEM, FINETUNED_MODEL

CHAT_SYSTEM = (
    "Anda adalah pakar hukum Indonesia yang presisi dan terpercaya. "
    "Jawab pertanyaan hukum secara akurat berdasarkan konteks yang diberikan. "
    "Jika informasi tidak ada dalam konteks, gunakan pengetahuan hukum Indonesia Anda. "
    "Berikan jawaban yang jelas, terstruktur, dan mudah dipahami."
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
        query = f"{question} site:hukumonline.com OR site:jdih.go.id OR site:peraturan.go.id"
        web_results = search_web(query)

    # ── Prompt assembly ────────────────────────────────────────────────────
    legal_toon = format_legal_context_toon(chunks)
    web_toon   = format_web_context_toon(web_results)

    prompt = (
        f"{legal_toon}\n{web_toon}\n\n"
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
        system = FINETUNED_SYSTEM if model == FINETUNED_MODEL else VANILLA_SYSTEM
        print(f"[rag] Using Ollama model: {model}")
        try:
            answer = generate_local(prompt=prompt, model=model, system_prompt=system)
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
