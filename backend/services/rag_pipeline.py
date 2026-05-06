import os
from services.milvus_client import search_milvus
from services.searxng_client import search_web
from services.toon_formatter import format_legal_context_toon, format_web_context_toon
from services.hf_inference import query_hf_api

def run_rag(question: str, use_web_search: bool, model: str = "qwen3.5-9b-nlaw") -> dict:
    top_k = int(os.getenv("TOP_K_CHUNKS", "5"))
    
    chunks, max_score = search_milvus(question, top_k)

    web_results = []
    if use_web_search:
        search_query = f"{question} site:hukumonline.com OR site:jdih.go.id OR site:peraturan.go.id"
        web_results = search_web(search_query)

    legal_toon = format_legal_context_toon(chunks)
    web_toon = format_web_context_toon(web_results)

    prompt = f"""Anda adalah pakar hukum Indonesia yang presisi. Gunakan konteks berikut untuk menjawab pertanyaan.
{legal_toon}
{web_toon}

Pertanyaan: {question}
Jawaban:"""

    answer = None
    try:
        from services.ollama_client import generate_local
        answer = generate_local(prompt=prompt, model=model)
    except Exception as e:
        print(f"Ollama generation failed: {e}")
        answer = None

    if not answer:
        print("Falling back to Hugging Face Inference API...")
        answer = query_hf_api(prompt)

    json_equivalent = str(chunks) + str(web_results)
    toon_combined = legal_toon + web_toon
    tokens_saved = max(0, len(json_equivalent) - len(toon_combined)) // 4

    sources = [{
        "pasal":   "-",
        "isi":     c.get("chunk_text", "")[:50] + "...",
        "sumber":  c.get("doc_name", ""),
        "page_no": c.get("page_no", 0)
    } for c in chunks]

    return {
        "answer":           answer.strip() if answer else "Gagal menghasilkan jawaban.",
        "sources":          sources,
        "web_results":      web_results,
        "retrieval_score":  max_score,
        "toon_tokens_saved": tokens_saved
    }
