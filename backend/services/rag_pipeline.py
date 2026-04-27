import os
from services.milvus_client import search_milvus
from services.searxng_client import search_web
from services.toon_formatter import format_legal_context_toon, format_web_context_toon
from services.hf_inference import query_hf_api

def run_rag(question: str, use_web_search: bool) -> dict:
    top_k = int(os.getenv("TOP_K_CHUNKS", "5"))
    threshold = float(os.getenv("SEARXNG_THRESHOLD", "0.70"))
    
    chunks, max_score = search_milvus(question, top_k)
    
    # Determine if web search is needed
    # Distance in L2. Smaller is better. Let's assume threshold check: if max_score (L2 dist) is large, confidence is low.
    # Note: L2 score is distance. Smaller distance = higher similarity.
    # We will trigger web search if user forces it, or if distance is too large.
    web_results = []
    # threshold for L2 might need adjustment. Let's use user toggle for now as primary.
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

    answer = query_hf_api(prompt)
    
    # Calculate token savings estimation (rough string length proxy)
    json_equivalent = str(chunks) + str(web_results)
    toon_combined = legal_toon + web_toon
    tokens_saved = max(0, len(json_equivalent) - len(toon_combined)) // 4 # rough estimate

    sources = [{
        "pasal": "-",
        "isi": c.get("chunk_text")[:50] + "...",
        "sumber": c.get("doc_name"),
        "page_no": c.get("page_no")
    } for c in chunks]

    return {
        "answer": answer.strip(),
        "sources": sources,
        "web_results": web_results,
        "retrieval_score": max_score, # Actually L2 distance
        "toon_tokens_saved": tokens_saved
    }
