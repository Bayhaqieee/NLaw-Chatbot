def format_legal_context_toon(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    header = f"legal_context[{len(chunks)}]{{no,pasal,isi,sumber}}:"
    rows = [
        f"{i+1},{c.get('category','-')},{c.get('chunk_text','').replace(',', '；')[:200]},{c.get('doc_name','-')}"
        for i, c in enumerate(chunks)
    ]
    return header + "\n" + "\n".join(rows)

def format_web_context_toon(web_results: list[dict]) -> str:
    if not web_results:
        return ""
    header = f"web_context[{len(web_results)}]{{no,snippet,url}}:"
    rows = [
        f"{i+1},{r.get('snippet','').replace(',', '；')[:150]},{r.get('url','-')}"
        for i, r in enumerate(web_results)
    ]
    return header + "\n" + "\n".join(rows)
