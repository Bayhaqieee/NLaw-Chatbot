from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    use_web_search: bool = False
    use_hf: bool = False           # True = HuggingFace Inference API; False (default) = local Ollama
    session_id: Optional[str] = None
    model: Optional[str] = "qwen3.5-9b-nlaw"  # local Ollama model name (ignored when use_hf=True)

class SourceCitation(BaseModel):
    pasal: str
    isi: str
    sumber: str
    page_no: int

class WebResult(BaseModel):
    snippet: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    web_results: List[WebResult]
    retrieval_score: float
    toon_tokens_saved: int

class UploadResponse(BaseModel):
    filename: str
    chunks_embedded: int
    status: str
