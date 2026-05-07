from fastapi import APIRouter
from models.schemas import ChatRequest, ChatResponse
from services.rag_pipeline import run_rag

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    result = run_rag(
        question=request.question,
        use_web_search=request.use_web_search,
        model=request.model,
        use_hf=request.use_hf,
    )
    return ChatResponse(**result)
