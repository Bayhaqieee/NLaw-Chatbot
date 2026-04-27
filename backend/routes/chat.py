from fastapi import APIRouter
from models.schemas import ChatRequest, ChatResponse
from services.rag_pipeline import run_rag

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    result = run_rag(request.question, request.use_web_search)
    return ChatResponse(**result)
