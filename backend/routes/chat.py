from fastapi import APIRouter
from models.schemas import ChatRequest, ChatResponse
from services.rag_pipeline import run_rag
import time

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    print(f"[chat] POST /api/chat received — model={request.model}, web_search={request.use_web_search}")
    print(f"[chat] Question: '{request.question[:80]}...'")
    t0 = time.time()
    try:
        result = run_rag(
            question=request.question,
            use_web_search=request.use_web_search,
            model=request.model,
            use_hf=request.use_hf,
        )
        elapsed = time.time() - t0
        print(f"[chat] Response generated in {elapsed:.1f}s — answer length: {len(result.get('answer', ''))} chars")
        return ChatResponse(**result)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[chat] ERROR after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        raise

