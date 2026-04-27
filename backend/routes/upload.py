from fastapi import APIRouter, UploadFile, File
from models.schemas import UploadResponse
from services.pdf_parser import extract_text_from_pdf, chunk_text
from services.milvus_client import get_embedding_model, insert_chunks, COLLECTION_NAME
from datetime import datetime

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return UploadResponse(filename=file.filename, chunks_embedded=0, status="Error: Only PDF allowed")

    contents = await file.read()
    pages_data = extract_text_from_pdf(contents)
    model = get_embedding_model()

    batch = []
    for page_data in pages_data:
        for chunk in chunk_text(page_data["text"]):
            if len(chunk) < 10:
                continue
            embedding = model.encode(chunk).tolist()
            batch.append({
                "doc_name":    file.filename,
                "category":    "USER_UPLOAD",
                "chunk_text":  chunk,
                "page_no":     page_data["page_no"],
                "embedding":   embedding,
                "upload_by":   "user",
                "uploaded_at": datetime.utcnow().isoformat(),
            })

    if batch:
        insert_chunks(batch)

    return UploadResponse(filename=file.filename, chunks_embedded=len(batch), status="Success")
