from fastapi import APIRouter
from services.milvus_client import get_client, COLLECTION_NAME

router = APIRouter()

@router.get("/documents")
def list_documents():
    client = get_client()
    stats = client.get_collection_stats(COLLECTION_NAME)
    return {"status": "ok", "collection_entities": stats.get("row_count", 0)}

@router.delete("/documents/{doc_name}")
def delete_document(doc_name: str):
    client = get_client()
    client.delete(collection_name=COLLECTION_NAME, filter=f"doc_name == '{doc_name}'")
    return {"status": f"Deleted chunks for {doc_name}"}
