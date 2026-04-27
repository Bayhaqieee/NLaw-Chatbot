import os
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nusantara_law")

def create_collection():
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    print(f"Connecting to Milvus at {uri}...")
    client = MilvusClient(uri=uri)

    if client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Dropping it...")
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",         DataType.INT64,        is_primary=True)
    schema.add_field("doc_name",   DataType.VARCHAR,      max_length=256)
    schema.add_field("category",   DataType.VARCHAR,      max_length=64)
    schema.add_field("chunk_text", DataType.VARCHAR,      max_length=4096)
    schema.add_field("page_no",    DataType.INT32)
    schema.add_field("embedding",  DataType.FLOAT_VECTOR, dim=768)
    schema.add_field("upload_by",  DataType.VARCHAR,      max_length=64)
    schema.add_field("uploaded_at",DataType.VARCHAR,      max_length=32)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="L2",
        params={"M": 8, "efConstruction": 64}
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print(f"Collection '{COLLECTION_NAME}' created and indexed successfully.")

if __name__ == "__main__":
    create_collection()
