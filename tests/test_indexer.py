
from app.ingestion.ingestion_manager import process_file
from app.retrieval.indexer import index_document
from app.retrieval.vector_store import get_qdrant_client, COLLECTION_NAME

file_path = r"uploads\sample_unillm_notes.docx"

document = process_file(file_path)
print("Document keys:", document.keys())
print("Document metadata:", document.get("metadata"))

print("Document extracted. Starting indexing...")

result = index_document(document)
print("Indexing result:", result)

client = get_qdrant_client()
info = client.get_collection(COLLECTION_NAME)

print("Qdrant points:", info.points_count)
