from app.ingestion.ingestion_manager import process_file




result = process_file("uploads/studymate-rag-master-prompt-v4.md.pdf")

print(result["metadata"])

for paragraph in result["data"]:
    print(paragraph)
