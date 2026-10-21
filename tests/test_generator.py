from app.generation.generator import generate_answer

chunks = [
    {
        "chunk_id": "image_001",
        "parent_id": "parent_001",
        "document_id": "doc_001",
        "location": {"type": "page", "number": 3},
        "content": {
            "type": "image",
            "image_path": "extracted_images/test_shape.png",
        },
        "metadata": {
            "source": "shapes_doc.pdf",
            "file_type": "pdf",
        },
    }
]

question = "What color and shape is in the provided image?"

result = generate_answer(question, chunks)

print("\n--- GENERATED ANSWER ---")
print(result["answer"])

print("\n--- SOURCES ---")
for source in result["sources"]:
    print(source)

print("\n--- MODEL ---")
print(result.get("model", "N/A"))