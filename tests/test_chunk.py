from app.ingestion.ingestion_manager import process_file
from app.chunking.chunker import create_chunks


file_path = r"E:\\UNILM\\uploads\\RA-UNIT-1.pptx"


document = process_file(file_path)

# Temporary test document ID
document["metadata"]["document_id"] = "test_document_001"


result = create_chunks(document)


print("\nDocument ID:")
print(result["document_id"])


print("\nParents:")

for parent in result["parents"]:

    print(
        "\n",
        parent["parent_id"],
        "|",
        parent["location"],
        "|",
        "Items:",
        parent["metadata"]["item_count"],
        "|",
        "Estimated tokens:",
        parent["metadata"]["estimated_tokens"]
    )


print("\nChildren:")

for child in result["children"]:

    print(
        "\n",
        child["chunk_id"],
        "|",
        "Parent:",
        child["parent_id"],
        "|",
        "Type:",
        child["content"]["type"],
        "|",
        "Tokens:",
        child["metadata"].get(
            "estimated_tokens",
            "N/A"
        )
    )

    if child["content"]["type"] == "text":

        print(
            "Text:",
            child["content"]["text"][:200],
            "..."
        )

    elif child["content"]["type"] == "table":

        print(
            "Table rows:",
            len(child["content"]["data"])
        )

    elif child["content"]["type"] == "image":

        print(
            "Image:",
            child["content"]["image_path"]
        )
        