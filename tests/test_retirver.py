import sys
from pathlib import Path
from unittest import result

# Locate the app directory.
PROJECT_DIR = Path(__file__).resolve().parent
APP_DIR = PROJECT_DIR / "app"

print("Test file:", __file__)
print("App directory:", APP_DIR)
print("App directory exists:", APP_DIR.is_dir())

# Make the app modules available to Python.
sys.path.insert(0, str(APP_DIR))

from retrieval.retriever import retrieve_chunks


def main():
    question = input("Enter a question about your study material: ").strip()

    if not question:
        print("Please enter a question.")
        return

    results = retrieve_chunks(question, limit=5)

    if not results:
        print("No matching chunks found.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Result: {index}")
        print("RRF Score:", result.get("score"))
        print("Reranker Score:", result.get("rerank_score"))
        print(f"Location: {result.get('location')}")

        content = result.get("content") or {}
        print("Content type:", content.get("type"))
        content_type = content.get("type")
        if content_type == "text":
            text = content.get("text", "")
            print("Character count:", len(text))
            print("Word count:", len(text.split()))

        if content.get("type") == "text":
            print(content.get("text", ""))
        elif content.get("type") == "table":
            print(content.get("data", ""))
        elif content.get("type") == "image":
            print("Image:", content.get("image_path", ""))


if __name__ == "__main__":
    main() 