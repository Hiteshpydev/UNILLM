# 📚 UNILLM — Multimodal RAG Study Assistant

UNILLM is a multimodal Retrieval-Augmented Generation (RAG) study material assistant designed to help students interact with mixed-format learning resources. The application ingests PDFs, PowerPoint presentations, and Word documents, extracts text, structured tables, and embedded diagrams, and uses hybrid retrieval with Gemini 3.8 Flash to generate grounded, cited answers.

---

## 🛠️ Technology Stack

* **Frontend / User Interface:** [Streamlit](https://streamlit.io/)
* **Generative AI / LLM:** Google Gemini (`gemini-3.8-flash`) via `google-genai` SDK
* **Vector Database:** [Qdrant Cloud](https://qdrant.tech/) (Dense Cosine Similarity + Qdrant Sparse BM25)
* **Reranking:** Jina AI Reranker (`jina-reranker-m0`)
* **Document Parsing & OCR:** PyMuPDF (`fitz`), `python-pptx`, `python-docx`, Tesseract OCR (`pytesseract`), Pillow
* **Language & Runtime:** Python 3.10+

---

## 🏗️ System Architecture

UNILLM processes documents through a multi-stage pipeline designed for context preservation and accurate retrieval.

```
┌────────────────────────────────────────────────────────┐
│                    Streamlit UI                        │
└───────────┬────────────────────────────────┬───────────┘
            │ 1. Upload Documents            │ 2. Ask Question
            ▼                                ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│   Ingestion Pipeline    │      │   Hybrid Retriever      │
│  (PyMuPDF, pptx, docx)  │      │  (Dense + Qdrant BM25)  │
└───────────┬─────────────┘      └───────────┬─────────────┘
            │ Data Contract                  │ Fusion & Rerank
            ▼                                ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│    Parent-Child         │      │      Jina Reranker      │
│      Chunker            │      │   (jina-reranker-m0)    │
└───────────┬─────────────┘      └───────────┬─────────────┘
            │ Candidates                     │ Top Context Chunks
            ▼                                ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│     Qdrant Cloud        │      │    Gemini Generator     │
│    (Vector Store)       │      │   (gemini-3.8-flash)    │
└─────────────────────────┘      └─────────────────────────┘
```

---

## 🧩 Detailed Component Overview

### 1. Ingestion Engine (`app/ingestion/`)
* **PDF Processing (`pdf_parser.py`):** Uses PyMuPDF (`fitz`) to extract native text and structured tables. Pages lacking native text fall back to Tesseract OCR (`pytesseract`). Embedded raster images are extracted to the `extracted_images/` directory.
* **PowerPoint Processing (`pptx_parser.py`):** Parses slide decks via `python-pptx`, capturing slide text, structured tables, and diagram assets.
* **Word Document Processing (`docx_parser.py`):** Extracts paragraphs, tables, and inline images using `python-docx`.
* **Normalized Data Schema:** All parsers conform to a standardized output contract:
  ```json
  {
      "location": {"type": "page | slide | paragraph", "number": 1},
      "content": {
          "type": "text | table | image",
          "text": "...",         
          "data": [...],        
          "image_path": "..."   
      }
  }
  ```

---

### 2. Structure-Aware Parent-Child Chunking (`app/chunking/`)
To preserve broader contextual explanations while maintaining search precision:
* **Parent Chunks (~1,800 tokens):** Group sequential text and structural elements into large context blocks to avoid losing thematic context.
* **Child Chunks (~800 tokens, ~100 token overlap):** Created directly from parent blocks for index granularity.
  * **Text Children:** Built by combining parent text, sentence-splitting, and sliding-window packing.
  * **Table Children:** Standalone chunks preserving structured tabular rows and headers.
  * **Image Children:** Dedicated chunks retaining the image asset path (`extracted_images/...`).

---

### 3. Vector Storage & Hybrid Retrieval (`app/retrieval/`)
* **Qdrant Vector Database (`vector_store.py`):** Stores dense embeddings and sparse text representations in the `unillm_chunks` collection.
* **Hybrid Search (`retriever.py`):** Combines dense vector similarity with sparse Qdrant BM25 keyword matching using Reciprocal Rank Fusion (RRF).
* **Jina Reranker (`reranker.py`):** Re-scores candidates using the `jina-reranker-m0` API model across text, formatted table markdown, and base64-encoded 512×512 RGB image data.
* **Image Embeddings (`embedder.py`):** Resizes every document image to 512×512 RGB pixels before generating its Jina CLIP embedding.

---

### 4. Multimodal Answer Generation (`app/generation/`)
* **Model:** Powered by `gemini-3.8-flash` via the `google-genai` SDK.
* **Multimodal Context Loading:**
  * **Text:** Injected directly into the model context.
  * **Tables:** Converted to Markdown/JSON formats.
  * **Images:** Loaded from disk and passed as raw binary image parts to Gemini's visual context.
* **Source Citations:** Generates grounded answers with strict `[Source N]` inline citations linked to source metadata.

---

## 🔑 Environment Configuration (`.env.example`)

The application requires API credentials for Gemini, Qdrant, and Jina AI. Copy `.env.example` to `.env` and fill in your keys:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
JINA_API_KEY=your_jina_api_key_here
```

---

## 🔒 Session Isolation & Data Privacy

UNILLM uses session-scoped isolation to guarantee data privacy:
1. **Session UUID:** A unique `session_id` is assigned upon starting a session.
2. **Payload Tagging:** Every chunk uploaded is tagged with the current `session_id` in Qdrant.
3. **Hard Filtering:** Retrieval queries enforce a mandatory Qdrant payload filter (`models.FieldCondition(key="session_id", match=session_id)`), preventing cross-session data leakage.
4. **Session Cleanup:** Ending a study session purges all indexed vectors from Qdrant, deletes extracted image files from disk, and resets state variables.

---

## 📂 Project Directory Structure

```text
E:\UNILM\
├── app\
│   ├── app.py                      # Main Streamlit Application UI
│   ├── ingestion\                  # Document Ingestion Engine
│   │   ├── docx_parser.py          # Word document parser
│   │   ├── ingestion_manager.py    # Multi-format router
│   │   ├── ocr.py                  # Tesseract OCR engine wrapper
│   │   ├── pdf_parser.py          # PyMuPDF & OCR parser
│   │   └── pptx_parser.py         # PowerPoint parser
│   ├── chunking\                   # Parent-child chunking logic
│   │   └── chunker.py
│   ├── embeddings\                 # Embedding helpers
│   │   └── embedder.py
│   ├── retrieval\                  # Search, reranking, & indexing
│   │   ├── indexer.py              # Chunk indexing pipeline
│   │   ├── reranker.py             # Jina M0 reranker integration
│   │   ├── retriever.py            # Hybrid RRF retriever
│   │   └── vector_store.py         # Qdrant client & collection management
│   └── generation\                 # Gemini multimodal generation
│       └── generator.py
├── extracted_images\               # Local runtime storage for extracted images
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependency configuration
└── README.md                       # Project documentation
``` 