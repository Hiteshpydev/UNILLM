import os
import uuid
import hashlib
import tempfile
from pathlib import Path
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ingestion.ingestion_manager import process_file
from retrieval.indexer import index_document
from retrieval.retriever import retrieve_chunks
from generation.generator import generate_answer
from retrieval.vector_store import (
    get_qdrant_client,
    delete_session_chunks,
)


# ============================================================
# STREAMLIT CLOUD SECRETS
# ============================================================

def _load_streamlit_secrets() -> None:
    """Make Streamlit Cloud secrets available to existing integrations."""

    secret_names = (
        "GEMINI_API_KEY",
        "QDRANT_API_KEY",
        "QDRANT_URL",
        "JINA_API_KEY",
    )

    try:
        for name in secret_names:
            if name not in os.environ and name in st.secrets:
                value = st.secrets[name]

                if isinstance(value, str) and value.strip():
                    os.environ[name] = value.strip()
    except StreamlitSecretNotFoundError:
        # Local development can continue using values from .env.
        return


_load_streamlit_secrets()

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UNILLM",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "documents" not in st.session_state:
    st.session_state.documents = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "session_image_paths" not in st.session_state:
    st.session_state.session_image_paths = set()


def _collect_image_paths(value) -> set[str]:
    """Find extracted image paths in parser and chunker output."""

    paths = set()

    if isinstance(value, dict):
        image_path = value.get("image_path")

        if isinstance(image_path, str) and image_path.strip():
            paths.add(str(Path(image_path).resolve()))

        for item in value.values():
            paths.update(_collect_image_paths(item))

    elif isinstance(value, list):
        for item in value:
            paths.update(_collect_image_paths(item))

    return paths


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📚 UNILLM")

st.write(
    "Upload your study materials to process and index them "
    "for your study session."
)

st.caption(
    f"Study session ID: {st.session_state.session_id}"
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload your study materials",
    type=["pdf", "pptx", "docx"],
    accept_multiple_files=True,
    help="Supported formats: PDF, PowerPoint, and Word documents.",
)


# ============================================================
# PROCESS AND INDEX DOCUMENTS
# ============================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        # Avoid processing the same filename repeatedly
        # during Streamlit reruns.
        if uploaded_file.name in st.session_state.processed_files:
            continue

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1].lower()

        file_bytes = uploaded_file.getvalue()

        document_id = hashlib.sha256(
            file_bytes
        ).hexdigest()

        temp_path = None

        try:

            # ------------------------------------------------
            # Save uploaded file temporarily
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension,
            ) as temp_file:

                temp_file.write(file_bytes)
                temp_path = temp_file.name

            # ------------------------------------------------
            # Parse document
            # ------------------------------------------------

            result = process_file(temp_path)

            # Attach document and session metadata.
            result["metadata"]["document_id"] = document_id
            result["metadata"]["session_id"] = (
                st.session_state.session_id
            )

            # ------------------------------------------------
            # Track extracted images
            # ------------------------------------------------

            # The parser stores images under:
            # data -> content -> image_path

            st.session_state.session_image_paths.update(
                _collect_image_paths(result)
            )

            # ------------------------------------------------
            # Index document in Qdrant
            # ------------------------------------------------

            index_result = index_document(result)

            # ------------------------------------------------
            # Update Streamlit session state
            # ------------------------------------------------

            st.session_state.documents.append(result)

            st.session_state.processed_files.add(
                uploaded_file.name
            )

            st.success(
                f"Successfully processed: {uploaded_file.name}"
            )

            st.write(
                "Indexed chunks: "
                f"{index_result.get('indexed_count', 0)}"
            )

        except ValueError as e:

            st.error(
                f"Could not process {uploaded_file.name}: {e}"
            )

        except Exception as e:

            st.error(
                f"Unexpected error while processing "
                f"{uploaded_file.name}: {e}"
            )

        finally:

            # ------------------------------------------------
            # Remove temporary uploaded file
            # ------------------------------------------------

            if temp_path and os.path.exists(temp_path):

                try:
                    os.remove(temp_path)

                except OSError as e:
                    st.warning(
                        f"Could not remove temporary upload file: {e}"
                    )


# ============================================================
# SESSION DOCUMENT SUMMARY
# ============================================================

if st.session_state.documents:

    st.success(
        "Total documents in this session: "
        f"{len(st.session_state.documents)}"
    )

    st.subheader("📄 Uploaded Documents")

    for document in st.session_state.documents:

        metadata = document.get("metadata", {})

        st.write(
            f"**{metadata.get('source', 'Unknown document')}**"
        )

        st.write(
            f"Document ID: `{metadata.get('document_id', 'N/A')}`"
        )

        st.write(
            f"Type: `{metadata.get('file_type', 'Unknown')}`"
        )

        st.write(
            f"Extracted items: {len(document.get('data', []))}"
        )

        st.divider()


# ============================================================
# QUESTION & ANSWER INTERFACE
# ============================================================

st.divider()
st.header("💬 Ask Questions About Your Study Materials")

user_question = st.text_input(
    "Ask a question:",
    placeholder="e.g., What is machine learning?",
    disabled=not bool(st.session_state.documents),
)

if not st.session_state.documents:
    st.info("💡 Please upload at least one study document to begin asking questions.")

if st.button("Ask Question", type="primary", disabled=not bool(user_question.strip())):

    with st.spinner("Searching materials and generating answer..."):
        try:
            # 1. Retrieve top chunks for active session
            retrieved_chunks = retrieve_chunks(
                question=user_question,
                session_id=st.session_state.session_id,
                limit=5,
            )

            if not retrieved_chunks:
                st.warning("No relevant information found in your uploaded materials for this question.")
            else:
                # 2. Generate grounded answer
                gen_result = generate_answer(
                    question=user_question,
                    chunks=retrieved_chunks,
                )

                # 3. Display Answer
                st.subheader("💡 Answer")
                st.markdown(gen_result["answer"])

                # 4. Display Citations & Sources
                st.subheader("📌 Retrieved Sources")
                with st.expander("View Source Metadata & Chunks", expanded=False):
                    for source in gen_result.get("sources", []):
                        st.markdown(f"**{source.get('label', 'Source')}**")
                        st.json(source)

        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")


# ============================================================
# END STUDY SESSION
# ============================================================

st.divider()

st.header("📚 Before You End Your Study Session")

st.markdown(
    """
    **⚠️ Important: Press the button below before ending your
    study session.**

    Ending your session will:

    - 🗑️ Delete the indexed chunks belonging to this session from Qdrant.
    - 📄 Clear the current session's document list.
    - 🔄 Reset the processed-file tracking.
    - 🖼️ Attempt to remove tracked extracted images.
    - 🆕 Start a fresh study session.

    **Please note:** If you close the browser without pressing the button,
    your indexed data may remain in the vector database.
    """
)

st.subheader("🗑️ End Study Session")


# ============================================================
# SESSION CLEANUP BUTTON
# ============================================================

if st.button(
    "End Study Session",
    type="secondary",
    use_container_width=True,
):

    current_session_id = st.session_state.session_id

    deleted_count = 0
    qdrant_cleanup_error = None

    try:
        client = get_qdrant_client()
        deleted_count = delete_session_chunks(
            client=client,
            session_id=current_session_id,
        )
    except Exception as e:
        qdrant_cleanup_error = str(e)

    image_cleanup_errors = []
    deleted_image_count = 0

    for image_path in list(st.session_state.session_image_paths):
        try:
            path = Path(image_path).resolve()

            if path.is_file():
                path.unlink()
                deleted_image_count += 1

        except OSError as e:
            image_cleanup_errors.append(f"{image_path}: {e}")

    st.session_state.session_image_paths = set()

    if qdrant_cleanup_error:
        st.error(
            "❌ Qdrant cleanup failed, so the study session was not reset. "
            f"Error: {qdrant_cleanup_error}"
        )

        if image_cleanup_errors:
            st.warning(
                "Some extracted images could not be deleted:\n\n"
                + "\n".join(image_cleanup_errors)
            )
        else:
            st.info(
                f"Deleted {deleted_image_count} extracted image(s)."
            )
    else:
        st.session_state.documents = []
        st.session_state.processed_files = set()
        st.session_state.session_id = str(uuid.uuid4())

        st.success(
            "✅ Study session ended successfully. "
            f"Removed {deleted_count} indexed point(s) from Qdrant "
            f"and {deleted_image_count} extracted image(s)."
        )

        if image_cleanup_errors:
            st.warning(
                "Some extracted images could not be deleted:\n\n"
                + "\n".join(image_cleanup_errors)
            )

        st.rerun()