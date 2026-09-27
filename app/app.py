import streamlit as st
from ingestion.ingestion_manager import process_file
import tempfile
import os
import hashlib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="UNILLM",
    page_icon="📚"
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📚 UNILLM")

st.write(
    "Upload your study materials."
)


uploaded_files = st.file_uploader(
    "Upload your study materials",
    type=["pdf", "pptx", "docx"],
    accept_multiple_files=True
)


# --------------------------------------------------
# Process uploaded files
# --------------------------------------------------

if uploaded_files:

    for uploaded_file in uploaded_files:

        # Skip files that have already been processed
        if uploaded_file.name in st.session_state.processed_files:
            continue

        file_extension = os.path.splitext(
            uploaded_file.name
        )[1].lower()

        # Save uploaded file temporarily
        file_bytes = uploaded_file.read()

        document_id = hashlib.sha256(
            file_bytes
        ).hexdigest()


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        try:

            # Send file to ingestion manager
            result = process_file(temp_path)

            result["metadata"]["document_id"] = document_id

            st.session_state.documents.append(result)

            # Remember processed file
            st.session_state.processed_files.add(
                uploaded_file.name
            )

            st.success(
                f"Processed: {uploaded_file.name}"
            )

        except ValueError as e:

            st.error(
                f"Could not process "
                f"{uploaded_file.name}: {e}"
            )

        except Exception as e:

            st.error(
                f"Unexpected error while processing "
                f"{uploaded_file.name}: {e}"
            )

        finally:

            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)


# --------------------------------------------------
# Display session document count
# --------------------------------------------------

if st.session_state.documents:

    st.success(
        f"Total documents in session: "
        f"{len(st.session_state.documents)}"
    )


# --------------------------------------------------
# Display uploaded documents
# --------------------------------------------------

if st.session_state.documents:

    st.subheader("📚 Documents")

    for document in st.session_state.documents:

        metadata = document["metadata"]

        st.write(
            f"**{metadata['source']}**"
        )

        st.write(
            f"Document ID: `{metadata['document_id']}`"
        )


        st.write(
            f"Type: `{metadata['file_type']}`"
        )

        st.write(
            f"Extracted items: "
            f"{len(document['data'])}"
        )

        st.divider()
