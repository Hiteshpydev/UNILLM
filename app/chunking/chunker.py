import re


def estimate_tokens(text):
    """
    Rough token estimation.

    Used only for chunk sizing.
    We can replace this with the embedding model
    tokenizer later.
    """

    if not text:
        return 0

    return len(text.split())


def create_parents(
    document,
    target_tokens=1800
):
    """
    Create parent units from a normalized document.
    """

    document_id = document["metadata"]["document_id"]

    parents = []

    parent_counter = 1

    current_items = []
    current_tokens = 0

    def create_parent(items):

        nonlocal parent_counter

        if not items:
            return

        first_location = items[0]["location"]
        last_location = items[-1]["location"]

        parent_id = (
            f"{document_id}_parent_{parent_counter}"
        )

        parent = {
            "parent_id": parent_id,
            "document_id": document_id,

            "location": {
                "start": first_location,
                "end": last_location
            },

            "content": items,

            "metadata": {
                "source": document["metadata"]["source"],
                "file_type": document["metadata"]["file_type"],
                "item_count": len(items),
                "estimated_tokens": sum(
                    estimate_tokens(
                        item["content"].get("text", "")
                    )
                    for item in items
                )
            }
        }

        parents.append(parent)

        parent_counter += 1

    for item in document["data"]:

        content = item["content"]
        content_type = content["type"]

        if content_type == "text":

            text = content.get("text", "")

            item_tokens = estimate_tokens(text)

            if (
                current_items
                and current_tokens + item_tokens > target_tokens
            ):

                create_parent(current_items)

                current_items = []
                current_tokens = 0

            current_items.append(item)

            current_tokens += item_tokens

        else:

            if current_items:
                current_items.append(item)

            else:
                create_parent([item])

    if current_items:
        create_parent(current_items)

    return {
        "document_id": document_id,
        "parents": parents
    }


# ============================================================
# CHILD CHUNKING
# ============================================================


def split_text_into_sentences(text):
    """
    Split text into approximate sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def create_text_children(
    text,
    target_tokens=400,
    overlap_tokens=50
):
    """
    Split text into semantic-ish chunks.

    We prefer sentence boundaries and maintain
    a small overlap between chunks.
    """

    sentences = split_text_into_sentences(text)

    chunks = []

    current_sentences = []
    current_tokens = 0

    for sentence in sentences:

        sentence_tokens = estimate_tokens(sentence)

        # --------------------------------
        # Current chunk is full
        # --------------------------------

        if (
            current_sentences
            and current_tokens + sentence_tokens > target_tokens
        ):

            chunks.append(
                " ".join(current_sentences)
            )

            # --------------------------------
            # Create overlap
            # --------------------------------

            overlap_sentences = []

            overlap_count = 0

            for previous_sentence in reversed(
                current_sentences
            ):

                previous_tokens = estimate_tokens(
                    previous_sentence
                )

                if (
                    overlap_count + previous_tokens
                    > overlap_tokens
                ):
                    break

                overlap_sentences.insert(
                    0,
                    previous_sentence
                )

                overlap_count += previous_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_count

        current_sentences.append(sentence)

        current_tokens += sentence_tokens

    # --------------------------------
    # Remaining text
    # --------------------------------

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    return chunks

def create_children(
    parent,
    target_tokens=400,
    overlap_tokens=50
):
    """
    Create child chunks from a parent while preserving
    the original document order and source locations.

    Text items are combined into larger chunks.
    Tables and images remain individual children.
    """

    document_id = parent["document_id"]
    parent_id = parent["parent_id"]

    children = []
    chunk_counter = 1

    # ============================================================
    # Build ordered segments
    # ============================================================

    segments = []
    text_buffer = []

    for item in parent["content"]:

        content = item["content"]
        content_type = content["type"]

        if content_type == "text":

            text = content.get("text", "").strip()

            if text:
                text_buffer.append({
                    "text": text,
                    "location": item["location"]
                })

        else:

            # Flush accumulated text before non-text item
            if text_buffer:
                segments.append({
                    "type": "text",
                    "items": text_buffer
                })

                text_buffer = []

            segments.append({
                "type": content_type,
                "item": item
            })

    # Flush remaining text
    if text_buffer:
        segments.append({
            "type": "text",
            "items": text_buffer
        })

    # ============================================================
    # Create children in original document order
    # ============================================================

    for segment in segments:

        # --------------------------------------------------------
        # TEXT
        # --------------------------------------------------------

        if segment["type"] == "text":

            text_items = segment["items"]

            sentences = []

            # Keep the source location attached to every sentence
            for item in text_items:

                item_sentences = split_text_into_sentences(
                    item["text"]
                )

                for sentence in item_sentences:

                    sentences.append({
                        "text": sentence,
                        "location": item["location"]
                    })

            current_sentences = []
            current_tokens = 0

            for sentence_data in sentences:

                sentence = sentence_data["text"]
                sentence_tokens = estimate_tokens(sentence)

                # ------------------------------------------------
                # Current chunk is full
                # ------------------------------------------------

                if (
                    current_sentences
                    and current_tokens + sentence_tokens
                    > target_tokens
                ):

                    chunk_text = " ".join(
                        item["text"]
                        for item in current_sentences
                    )

                    locations = [
                        item["location"]
                        for item in current_sentences
                    ]

                    # Remove consecutive duplicate locations
                    unique_locations = []

                    for location in locations:
                        if location not in unique_locations:
                            unique_locations.append(location)

                    chunk_id = (
                        f"{parent_id}_chunk_{chunk_counter}"
                    )

                    children.append({

                        "chunk_id": chunk_id,

                        "parent_id": parent_id,

                        "document_id": document_id,

                        "location": unique_locations,

                        "content": {
                            "type": "text",
                            "text": chunk_text
                        },

                        "metadata": {
                            "source": parent["metadata"]["source"],
                            "file_type": parent["metadata"]["file_type"],
                            "estimated_tokens": estimate_tokens(
                                chunk_text
                            )
                        }
                    })

                    chunk_counter += 1

                    # --------------------------------------------
                    # Create overlap
                    # --------------------------------------------

                    overlap_sentences = []
                    overlap_count = 0

                    for previous_sentence in reversed(
                        current_sentences
                    ):

                        previous_tokens = estimate_tokens(
                            previous_sentence["text"]
                        )

                        if (
                            overlap_count + previous_tokens
                            > overlap_tokens
                        ):
                            break

                        overlap_sentences.insert(
                            0,
                            previous_sentence
                        )

                        overlap_count += previous_tokens

                    current_sentences = overlap_sentences
                    current_tokens = overlap_count

                current_sentences.append(
                    sentence_data
                )

                current_tokens += sentence_tokens

            # ----------------------------------------------------
            # Final text chunk
            # ----------------------------------------------------

            if current_sentences:

                chunk_text = " ".join(
                    item["text"]
                    for item in current_sentences
                )

                locations = [
                    item["location"]
                    for item in current_sentences
                ]

                unique_locations = []

                for location in locations:
                    if location not in unique_locations:
                        unique_locations.append(location)

                chunk_id = (
                    f"{parent_id}_chunk_{chunk_counter}"
                )

                children.append({

                    "chunk_id": chunk_id,

                    "parent_id": parent_id,

                    "document_id": document_id,

                    "location": unique_locations,

                    "content": {
                        "type": "text",
                        "text": chunk_text
                    },

                    "metadata": {
                        "source": parent["metadata"]["source"],
                        "file_type": parent["metadata"]["file_type"],
                        "estimated_tokens": estimate_tokens(
                            chunk_text
                        )
                    }
                })

                chunk_counter += 1

        # --------------------------------------------------------
        # TABLE / IMAGE
        # --------------------------------------------------------

        else:

            item = segment["item"]
            content = item["content"]
            content_type = content["type"]

            chunk_id = (
                f"{parent_id}_chunk_{chunk_counter}"
            )

            child_content = {
                "type": content_type
            }

            if content_type == "table":

                child_content["data"] = (
                    content["data"]
                )

            elif content_type == "image":

                child_content["image_path"] = (
                    content["image_path"]
                )

            children.append({

                "chunk_id": chunk_id,

                "parent_id": parent_id,

                "document_id": document_id,

                "location": item["location"],

                "content": child_content,

                "metadata": {
                    "source": parent["metadata"]["source"],
                    "file_type": parent["metadata"]["file_type"],
                    "location_type": item["location"]["type"],
                    "location_number": item["location"]["number"],
                    "estimated_tokens": estimate_tokens(
                        content.get("text", "")
                    )
                }
            })

            chunk_counter += 1

    return children

def create_chunks(
    document,
    parent_target_tokens=1800,
    child_target_tokens=400,
    child_overlap_tokens=0
):
    """
    Complete parent-child chunking pipeline.
    """

    parent_result = create_parents(
        document,
        target_tokens=parent_target_tokens
    )

    all_children = []

    for parent in parent_result["parents"]:

        children = create_children(
            parent,
            target_tokens=child_target_tokens,
            overlap_tokens=child_overlap_tokens
        )

        all_children.extend(children)

    return {
        "document_id": parent_result["document_id"],
        "parents": parent_result["parents"],
        "children": all_children
    }
    