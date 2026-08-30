import re


def split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "chunk_text": chunk_text,
                    "start_offset": start,
                    "end_offset": end,
                    "chunk_size": len(chunk_text),
                }
            )

        if end == text_length:
            break

        start = end - overlap

    return chunks


def split_paragraph_aware(text: str, chunk_size: int, overlap: int) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return split_text_with_overlap(text, chunk_size, overlap)

    chunks = []
    current_text = ""
    current_start = 0
    cursor = 0

    for paragraph in paragraphs:
        paragraph_start = text.find(paragraph, cursor)
        paragraph_end = paragraph_start + len(paragraph)

        if not current_text:
            current_start = paragraph_start

        candidate = (current_text + "\n\n" + paragraph).strip()

        if len(candidate) <= chunk_size:
            current_text = candidate
        else:
            if current_text:
                chunks.append(
                    {
                        "chunk_text": current_text.strip(),
                        "start_offset": current_start,
                        "end_offset": paragraph_start,
                        "chunk_size": len(current_text.strip()),
                    }
                )

            if len(paragraph) > chunk_size:
                large_chunks = split_text_with_overlap(
                    paragraph,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                for item in large_chunks:
                    chunks.append(
                        {
                            "chunk_text": item["chunk_text"],
                            "start_offset": paragraph_start + item["start_offset"],
                            "end_offset": paragraph_start + item["end_offset"],
                            "chunk_size": item["chunk_size"],
                        }
                    )

                current_text = ""
            else:
                current_text = paragraph
                current_start = paragraph_start

        cursor = paragraph_end

    if current_text:
        chunks.append(
            {
                "chunk_text": current_text.strip(),
                "start_offset": current_start,
                "end_offset": len(text),
                "chunk_size": len(current_text.strip()),
            }
        )

    return chunks
