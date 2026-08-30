import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import CASE_CHUNKS_PATH, EMBEDDINGS_PATH, INDEX_DIR


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_METADATA_PATH = INDEX_DIR / "chunk_embedding_metadata.pkl"


def read_jsonl(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def build_texts_and_metadata(chunks: list[dict]) -> tuple[list[str], list[dict]]:
    texts = []
    metadata = []

    for chunk in chunks:
        chunk_text = chunk.get("chunk_text", "")

        texts.append(chunk_text)

        metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "case_id": chunk["case_id"],
                "title": chunk.get("title", ""),
                "court": chunk.get("court", ""),
                "year": chunk.get("year", ""),
                "chunk_text": chunk_text,
            }
        )

    return texts, metadata


def create_embeddings(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)


def save_metadata(metadata: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(metadata, file)


def test_query_vector(model: SentenceTransformer) -> np.ndarray:
    query = "right to privacy under Article 21"

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("\nTest query embedded successfully.")
    print(f"Query: {query}")
    print(f"Query vector shape: {query_embedding.shape}")

    return query_embedding


def main() -> None:
    chunks = read_jsonl(CASE_CHUNKS_PATH)

    if not chunks:
        raise ValueError("No chunks found. Run Day 5 chunking first.")

    texts, metadata = build_texts_and_metadata(chunks)

    model = load_embedding_model(EMBEDDING_MODEL_NAME)

    embeddings = create_embeddings(model, texts)

    save_embeddings(embeddings, EMBEDDINGS_PATH)
    save_metadata(metadata, EMBEDDING_METADATA_PATH)

    test_query_vector(model)

    print("\nDay 16 Embedding Generation Report")
    print("=" * 70)
    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Metadata saved to: {EMBEDDING_METADATA_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()