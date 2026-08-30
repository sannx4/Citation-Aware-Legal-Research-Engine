import argparse
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    INDEX_DIR,
)


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_METADATA_PATH = INDEX_DIR / "chunk_embedding_metadata.pkl"


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {path}. "
            "Run Day 16 first: python src/retrieval/embed_chunks.py"
        )

    embeddings = np.load(path)

    return embeddings.astype("float32")


def load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. "
            "Run Day 16 first."
        )

    with path.open("rb") as file:
        return pickle.load(file)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {path}. Run this script with --rebuild."
        )

    return faiss.read_index(str(path))


def encode_query(query: str) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return query_vector.astype("float32")


def semantic_search(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = 5,
) -> list[dict]:

    query_vector = encode_query(query)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for rank, idx in enumerate(indices[0], start=1):
        if idx == -1:
            continue

        item = metadata[idx]

        results.append(
            {
                "rank": rank,
                "score": round(float(scores[0][rank - 1]), 4),
                "chunk_id": item["chunk_id"],
                "case_id": item["case_id"],
                "title": item.get("title", ""),
                "court": item.get("court", ""),
                "year": item.get("year", ""),
                "snippet": " ".join(item.get("chunk_text", "").split())[:250] + "...",
            }
        )

    return results


def print_results(query: str, results: list[dict]) -> None:
    print("\nDay 17 FAISS Semantic Search Results")
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']}")
        print(f"Case ID: {result['case_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Title: {result['title']}")
        print(f"Court: {result['court']}")
        print(f"Year: {result['year']}")
        print(f"Snippet: {result['snippet']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAISS semantic search over legal chunk embeddings."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query, example: right to privacy",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rebuild or not FAISS_INDEX_PATH.exists():
        embeddings = load_embeddings(EMBEDDINGS_PATH)
        index = build_faiss_index(embeddings)
        save_faiss_index(index, FAISS_INDEX_PATH)

        print("\nFAISS index created.")
        print(f"Vectors indexed: {embeddings.shape[0]}")
        print(f"Vector dimension: {embeddings.shape[1]}")
        print(f"Saved to: {FAISS_INDEX_PATH}")

    index = load_faiss_index(FAISS_INDEX_PATH)
    metadata = load_metadata(EMBEDDING_METADATA_PATH)

    results = semantic_search(
        query=args.query,
        index=index,
        metadata=metadata,
        top_k=args.top_k,
    )

    print_results(args.query, results)


if __name__ == "__main__":
    main()