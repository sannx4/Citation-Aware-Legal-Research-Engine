import argparse
import json
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import CASE_CHUNKS_PATH, BM25_INDEX_PATH


def read_jsonl(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def build_bm25_index(chunks: list[dict]) -> dict:
    tokenized_corpus = []

    for chunk in chunks:
        chunk_text = chunk.get("chunk_text", "")
        tokenized_corpus.append(tokenize(chunk_text))

    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "bm25": bm25,
        "chunks": chunks,
        "tokenized_corpus": tokenized_corpus,
    }


def save_index(index_data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as file:
        pickle.dump(index_data, file)


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {index_path}. Run with --rebuild first."
        )

    with index_path.open("rb") as file:
        return pickle.load(file)


def make_snippet(text: str, limit: int = 250) -> str:
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def bm25_search(query: str, index_data: dict, top_k: int = 5) -> list[dict]:
    bm25 = index_data["bm25"]
    chunks = index_data["chunks"]

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for rank, index in enumerate(ranked_indices[:top_k], start=1):
        chunk = chunks[index]

        results.append(
            {
                "rank": rank,
                "score": round(float(scores[index]), 4),
                "chunk_id": chunk["chunk_id"],
                "case_id": chunk["case_id"],
                "title": chunk.get("title", ""),
                "court": chunk.get("court", ""),
                "year": chunk.get("year", ""),
                "snippet": make_snippet(chunk.get("chunk_text", "")),
            }
        )

    return results


def print_results(query: str, results: list[dict]) -> None:
    print("\nDay 15 BM25 Search Results")
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
        description="BM25 keyword search over legal judgment chunks."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query, example: Article 21 privacy",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild BM25 index before searching.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rebuild or not BM25_INDEX_PATH.exists():
        chunks = read_jsonl(CASE_CHUNKS_PATH)
        index_data = build_bm25_index(chunks)
        save_index(index_data, BM25_INDEX_PATH)

        print("\nBM25 index created.")
        print(f"Chunks indexed: {len(chunks)}")
        print(f"Saved to: {BM25_INDEX_PATH}")

    index_data = load_index(BM25_INDEX_PATH)

    results = bm25_search(
        query=args.query,
        index_data=index_data,
        top_k=args.top_k,
    )

    print_results(args.query, results)


if __name__ == "__main__":
    main()