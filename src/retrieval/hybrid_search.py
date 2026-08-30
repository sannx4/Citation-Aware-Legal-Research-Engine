import argparse
import pickle
import re
import sys
from pathlib import Path
import csv

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    REPORTS_DIR,
)


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_METADATA_PATH = INDEX_DIR / "chunk_embedding_metadata.pkl"
HYBRID_REPORT_PATH = REPORTS_DIR / "hybrid_search_results.csv"


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def load_bm25_index(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {path}. Run Day 15 first."
        )

    with path.open("rb") as file:
        return pickle.load(file)


def load_faiss_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {path}. Run Day 17 first."
        )

    return faiss.read_index(str(path))


def load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding metadata not found: {path}. Run Day 16 first."
        )

    with path.open("rb") as file:
        return pickle.load(file)


def normalize_scores(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}

    values = list(score_map.values())
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        return {key: 1.0 for key in score_map}

    normalized = {}

    for key, value in score_map.items():
        normalized[key] = (value - min_score) / (max_score - min_score)

    return normalized


def bm25_search_scores(
    query: str,
    bm25_data: dict,
    top_k: int,
) -> dict[str, dict]:

    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results = {}

    for index in ranked_indices:
        chunk = chunks[index]
        chunk_id = chunk["chunk_id"]

        results[chunk_id] = {
            "score": float(scores[index]),
            "metadata": {
                "chunk_id": chunk["chunk_id"],
                "case_id": chunk["case_id"],
                "title": chunk.get("title", ""),
                "court": chunk.get("court", ""),
                "year": chunk.get("year", ""),
                "chunk_text": chunk.get("chunk_text", ""),
            },
        }

    return results


def faiss_search_scores(
    query: str,
    faiss_index: faiss.Index,
    metadata: list[dict],
    top_k: int,
) -> dict[str, dict]:

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = faiss_index.search(query_vector, top_k)

    results = {}

    for position, index in enumerate(indices[0]):
        if index == -1:
            continue

        item = metadata[index]
        chunk_id = item["chunk_id"]

        results[chunk_id] = {
            "score": float(scores[0][position]),
            "metadata": item,
        }

    return results


def hybrid_search(
    query: str,
    bm25_data: dict,
    faiss_index: faiss.Index,
    metadata: list[dict],
    top_k: int = 5,
    candidate_k: int = 20,
    alpha: float = 0.5,
) -> list[dict]:

    bm25_results = bm25_search_scores(query, bm25_data, candidate_k)
    faiss_results = faiss_search_scores(query, faiss_index, metadata, candidate_k)

    bm25_raw_scores = {
        chunk_id: data["score"]
        for chunk_id, data in bm25_results.items()
    }

    faiss_raw_scores = {
        chunk_id: data["score"]
        for chunk_id, data in faiss_results.items()
    }

    bm25_norm = normalize_scores(bm25_raw_scores)
    faiss_norm = normalize_scores(faiss_raw_scores)

    all_chunk_ids = set(bm25_results) | set(faiss_results)

    combined = []

    for chunk_id in all_chunk_ids:
        bm25_score = bm25_norm.get(chunk_id, 0.0)
        faiss_score = faiss_norm.get(chunk_id, 0.0)

        final_score = (alpha * bm25_score) + ((1 - alpha) * faiss_score)

        if chunk_id in bm25_results:
            item = bm25_results[chunk_id]["metadata"]
        else:
            item = faiss_results[chunk_id]["metadata"]

        combined.append(
            {
                "chunk_id": chunk_id,
                "case_id": item["case_id"],
                "title": item.get("title", ""),
                "court": item.get("court", ""),
                "year": item.get("year", ""),
                "bm25_score": round(bm25_score, 4),
                "faiss_score": round(faiss_score, 4),
                "hybrid_score": round(final_score, 4),
                "snippet": " ".join(item.get("chunk_text", "").split())[:250] + "...",
            }
        )

    combined.sort(key=lambda row: row["hybrid_score"], reverse=True)

    ranked_results = []

    for rank, row in enumerate(combined[:top_k], start=1):
        row["rank"] = rank
        ranked_results.append(row)

    return ranked_results


def write_comparison_report(
    query: str,
    results: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Day 18 Hybrid Search Comparison",
        "",
        f"Query: `{query}`",
        "",
        "## What this report shows",
        "",
        "This report compares combined BM25 keyword retrieval and FAISS semantic retrieval.",
        "BM25 helps with exact legal terms. FAISS helps with meaning-based search.",
        "",
        "## Results",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f"### Rank {result['rank']}: {result['title']}",
                "",
                f"- Case ID: `{result['case_id']}`",
                f"- Chunk ID: `{result['chunk_id']}`",
                f"- BM25 score: `{result['bm25_score']}`",
                f"- FAISS score: `{result['faiss_score']}`",
                f"- Hybrid score: `{result['hybrid_score']}`",
                "",
                f"Snippet: {result['snippet']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitation",
            "",
            "The hybrid score depends on the alpha weight. A small corpus may produce unstable ranking.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_results(query: str, results: list[dict]) -> None:
    print("\nDay 18 Hybrid Retrieval Results")
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Hybrid Score: {result['hybrid_score']}")
        print(f"BM25 Score: {result['bm25_score']}")
        print(f"FAISS Score: {result['faiss_score']}")
        print(f"Case ID: {result['case_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Title: {result['title']}")
        print(f"Court: {result['court']}")
        print(f"Year: {result['year']}")
        print(f"Snippet: {result['snippet']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + FAISS retrieval over legal chunks."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query, example: Section 420 cheating",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25. 0.5 means BM25 and FAISS are equal.",
    )

    return parser.parse_args()

def write_hybrid_results_csv(results: list[dict], output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "case_id",
        "chunk_id",
        "title",
        "court",
        "year",
        "hybrid_score",
        "bm25_score",
        "faiss_score",
        "snippet",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    "rank": row.get("rank", ""),
                    "case_id": row.get("case_id", ""),
                    "chunk_id": row.get("chunk_id", ""),
                    "title": row.get("title", ""),
                    "court": row.get("court", ""),
                    "year": row.get("year", ""),
                    "hybrid_score": row.get("hybrid_score", ""),
                    "bm25_score": row.get("bm25_score", ""),
                    "faiss_score": row.get("faiss_score", ""),
                    "snippet": row.get("snippet", ""),
                }
            )



def main() -> None:
    args = parse_args()

    bm25_data = load_bm25_index(BM25_INDEX_PATH)
    faiss_index = load_faiss_index(FAISS_INDEX_PATH)
    metadata = load_metadata(EMBEDDING_METADATA_PATH)

    results = hybrid_search(
        query=args.query,
        bm25_data=bm25_data,
        faiss_index=faiss_index,
        metadata=metadata,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        alpha=args.alpha,
    )

    print_results(args.query, results)

    write_comparison_report(
        query=args.query,
        results=results,
        output_path=HYBRID_REPORT_PATH,
    )
    
    from src.config import HYBRID_SEARCH_RESULTS_PATH

    write_hybrid_results_csv(
        results,
        HYBRID_SEARCH_RESULTS_PATH
    )

    print("\nComparison report saved to:")
    print(HYBRID_REPORT_PATH)


if __name__ == "__main__":
    main()