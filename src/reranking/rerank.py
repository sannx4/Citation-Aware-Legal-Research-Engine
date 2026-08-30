import argparse
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    REPORTS_DIR,
)

from src.retrieval.hybrid_search import (
    load_bm25_index,
    load_faiss_index,
    load_metadata,
    hybrid_search,
)


RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_METADATA_PATH = INDEX_DIR / "chunk_embedding_metadata.pkl"
RERANK_REPORT_PATH = REPORTS_DIR / "rerank_examples.md"


def load_reranker() -> CrossEncoder:
    print(f"Loading reranker model: {RERANK_MODEL_NAME}")
    return CrossEncoder(RERANK_MODEL_NAME)


def rerank_results(
    query: str,
    candidates: list[dict],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[dict]:

    pairs = []

    for candidate in candidates:
        pairs.append(
            [
                query,
                candidate["snippet"],
            ]
        )

    scores = reranker.predict(pairs)

    reranked = []

    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["rerank_score"] = round(float(score), 4)
        reranked.append(item)

    reranked.sort(
        key=lambda row: row["rerank_score"],
        reverse=True,
    )

    final_results = []

    for rank, item in enumerate(reranked[:top_k], start=1):
        item["rerank_rank"] = rank
        final_results.append(item)

    return final_results


def write_rerank_report(
    query: str,
    original_results: list[dict],
    reranked_results: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Day 20 Reranking Report",
        "",
        f"Query: `{query}`",
        "",
        "## What this report shows",
        "",
        "Hybrid search first retrieves candidate chunks. The cross-encoder then reranks those chunks by reading the query and chunk together.",
        "",
        "## Before reranking",
        "",
    ]

    for result in original_results[:5]:
        lines.extend(
            [
                f"### Hybrid Rank {result['rank']}: {result['title']}",
                "",
                f"- Case ID: `{result['case_id']}`",
                f"- Chunk ID: `{result['chunk_id']}`",
                f"- Hybrid score: `{result['hybrid_score']}`",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## After reranking",
            "",
        ]
    )

    for result in reranked_results:
        lines.extend(
            [
                f"### Rerank {result['rerank_rank']}: {result['title']}",
                "",
                f"- Case ID: `{result['case_id']}`",
                f"- Chunk ID: `{result['chunk_id']}`",
                f"- Hybrid score: `{result['hybrid_score']}`",
                f"- Rerank score: `{result['rerank_score']}`",
                "",
                f"Snippet: {result['snippet']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitation",
            "",
            "Cross-encoder reranking is slower than BM25 or FAISS because it scores each query-chunk pair separately.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def print_results(query: str, results: list[dict]) -> None:
    print("\nDay 20 Cross-Encoder Reranking Results")
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    for result in results:
        print(f"\nRerank: {result['rerank_rank']}")
        print(f"Rerank Score: {result['rerank_score']}")
        print(f"Hybrid Score: {result['hybrid_score']}")
        print(f"Case ID: {result['case_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Title: {result['title']}")
        print(f"Court: {result['court']}")
        print(f"Year: {result['year']}")
        print(f"Snippet: {result['snippet']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-encoder reranking over hybrid search candidates."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query, example: Article 21 privacy",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bm25_data = load_bm25_index(BM25_INDEX_PATH)
    faiss_index = load_faiss_index(FAISS_INDEX_PATH)
    metadata = load_metadata(EMBEDDING_METADATA_PATH)

    candidates = hybrid_search(
        query=args.query,
        bm25_data=bm25_data,
        faiss_index=faiss_index,
        metadata=metadata,
        top_k=args.candidate_k,
        candidate_k=args.candidate_k,
        alpha=0.5,
    )

    reranker = load_reranker()

    reranked_results = rerank_results(
        query=args.query,
        candidates=candidates,
        reranker=reranker,
        top_k=args.top_k,
    )

    print_results(args.query, reranked_results)

    write_rerank_report(
        query=args.query,
        original_results=candidates,
        reranked_results=reranked_results,
        output_path=RERANK_REPORT_PATH,
    )

    print("\nRerank report saved to:")
    print(RERANK_REPORT_PATH)


if __name__ == "__main__":
    main()