import csv
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import REPORTS_DIR
from src.retrieval.bm25_search import (
    BM25_INDEX_PATH,
    load_index as load_bm25_index,
    bm25_search,
)
from src.retrieval.faiss_index import (
    FAISS_INDEX_PATH,
    EMBEDDING_METADATA_PATH,
    load_faiss_index,
    load_metadata,
    semantic_search,
)
from src.retrieval.hybrid_search import (
    load_bm25_index as load_hybrid_bm25,
    load_faiss_index as load_hybrid_faiss,
    hybrid_search,
)


RETRIEVAL_QUERIES_PATH = ROOT_DIR / "data" / "eval" / "retrieval_queries.csv"
RETRIEVAL_REPORT_PATH = REPORTS_DIR / "retrieval_evaluation_report.csv"
RETRIEVAL_NOTES_PATH = REPORTS_DIR / "retrieval_evaluation_notes.md"


def parse_relevant_cases(value: str) -> set[str]:
    return {case_id.strip() for case_id in value.split(";") if case_id.strip()}


def recall_at_k(results: list[dict], relevant_case_ids: set[str], k: int) -> float:
    returned_case_ids = {
        result["case_id"]
        for result in results[:k]
    }

    if not relevant_case_ids:
        return 0.0

    found = returned_case_ids & relevant_case_ids

    return len(found) / len(relevant_case_ids)


def reciprocal_rank(results: list[dict], relevant_case_ids: set[str]) -> float:
    for index, result in enumerate(results, start=1):
        if result["case_id"] in relevant_case_ids:
            return 1 / index

    return 0.0


def evaluate_system(
    system_name: str,
    queries: pd.DataFrame,
    search_function,
    top_k: int = 5,
) -> list[dict]:

    rows = []

    for _, row in queries.iterrows():
        query = row["query"]
        relevant_case_ids = parse_relevant_cases(row["relevant_case_ids"])

        results = search_function(query, top_k)

        r_at_1 = recall_at_k(results, relevant_case_ids, 1)
        r_at_3 = recall_at_k(results, relevant_case_ids, 3)
        r_at_5 = recall_at_k(results, relevant_case_ids, 5)
        rr = reciprocal_rank(results, relevant_case_ids)

        top_cases = ";".join(
            result["case_id"]
            for result in results[:top_k]
        )

        rows.append(
            {
                "system": system_name,
                "query": query,
                "relevant_case_ids": ";".join(sorted(relevant_case_ids)),
                "top_cases": top_cases,
                "recall_at_1": round(r_at_1, 4),
                "recall_at_3": round(r_at_3, 4),
                "recall_at_5": round(r_at_5, 4),
                "reciprocal_rank": round(rr, 4),
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "system",
        "query",
        "relevant_case_ids",
        "top_cases",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict]) -> list[dict]:
    dataframe = pd.DataFrame(rows)

    summary = []

    for system_name, group in dataframe.groupby("system"):
        summary.append(
            {
                "system": system_name,
                "mean_recall_at_1": round(group["recall_at_1"].mean(), 4),
                "mean_recall_at_3": round(group["recall_at_3"].mean(), 4),
                "mean_recall_at_5": round(group["recall_at_5"].mean(), 4),
                "mrr": round(group["reciprocal_rank"].mean(), 4),
            }
        )

    return summary


def write_notes(summary_rows: list[dict], output_path: Path) -> None:
    lines = [
        "# Day 19 Retrieval Evaluation Report",
        "",
        "## What was evaluated",
        "",
        "This evaluation compares BM25 keyword search, FAISS semantic search, and Hybrid retrieval.",
        "",
        "## Metrics",
        "",
        "- Recall@k checks whether the correct case appears in the top-k results.",
        "- MRR checks how early the first correct result appears.",
        "",
        "## Summary",
        "",
    ]

    for row in summary_rows:
        lines.append(
            f"- {row['system']}: "
            f"Recall@1={row['mean_recall_at_1']}, "
            f"Recall@3={row['mean_recall_at_3']}, "
            f"Recall@5={row['mean_recall_at_5']}, "
            f"MRR={row['mrr']}"
        )

    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "This evaluation uses a small manually labeled query set. Larger evaluation data is needed for reliable conclusions.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not RETRIEVAL_QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {RETRIEVAL_QUERIES_PATH}"
        )

    queries = pd.read_csv(RETRIEVAL_QUERIES_PATH)

    bm25_data = load_bm25_index(BM25_INDEX_PATH)
    faiss_index = load_faiss_index(FAISS_INDEX_PATH)
    metadata = load_metadata(EMBEDDING_METADATA_PATH)

    hybrid_bm25 = load_hybrid_bm25(BM25_INDEX_PATH)
    hybrid_faiss = load_hybrid_faiss(FAISS_INDEX_PATH)

    def run_bm25(query: str, top_k: int) -> list[dict]:
        return bm25_search(query, bm25_data, top_k)

    def run_faiss(query: str, top_k: int) -> list[dict]:
        return semantic_search(query, faiss_index, metadata, top_k)

    def run_hybrid(query: str, top_k: int) -> list[dict]:
        return hybrid_search(
            query=query,
            bm25_data=hybrid_bm25,
            faiss_index=hybrid_faiss,
            metadata=metadata,
            top_k=top_k,
            candidate_k=20,
            alpha=0.5,
        )

    all_rows = []

    all_rows.extend(
        evaluate_system("BM25", queries, run_bm25)
    )

    all_rows.extend(
        evaluate_system("FAISS", queries, run_faiss)
    )

    all_rows.extend(
        evaluate_system("HYBRID", queries, run_hybrid)
    )

    write_csv(all_rows, RETRIEVAL_REPORT_PATH)

    summary_rows = summarize(all_rows)
    write_notes(summary_rows, RETRIEVAL_NOTES_PATH)

    print("\nDay 19 Retrieval Evaluation Report")
    print("=" * 70)

    for row in summary_rows:
        print(
            f"{row['system']} | "
            f"Recall@1={row['mean_recall_at_1']} | "
            f"Recall@3={row['mean_recall_at_3']} | "
            f"Recall@5={row['mean_recall_at_5']} | "
            f"MRR={row['mrr']}"
        )

    print("=" * 70)
    print(f"Detailed report saved to: {RETRIEVAL_REPORT_PATH}")
    print(f"Notes saved to: {RETRIEVAL_NOTES_PATH}")


if __name__ == "__main__":
    main()