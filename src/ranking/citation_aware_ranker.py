import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    AUTHORITY_SCORES_PATH,
    STATUTE_MENTIONS_PATH,
    CITATION_AWARE_RANKING_PATH,
    HYBRID_SEARCH_RESULTS_PATH,
    RERANKED_RESULTS_PATH,
)


CURRENT_YEAR = 2026


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return [1.0 for _ in values]

    return [
        (value - min_value) / (max_value - min_value)
        for value in values
    ]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def get_score_column(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def load_hybrid_results(path: Path) -> pd.DataFrame:
    dataframe = load_csv(path)

    if dataframe.empty:
        raise FileNotFoundError(
            f"Hybrid results not found: {path}. "
            "Run hybrid search first."
        )

    score_column = get_score_column(
        dataframe,
        ["hybrid_score", "final_score", "score", "combined_score"],
    )

    if score_column is None:
        raise ValueError(
            "Hybrid results must contain one score column: "
            "hybrid_score/final_score/score/combined_score"
        )

    dataframe["hybrid_score_raw"] = dataframe[score_column].astype(float)

    normalized = normalize(dataframe["hybrid_score_raw"].tolist())
    dataframe["hybrid_score"] = normalized

    return dataframe


def load_rerank_scores(path: Path) -> dict[str, float]:
    dataframe = load_csv(path)

    if dataframe.empty:
        return {}

    score_column = get_score_column(
        dataframe,
        ["rerank_score", "cross_encoder_score", "score", "final_score"],
    )

    if score_column is None:
        return {}

    dataframe["rerank_score_raw"] = dataframe[score_column].astype(float)
    dataframe["rerank_score"] = normalize(dataframe["rerank_score_raw"].tolist())

    result = {}

    for _, row in dataframe.iterrows():
        case_id = str(row.get("case_id", ""))
        chunk_id = str(row.get("chunk_id", ""))

        if chunk_id:
            result[chunk_id] = float(row["rerank_score"])

        if case_id:
            result[case_id] = max(
                result.get(case_id, 0.0),
                float(row["rerank_score"]),
            )

    return result


def load_authority_scores(path: Path) -> dict[str, float]:
    dataframe = load_csv(path)

    if dataframe.empty:
        return {}

    if "authority_score" not in dataframe.columns:
        return {}

    return {
        str(row["case_id"]): float(row["authority_score"])
        for _, row in dataframe.iterrows()
    }


def build_statute_match_scores(query: str, path: Path) -> dict[str, float]:
    dataframe = load_csv(path)

    if dataframe.empty:
        return {}

    query_tokens = tokenize(query)
    scores = {}

    for _, row in dataframe.iterrows():
        case_id = str(row.get("case_id", ""))
        mention_text = str(row.get("mention_text", ""))
        reference_number = str(row.get("reference_number", ""))
        mention_type = str(row.get("mention_type", ""))

        searchable = f"{mention_type} {mention_text} {reference_number}"
        overlap = len(query_tokens & tokenize(searchable))

        if overlap <= 0:
            continue

        scores[case_id] = scores.get(case_id, 0.0) + overlap

    if not scores:
        return {}

    case_ids = list(scores.keys())
    normalized_values = normalize(list(scores.values()))

    return {
        case_id: score
        for case_id, score in zip(case_ids, normalized_values)
    }


def recency_validity_score(year_value) -> float:
    try:
        year = int(float(year_value))
    except Exception:
        return 0.50

    age = max(CURRENT_YEAR - year, 0)

    if age <= 5:
        return 1.00

    if age <= 15:
        return 0.80

    if age <= 30:
        return 0.60

    if age <= 50:
        return 0.40

    return 0.25


def explain_result(
    hybrid_score: float,
    rerank_score: float,
    authority_score: float,
    statute_match_score: float,
    recency_score: float,
) -> str:
    reasons = []

    if hybrid_score > 0:
        reasons.append("matched query through hybrid retrieval")

    if rerank_score > 0:
        reasons.append("reranker considered it relevant")

    if authority_score >= 0.5:
        reasons.append("strong legal authority score")

    if statute_match_score > 0:
        reasons.append("matched statute/article reference")

    if recency_score >= 0.8:
        reasons.append("recent or currently useful precedent")

    if not reasons:
        reasons.append("included as fallback candidate")

    return "; ".join(reasons)


def citation_aware_rank(query: str, top_k: int = 10) -> list[dict]:
    hybrid_df = load_hybrid_results(HYBRID_SEARCH_RESULTS_PATH)
    rerank_scores = load_rerank_scores(RERANKED_RESULTS_PATH)
    authority_scores = load_authority_scores(AUTHORITY_SCORES_PATH)
    statute_scores = build_statute_match_scores(query, STATUTE_MENTIONS_PATH)

    rows = []

    for _, row in hybrid_df.iterrows():
        case_id = str(row.get("case_id", ""))
        chunk_id = str(row.get("chunk_id", ""))

        hybrid_score = float(row.get("hybrid_score", 0.0))

        rerank_score = max(
            rerank_scores.get(chunk_id, 0.0),
            rerank_scores.get(case_id, 0.0),
        )

        authority_score = authority_scores.get(case_id, 0.0)
        statute_match_score = statute_scores.get(case_id, 0.0)
        recency_score = recency_validity_score(row.get("year", ""))

        final_score = (
            0.40 * hybrid_score
            + 0.20 * rerank_score
            + 0.20 * authority_score
            + 0.10 * statute_match_score
            + 0.10 * recency_score
        )

        rows.append(
            {
                "case_id": case_id,
                "chunk_id": chunk_id,
                "title": row.get("title", ""),
                "court": row.get("court", ""),
                "year": row.get("year", ""),
                "hybrid_score": round(hybrid_score, 4),
                "rerank_score": round(rerank_score, 4),
                "authority_score": round(authority_score, 4),
                "statute_match_score": round(statute_match_score, 4),
                "recency_validity_score": round(recency_score, 4),
                "final_score": round(final_score, 4),
                "why_ranked": explain_result(
                    hybrid_score,
                    rerank_score,
                    authority_score,
                    statute_match_score,
                    recency_score,
                ),
                "snippet": row.get("snippet", ""),
            }
        )

    rows.sort(key=lambda item: item["final_score"], reverse=True)

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows[:top_k]


def write_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "case_id",
        "chunk_id",
        "title",
        "court",
        "year",
        "hybrid_score",
        "rerank_score",
        "authority_score",
        "statute_match_score",
        "recency_validity_score",
        "final_score",
        "why_ranked",
        "snippet",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results(query: str, rows: list[dict]) -> None:
    print("\nCitation-Aware Ranking Report")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"Final ranked results: {len(rows)}")
    print(f"Output saved to: {CITATION_AWARE_RANKING_PATH}")
    print("=" * 70)

    for row in rows:
        print(
            f"\nRank {row['rank']} | {row['case_id']} | "
            f"final={row['final_score']} | {row['title']}"
        )
        print(
            f"hybrid={row['hybrid_score']} | "
            f"rerank={row['rerank_score']} | "
            f"authority={row['authority_score']} | "
            f"statute={row['statute_match_score']} | "
            f"recency={row['recency_validity_score']}"
        )
        print(f"Why ranked: {row['why_ranked']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Citation-aware legal ranking."
    )

    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = citation_aware_rank(
        query=args.query,
        top_k=args.top_k,
    )

    write_results(rows, CITATION_AWARE_RANKING_PATH)
    print_results(args.query, rows)


if __name__ == "__main__":
    main()
