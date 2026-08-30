import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CITATION_AWARE_RANKING_PATH,
    AUTHORITY_SCORES_PATH,
    STATUTE_MENTIONS_PATH,
    EXPLAINABLE_RANKING_PATH,
    EXPLAINABLE_RANKING_CSV_PATH,
)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_authority_details() -> dict[str, dict]:
    dataframe = load_csv(AUTHORITY_SCORES_PATH)

    if dataframe.empty:
        return {}

    return {
        str(row.get("case_id", "")): row.to_dict()
        for _, row in dataframe.iterrows()
    }


def load_statute_mentions() -> dict[str, list[str]]:
    dataframe = load_csv(STATUTE_MENTIONS_PATH)

    if dataframe.empty:
        return {}

    statute_map = {}

    for _, row in dataframe.iterrows():
        case_id = str(row.get("case_id", ""))
        mention = str(row.get("mention_text", ""))

        if case_id and mention:
            statute_map.setdefault(case_id, []).append(mention)

    return statute_map


def matched_query_terms(query: str, snippet: str) -> list[str]:
    query_tokens = tokenize(query)
    snippet_tokens = tokenize(snippet)

    matched = sorted(query_tokens & snippet_tokens)

    return matched


def build_explanation(
    query: str,
    row: dict,
    authority_details: dict[str, dict],
    statute_map: dict[str, list[str]],
) -> list[str]:

    reasons = []

    case_id = str(row.get("case_id", ""))
    court = str(row.get("court", ""))
    snippet = str(row.get("snippet", ""))

    hybrid_score = float(row.get("hybrid_score", 0.0))
    rerank_score = float(row.get("rerank_score", 0.0))
    authority_score = float(row.get("authority_score", 0.0))
    statute_score = float(row.get("statute_match_score", 0.0))
    recency_score = float(row.get("recency_validity_score", 0.0))

    terms = matched_query_terms(query, snippet)

    if terms:
        reasons.append(
            "Matched query terms: " + ", ".join(terms)
        )

    if "article" in tokenize(query) and statute_score > 0:
        statutes = sorted(set(statute_map.get(case_id, [])))

        if statutes:
            reasons.append(
                "Matched legal provision: " + ", ".join(statutes[:5])
            )
        else:
            reasons.append("Matched legal statute/article reference")

    if hybrid_score >= 0.75:
        reasons.append("Highly relevant in hybrid BM25 + semantic search")
    elif hybrid_score > 0:
        reasons.append("Partially relevant in hybrid retrieval")

    if rerank_score > 0:
        reasons.append("Cross-encoder reranker considered it relevant")

    if authority_score >= 0.50:
        reasons.append("Strong legal authority score")

    if "supreme court" in court.lower():
        reasons.append("Supreme Court authority")

    details = authority_details.get(case_id, {})

    pagerank_score = float(details.get("pagerank_score", 0.0) or 0.0)
    bench_strength = float(details.get("bench_strength_weight", 0.0) or 0.0)
    penalty = float(details.get("negative_treatment_penalty", 0.0) or 0.0)

    if pagerank_score >= 0.50:
        reasons.append("High citation PageRank authority")

    if bench_strength >= 0.90:
        reasons.append("Strong bench strength / Constitution Bench")

    if penalty == 0:
        reasons.append("Not negatively treated")
    else:
        reasons.append("Has negative treatment penalty")

    if recency_score >= 0.80:
        reasons.append("Recent or currently useful precedent")
    elif recency_score <= 0.30:
        reasons.append("Older precedent, ranked lower for recency")

    if not reasons:
        reasons.append("Included as fallback candidate")

    return reasons


def explain_rankings(query: str, top_k: int = 10) -> list[dict]:
    ranked = load_csv(CITATION_AWARE_RANKING_PATH)

    if ranked.empty:
        raise FileNotFoundError(
            f"Missing citation-aware ranking file: {CITATION_AWARE_RANKING_PATH}. "
            "Run: python src/ranking/citation_aware_ranker.py \"Article 21 privacy\" --top-k 5"
        )

    authority_details = load_authority_details()
    statute_map = load_statute_mentions()

    output = []

    for _, item in ranked.head(top_k).iterrows():
        row = item.to_dict()

        reasons = build_explanation(
            query=query,
            row=row,
            authority_details=authority_details,
            statute_map=statute_map,
        )

        output.append(
            {
                "rank": int(row.get("rank", 0)),
                "case_id": row.get("case_id", ""),
                "case": row.get("title", ""),
                "court": row.get("court", ""),
                "year": row.get("year", ""),
                "final_score": float(row.get("final_score", 0.0)),
                "why_ranked": reasons,
                "snippet": row.get("snippet", ""),
            }
        )

    return output


def write_json(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "case_id",
        "case",
        "court",
        "year",
        "final_score",
        "why_ranked",
        "snippet",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    **row,
                    "why_ranked": " | ".join(row["why_ranked"]),
                }
            )


def print_report(query: str, rows: list[dict]) -> None:
    print("\nGap 21 Explainable Ranking Report")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"Explained results: {len(rows)}")
    print(f"JSON saved to: {EXPLAINABLE_RANKING_PATH}")
    print(f"CSV saved to: {EXPLAINABLE_RANKING_CSV_PATH}")
    print("=" * 70)

    for row in rows:
        print(
            f"\nRank {row['rank']} | {row['case_id']} | "
            f"score={row['final_score']} | {row['case']}"
        )

        print("Why ranked:")

        for reason in row["why_ranked"]:
            print(f"  - {reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain citation-aware legal ranking results."
    )

    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = explain_rankings(
        query=args.query,
        top_k=args.top_k,
    )

    write_json(rows, EXPLAINABLE_RANKING_PATH)
    write_csv(rows, EXPLAINABLE_RANKING_CSV_PATH)
    print_report(args.query, rows)


if __name__ == "__main__":
    main()
