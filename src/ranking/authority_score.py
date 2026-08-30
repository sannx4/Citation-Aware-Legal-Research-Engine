import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CLEAN_CASES_PATH,
    EDGES_PATH,
    GRAPH_PATH,
    AUTHORITY_SCORES_PATH,
    CITATION_TREATMENTS_PATH,
)

from src.ranking.court_hierarchy import court_weight
from src.ranking.bench_strength import extract_bench_size, bench_strength_weight
from src.ranking.precedent_validity import (
    negative_treatment_penalty,
    positive_treatment_bonus,
)


CURRENT_YEAR = datetime.now().year


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def normalize_year(year_value) -> int:
    try:
        return int(float(year_value))
    except Exception:
        return CURRENT_YEAR


def recency_weight(year: int) -> float:
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


def load_cases() -> list[dict]:
    cases = read_jsonl(CLEAN_CASES_PATH)

    normalized = []

    for case in cases:
        text = (
            case.get("cleaned_text")
            or case.get("raw_text")
            or case.get("text")
            or ""
        )

        normalized.append(
            {
                "case_id": str(case.get("case_id", "")),
                "title": str(case.get("title", "")),
                "court": str(case.get("court", "")),
                "year": normalize_year(case.get("year", CURRENT_YEAR)),
                "text": text,
            }
        )

    return [case for case in normalized if case["case_id"]]


def compute_pagerank_scores(case_ids: list[str]) -> dict[str, float]:
    if GRAPH_PATH.exists():
        try:
            graph = nx.read_graphml(GRAPH_PATH)
            pagerank = nx.pagerank(graph)
        except Exception:
            pagerank = {}
    elif EDGES_PATH.exists():
        edges = pd.read_csv(EDGES_PATH)
        graph = nx.DiGraph()

        for case_id in case_ids:
            graph.add_node(case_id)

        for _, row in edges.iterrows():
            source = str(row.get("source", ""))
            target = str(row.get("target", ""))

            if source and target:
                graph.add_edge(source, target)

        pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
    else:
        pagerank = {}

    values = list(pagerank.values())

    if not values:
        return {case_id: 0.0 for case_id in case_ids}

    min_value = min(values)
    max_value = max(values)

    normalized = {}

    for case_id in case_ids:
        raw = float(pagerank.get(case_id, 0.0))

        if max_value == min_value:
            normalized[case_id] = 1.0 if raw > 0 else 0.0
        else:
            normalized[case_id] = (raw - min_value) / (max_value - min_value)

    return normalized


def load_treatments() -> dict[str, list[str]]:
    if not CITATION_TREATMENTS_PATH.exists():
        return {}

    treatments_df = pd.read_csv(CITATION_TREATMENTS_PATH)
    treatment_map = {}

    for _, row in treatments_df.iterrows():
        target_case_id = str(
            row.get("target_case_id")
            or row.get("cited_case_id")
            or row.get("target")
            or ""
        )

        treatment = str(row.get("treatment", "")).upper().strip()

        if not target_case_id or not treatment:
            continue

        treatment_map.setdefault(target_case_id, []).append(treatment)

    return treatment_map


def calculate_authority_scores() -> list[dict]:
    cases = load_cases()

    if not cases:
        raise FileNotFoundError(
            f"No cases found in {CLEAN_CASES_PATH}. Run preprocessing first."
        )

    case_ids = [case["case_id"] for case in cases]
    pagerank_scores = compute_pagerank_scores(case_ids)
    treatment_map = load_treatments()

    rows = []

    for case in cases:
        case_id = case["case_id"]
        treatments = treatment_map.get(case_id, [])

        court_score = court_weight(case["court"])
        bench_size = extract_bench_size(case["text"])
        bench_score = bench_strength_weight(bench_size)
        pr_score = pagerank_scores.get(case_id, 0.0)
        recent_score = recency_weight(case["year"])
        negative_penalty = negative_treatment_penalty(treatments)
        positive_bonus = positive_treatment_bonus(treatments)

        authority_score = (
            0.30 * court_score
            + 0.20 * bench_score
            + 0.25 * pr_score
            + 0.15 * recent_score
            + positive_bonus
            - negative_penalty
        )

        authority_score = max(0.0, min(authority_score, 1.0))

        rows.append(
            {
                "case_id": case_id,
                "title": case["title"],
                "court": case["court"],
                "year": case["year"],
                "court_weight": round(court_score, 4),
                "bench_size": bench_size,
                "bench_strength_weight": round(bench_score, 4),
                "pagerank_score": round(pr_score, 4),
                "recency_score": round(recent_score, 4),
                "positive_treatment_bonus": round(positive_bonus, 4),
                "negative_treatment_penalty": round(negative_penalty, 4),
                "authority_score": round(authority_score, 4),
                "treatments": ";".join(treatments),
            }
        )

    rows.sort(key=lambda row: row["authority_score"], reverse=True)

    for rank, row in enumerate(rows, start=1):
        row["authority_rank"] = rank

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "authority_rank",
        "case_id",
        "title",
        "court",
        "year",
        "court_weight",
        "bench_size",
        "bench_strength_weight",
        "pagerank_score",
        "recency_score",
        "positive_treatment_bonus",
        "negative_treatment_penalty",
        "authority_score",
        "treatments",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = calculate_authority_scores()
    write_csv(rows, AUTHORITY_SCORES_PATH)

    print("\nLegal Authority Ranking Report")
    print("=" * 70)
    print(f"Cases scored: {len(rows)}")
    print(f"Output saved to: {AUTHORITY_SCORES_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"Rank {row['authority_rank']} | "
            f"{row['case_id']} | "
            f"authority={row['authority_score']} | "
            f"court={row['court_weight']} | "
            f"bench={row['bench_strength_weight']} | "
            f"pagerank={row['pagerank_score']} | "
            f"recency={row['recency_score']} | "
            f"penalty={row['negative_treatment_penalty']} | "
            f"{row['title']}"
        )


if __name__ == "__main__":
    main()
