import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    KG_REASONING_RESULT_PATH,
    CITATION_AWARE_RANKING_PATH,
    EXPLAINABLE_RANKING_CSV_PATH,
    EVIDENCE_PACK_PATH,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_evidence_pack(query: str, top_k: int = 3):
    kg = load_json(KG_REASONING_RESULT_PATH)

    ranking = pd.read_csv(CITATION_AWARE_RANKING_PATH)
    explanations = pd.read_csv(EXPLAINABLE_RANKING_CSV_PATH)

    evidence = []

    for _, row in ranking.head(top_k).iterrows():

        case_id = row["case_id"]

        explanation_row = explanations[
            explanations["case_id"] == case_id
        ]

        why_ranked = ""

        if len(explanation_row):
            why_ranked = explanation_row.iloc[0]["why_ranked"]

        evidence.append(
            {
                "case_id": case_id,
                "case": row["title"],
                "court": row["court"],
                "year": row["year"],
                "final_score": float(row["final_score"]),
                "why_ranked": why_ranked,
                "snippet": row["snippet"],
            }
        )

    return {
        "query": query,
        "intent": kg["intent"],
        "concepts": kg["concepts"],
        "statutes": kg["statutes"],
        "likely_cases": kg["likely_cases"],
        "rewritten_query": kg["rewritten_query"],
        "evidence": evidence,
    }


def save_pack(pack):
    EVIDENCE_PACK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with EVIDENCE_PACK_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            pack,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_report(pack):
    print("\nPhase 5.1 Evidence Pack Report")
    print("=" * 70)

    print(f"Query: {pack['query']}")
    print(f"Concepts: {len(pack['concepts'])}")
    print(f"Statutes: {len(pack['statutes'])}")
    print(f"Evidence cases: {len(pack['evidence'])}")

    print("=" * 70)

    for item in pack["evidence"]:

        print(
            f"{item['case_id']} | "
            f"{item['case']} | "
            f"score={item['final_score']}"
        )

    print("=" * 70)
    print(f"Saved to: {EVIDENCE_PACK_PATH}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    pack = build_evidence_pack(
        query=args.query,
        top_k=args.top_k,
    )

    save_pack(pack)

    print_report(pack)


if __name__ == "__main__":
    main()