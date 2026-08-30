import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import QUERY_ANALYSIS_REPORT_PATH
from src.query.query_intent_classifier import classify_intent
from src.query.query_entity_extractor import extract_concepts, infer_statutes
from src.query.query_rewriter import rewrite_query


def analyze_query(query: str) -> dict:
    intent = classify_intent(query)
    concepts = extract_concepts(query)
    statutes = infer_statutes(query, concepts)
    rewritten_query = rewrite_query(query, concepts, statutes)

    return {
        "original_query": query,
        "intent": intent,
        "concepts": concepts,
        "statutes": statutes,
        "rewritten_query": rewritten_query,
    }


def save_report(result: dict) -> None:
    QUERY_ANALYSIS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = QUERY_ANALYSIS_REPORT_PATH.exists()

    fieldnames = [
        "original_query",
        "intent",
        "concepts",
        "statutes",
        "rewritten_query",
    ]

    with QUERY_ANALYSIS_REPORT_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "original_query": result["original_query"],
                "intent": result["intent"],
                "concepts": "; ".join(result["concepts"]),
                "statutes": "; ".join(result["statutes"]),
                "rewritten_query": result["rewritten_query"],
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and rewrite a legal search query."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Example: Can police check my phone without permission?",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_query(args.query)
    save_report(result)

    print("\nGap 13 Query Understanding Report")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)
    print(f"Report saved to: {QUERY_ANALYSIS_REPORT_PATH}")


if __name__ == "__main__":
    main()
