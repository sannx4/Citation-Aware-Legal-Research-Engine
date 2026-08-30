import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    KG_REASONING_RESULT_PATH,
    EXPLAINABLE_RANKING_CSV_PATH,
    RAG_ANSWER_PATH,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_evidence() -> pd.DataFrame:
    if not EXPLAINABLE_RANKING_CSV_PATH.exists():
        raise FileNotFoundError(
            "Run ranking explainer first: "
            "python src/explainability/ranking_explainer.py \"Article 21 privacy\" --top-k 5"
        )

    return pd.read_csv(EXPLAINABLE_RANKING_CSV_PATH)


def generate_answer(query: str) -> dict:
    kg = load_json(KG_REASONING_RESULT_PATH)
    evidence = load_evidence()

    top_cases = []

    for _, row in evidence.head(3).iterrows():
        top_cases.append(
            {
                "case_id": row.get("case_id", ""),
                "case": row.get("case", ""),
                "court": row.get("court", ""),
                "year": row.get("year", ""),
                "why_ranked": row.get("why_ranked", ""),
                "snippet": row.get("snippet", ""),
            }
        )

    answer = (
        f"The query appears to involve {', '.join(kg.get('concepts', []))}. "
        f"The most relevant legal provision is {', '.join(kg.get('statutes', []))}. "
        f"The strongest available precedent in the retrieved evidence is "
        f"{top_cases[0]['case']}."
    )

    return {
        "query": query,
        "intent": kg.get("intent", ""),
        "concepts": kg.get("concepts", []),
        "statutes": kg.get("statutes", []),
        "likely_cases": kg.get("likely_cases", []),
        "answer": answer,
        "evidence": top_cases,
        "disclaimer": "Research assistance only. Not legal advice.",
    }


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()

    result = generate_answer(args.query)
    write_json(result, RAG_ANSWER_PATH)

    print("\nRAG Answer Generation Report")
    print("=" * 70)
    print(result["answer"])
    print("=" * 70)
    print(f"Saved to: {RAG_ANSWER_PATH}")


if __name__ == "__main__":
    main()