import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CORPUS_REGISTRY_PATH,
    CORPUS_STATS_PATH,
)


def clean_value(value) -> str:
    value = str(value).strip()

    if not value:
        return ""

    if value.lower() == "nan":
        return ""

    return value


def count_values(series) -> dict:
    values = []

    for value in series:
        cleaned = clean_value(value)

        if cleaned:
            values.append(cleaned)

    return dict(
        sorted(
            Counter(values).items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def main() -> None:
    if not CORPUS_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Corpus registry not found: {CORPUS_REGISTRY_PATH}\n"
            "Run: python src/ingestion/corpus_registry.py"
        )

    dataframe = pd.read_csv(
        CORPUS_REGISTRY_PATH
    )

    if "status" not in dataframe.columns:
        raise ValueError(
            "corpus_registry.csv does not contain a status column."
        )

    ready = dataframe[
        dataframe["status"]
        .astype(str)
        .str.lower()
        .eq("ready")
    ].copy()

    total_pages = (
        pd.to_numeric(
            ready.get("page_count", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    total_bytes = (
        pd.to_numeric(
            ready.get("file_size", pd.Series(dtype=float)),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    average_pages = (
        float(total_pages) / len(ready)
        if len(ready) > 0
        else 0.0
    )

    average_file_size_mb = (
        (float(total_bytes) / len(ready))
        / (1024 * 1024)
        if len(ready) > 0
        else 0.0
    )

    stats = {
        "total_registry_rows": int(len(dataframe)),
        "valid_unique_judgments": int(len(ready)),
        "invalid_documents": int(
            (
                dataframe["status"]
                .astype(str)
                .str.lower()
                == "invalid"
            ).sum()
        ),
        "duplicate_documents": int(
            (
                dataframe["status"]
                .astype(str)
                .str.lower()
                == "duplicate"
            ).sum()
        ),
        "total_pages": int(total_pages),
        "average_pages_per_judgment": round(
            average_pages,
            2,
        ),
        "total_size_mb": round(
            float(total_bytes) / (1024 * 1024),
            2,
        ),
        "average_file_size_mb": round(
            average_file_size_mb,
            2,
        ),
        "sources": (
            count_values(ready["source"])
            if "source" in ready.columns
            else {}
        ),
        "courts": (
            count_values(ready["court"])
            if "court" in ready.columns
            else {}
        ),
        "years": (
            count_values(ready["year"])
            if "year" in ready.columns
            else {}
        ),
    }

    CORPUS_STATS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CORPUS_STATS_PATH.write_text(
        json.dumps(
            stats,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nCorpus Statistics Report")
    print("=" * 70)

    print(
        f"Registry documents: "
        f"{stats['total_registry_rows']}"
    )

    print(
        f"Valid unique judgments: "
        f"{stats['valid_unique_judgments']}"
    )

    print(
        f"Invalid documents: "
        f"{stats['invalid_documents']}"
    )

    print(
        f"Duplicate documents: "
        f"{stats['duplicate_documents']}"
    )

    print(
        f"Total pages: "
        f"{stats['total_pages']}"
    )

    print(
        f"Average pages/judgment: "
        f"{stats['average_pages_per_judgment']}"
    )

    print(
        f"Corpus size: "
        f"{stats['total_size_mb']} MB"
    )

    print(
        f"Average file size: "
        f"{stats['average_file_size_mb']} MB"
    )

    print(
        f"Sources represented: "
        f"{len(stats['sources'])}"
    )

    print(
        f"Courts represented: "
        f"{len(stats['courts'])}"
    )

    print(
        f"Years represented: "
        f"{len(stats['years'])}"
    )

    print(
        f"Stats saved to: "
        f"{CORPUS_STATS_PATH}"
    )

    print("=" * 70)

    if stats["courts"]:
        print("\nTop courts")

        for court, count in list(
            stats["courts"].items()
        )[:10]:
            print(
                f"{court} | {count}"
            )


if __name__ == "__main__":
    main()