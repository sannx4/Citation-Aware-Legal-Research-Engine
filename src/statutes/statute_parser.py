import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import STATUTES_JSONL_PATH, STATUTE_SECTIONS_PATH


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}. Run load_indiacode.py first."
        )

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def build_section_rows(statutes: list[dict]) -> list[dict]:
    rows = []

    for statute in statutes:
        statute_id = statute["statute_id"]
        act_name = statute["act_name"]
        short_name = statute["short_name"]
        source = statute.get("source", "")

        for section in statute.get("sections", []):
            rows.append(
                {
                    "statute_id": statute_id,
                    "act_name": act_name,
                    "short_name": short_name,
                    "source": source,
                    "section_id": section["section_id"],
                    "section_type": section["section_type"],
                    "section_number": section["section_number"],
                    "heading": section.get("heading", ""),
                    "full_text": section.get("text", ""),
                }
            )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "statute_id",
        "act_name",
        "short_name",
        "source",
        "section_id",
        "section_type",
        "section_number",
        "heading",
        "full_text",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    statutes = read_jsonl(STATUTES_JSONL_PATH)
    rows = build_section_rows(statutes)
    write_csv(rows, STATUTE_SECTIONS_PATH)

    print("\nStatute Parser Report")
    print("=" * 70)
    print(f"Statutes parsed: {len(statutes)}")
    print(f"Sections/articles created: {len(rows)}")
    print(f"Output saved to: {STATUTE_SECTIONS_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['section_id']} | "
            f"{row['section_type']} {row['section_number']} | "
            f"{row['heading']}"
        )


if __name__ == "__main__":
    main()
