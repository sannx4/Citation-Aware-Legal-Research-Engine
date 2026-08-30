import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import ADVANCED_CLEAN_CASES_PATH, SECTION_SEGMENTS_PATH


SECTION_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(facts?|issues?|arguments?|submissions?|reasoning|analysis|discussion|held|holding|decision|order|conclusion|judgment)\s*:?\s*$"
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_section_name(name: str) -> str:
    value = name.lower().strip()

    if value.startswith("fact"):
        return "FACTS"
    if value.startswith("issue"):
        return "ISSUES"
    if value in {"arguments", "argument", "submissions", "submission"}:
        return "ARGUMENTS"
    if value in {"reasoning", "analysis", "discussion"}:
        return "REASONING"
    if value in {"held", "holding"}:
        return "HOLDING"
    if value in {"decision", "order", "conclusion", "judgment"}:
        return "DECISION"

    return "UNKNOWN"


def segment_sections(text: str) -> list[dict]:
    matches = list(SECTION_HEADING_PATTERN.finditer(text))

    if not matches:
        return [
            {
                "section_type": "UNKNOWN",
                "section_heading": "",
                "start_char": 0,
                "end_char": len(text),
                "section_text": text.strip(),
            }
        ]

    sections = []

    for index, match in enumerate(matches):
        heading = match.group(1)
        section_type = normalize_section_name(heading)

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        if not section_text:
            continue

        sections.append(
            {
                "section_type": section_type,
                "section_heading": heading,
                "start_char": start,
                "end_char": end,
                "section_text": section_text,
            }
        )

    return sections


def build_section_rows(records: list[dict]) -> list[dict]:
    rows = []

    for record in records:
        case_id = record.get("case_id", "")
        text = record.get("pii_masked_text") or record.get("advanced_cleaned_text", "")

        sections = segment_sections(text)

        for index, section in enumerate(sections, start=1):
            rows.append(
                {
                    "section_id": f"{case_id}_SECTION_{index:04d}",
                    "case_id": case_id,
                    "title": record.get("title", ""),
                    "court": record.get("court", ""),
                    "year": record.get("year", ""),
                    **section,
                }
            )

    return rows


def main() -> None:
    records = read_jsonl(ADVANCED_CLEAN_CASES_PATH)
    rows = build_section_rows(records)

    write_jsonl(rows, SECTION_SEGMENTS_PATH)

    print("\nSection Segmentation Report")
    print("=" * 70)
    print(f"Input cases: {len(records)}")
    print(f"Sections created: {len(rows)}")
    print(f"Output saved to: {SECTION_SEGMENTS_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(f"{row['section_id']} | {row['section_type']} | {row['case_id']}")


if __name__ == "__main__":
    main()
