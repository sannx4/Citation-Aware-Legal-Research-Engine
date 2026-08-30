import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import RAW_EXTRACTED_CASES_PATH, CLEAN_CASES_PATH, REPORTS_DIR


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_numbers(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if re.fullmatch(r"Page\s+\d+", stripped, flags=re.IGNORECASE):
            continue

        if re.fullmatch(r"\d+", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_common_headers_footers(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []

    unwanted_patterns = [
        r"^SUPREME COURT OF INDIA$",
        r"^HIGH COURT OF .*",
        r"^Downloaded from.*",
        r"^www\..*",
    ]

    for line in lines:
        stripped = line.strip()

        should_remove = any(
            re.fullmatch(pattern, stripped, flags=re.IGNORECASE)
            for pattern in unwanted_patterns
        )

        if not should_remove:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def fix_broken_lines(text: str) -> str:
    lines = text.split("\n")
    fixed_lines = []
    buffer = ""

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if buffer:
                fixed_lines.append(buffer.strip())
                buffer = ""
            fixed_lines.append("")
            continue

        if buffer:
            buffer += " " + stripped
        else:
            buffer = stripped

        if stripped.endswith((".", ":", ";", "?", "!")):
            fixed_lines.append(buffer.strip())
            buffer = ""

    if buffer:
        fixed_lines.append(buffer.strip())

    return "\n".join(fixed_lines)


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = remove_page_numbers(text)
    text = remove_common_headers_footers(text)
    text = fix_broken_lines(text)
    text = normalize_whitespace(text)
    return text


def read_jsonl(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_cleaning_examples(records: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    example_path = REPORTS_DIR / "cleaning_examples.md"

    with example_path.open("w", encoding="utf-8") as file:
        file.write("# Day 4 Cleaning Examples\n\n")

        for record in records[:3]:
            raw_text = record.get("raw_text", "")
            cleaned_text = record.get("cleaned_text", "")

            file.write(f"## {record.get('case_id')} - {record.get('title')}\n\n")
            file.write("### Before\n\n")
            file.write("```text\n")
            file.write(raw_text[:700])
            file.write("\n```\n\n")

            file.write("### After\n\n")
            file.write("```text\n")
            file.write(cleaned_text[:700])
            file.write("\n```\n\n")


def clean_cases() -> None:
    records = read_jsonl(RAW_EXTRACTED_CASES_PATH)

    cleaned_records = []

    print("\nDay 4 Text Cleaning Report")
    print("=" * 70)

    for record in records:
        raw_text = record.get("raw_text", "")
        cleaned_text = clean_text(raw_text)

        new_record = dict(record)
        new_record["cleaned_text"] = cleaned_text
        new_record["raw_text_length"] = len(raw_text)
        new_record["cleaned_text_length"] = len(cleaned_text)

        cleaned_records.append(new_record)

        print(
            f"{record.get('case_id')} | "
            f"raw={len(raw_text)} chars | "
            f"cleaned={len(cleaned_text)} chars"
        )

    write_jsonl(cleaned_records, CLEAN_CASES_PATH)
    save_cleaning_examples(cleaned_records)

    print("=" * 70)
    print(f"Cleaned cases saved to: {CLEAN_CASES_PATH}")
    print(f"Cleaning examples saved to: {REPORTS_DIR / 'cleaning_examples.md'}")


if __name__ == "__main__":
    clean_cases()