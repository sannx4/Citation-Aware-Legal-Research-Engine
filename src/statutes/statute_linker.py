import csv
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    STATUTE_MENTIONS_PATH,
    STATUTE_SECTIONS_PATH,
    STATUTE_LINKS_PATH,
)


def normalize_number(value: str) -> str:
    return str(value).strip().upper().replace(" ", "")


def load_statute_mentions() -> pd.DataFrame:
    if not STATUTE_MENTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Statute mentions not found: {STATUTE_MENTIONS_PATH}. "
            "Run extract_statutes.py first."
        )

    return pd.read_csv(STATUTE_MENTIONS_PATH)


def load_statute_sections() -> pd.DataFrame:
    if not STATUTE_SECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Statute sections not found: {STATUTE_SECTIONS_PATH}. "
            "Run statute_parser.py first."
        )

    return pd.read_csv(STATUTE_SECTIONS_PATH)


def infer_act_hint(mention_text: str, chunk_text: str = "") -> str:
    combined = f"{mention_text} {chunk_text}".lower()

    if "ipc" in combined or "indian penal code" in combined:
        return "IPC"

    if "contract act" in combined or "indian contract act" in combined:
        return "INDIAN_CONTRACT_ACT"

    if "article" in combined or "constitution" in combined:
        return "CONSTITUTION_OF_INDIA"

    return ""


def find_best_section_match(
    mention: dict,
    sections: pd.DataFrame,
) -> dict | None:
    mention_type = str(mention.get("mention_type", "")).upper()
    reference_number = normalize_number(mention.get("reference_number", ""))
    mention_text = str(mention.get("mention_text", ""))

    act_hint = infer_act_hint(mention_text)

    candidates = sections[
        sections["section_number"].astype(str).map(normalize_number)
        == reference_number
    ]

    if mention_type == "ARTICLE":
        candidates = candidates[
            candidates["section_type"].astype(str).str.upper() == "ARTICLE"
        ]

    if mention_type == "SECTION":
        candidates = candidates[
            candidates["section_type"].astype(str).str.upper() == "SECTION"
        ]

    if candidates.empty:
        return None

    if act_hint:
        hinted = candidates[
            candidates["statute_id"].astype(str).str.upper() == act_hint
        ]

        if not hinted.empty:
            row = hinted.iloc[0].to_dict()
            row["match_confidence"] = 0.95
            row["match_reason"] = "number_type_and_act_hint_match"
            return row

    row = candidates.iloc[0].to_dict()
    row["match_confidence"] = 0.80
    row["match_reason"] = "number_and_type_match"
    return row


def build_statute_links(
    mentions: pd.DataFrame,
    sections: pd.DataFrame,
) -> list[dict]:
    rows = []

    for _, mention_row in mentions.iterrows():
        mention = mention_row.to_dict()
        match = find_best_section_match(mention, sections)

        if match is None:
            rows.append(
                {
                    "case_id": mention.get("case_id", ""),
                    "chunk_id": mention.get("chunk_id", ""),
                    "mention_type": mention.get("mention_type", ""),
                    "mention_text": mention.get("mention_text", ""),
                    "reference_number": mention.get("reference_number", ""),
                    "section_id": "",
                    "statute_id": "",
                    "act_name": "",
                    "section_heading": "",
                    "section_full_text": "",
                    "match_confidence": 0.0,
                    "match_reason": "no_statute_section_found",
                }
            )
            continue

        rows.append(
            {
                "case_id": mention.get("case_id", ""),
                "chunk_id": mention.get("chunk_id", ""),
                "mention_type": mention.get("mention_type", ""),
                "mention_text": mention.get("mention_text", ""),
                "reference_number": mention.get("reference_number", ""),
                "section_id": match.get("section_id", ""),
                "statute_id": match.get("statute_id", ""),
                "act_name": match.get("act_name", ""),
                "section_heading": match.get("heading", ""),
                "section_full_text": match.get("full_text", ""),
                "match_confidence": match.get("match_confidence", 0.0),
                "match_reason": match.get("match_reason", ""),
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "chunk_id",
        "mention_type",
        "mention_text",
        "reference_number",
        "section_id",
        "statute_id",
        "act_name",
        "section_heading",
        "section_full_text",
        "match_confidence",
        "match_reason",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    mentions = load_statute_mentions()
    sections = load_statute_sections()

    rows = build_statute_links(mentions, sections)
    write_csv(rows, STATUTE_LINKS_PATH)

    linked = sum(1 for row in rows if row["section_id"])
    unlinked = len(rows) - linked

    print("\nGap 15 Statute Database Linking Report")
    print("=" * 70)
    print(f"Statute mentions checked: {len(rows)}")
    print(f"Linked to full statute text: {linked}")
    print(f"Unlinked mentions: {unlinked}")
    print(f"Output saved to: {STATUTE_LINKS_PATH}")
    print("=" * 70)

    for row in rows[:10]:
        print(
            f"{row['case_id']} | "
            f"{row['mention_text']} -> {row['section_id']} | "
            f"confidence={row['match_confidence']}"
        )


if __name__ == "__main__":
    main()
