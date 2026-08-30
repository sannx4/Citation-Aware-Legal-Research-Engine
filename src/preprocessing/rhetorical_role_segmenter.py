import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import SECTION_SEGMENTS_PATH, RHETORICAL_ROLE_SEGMENTS_PATH


ROLE_KEYWORDS = {
    "FACTS": [
        "facts",
        "background",
        "petitioner",
        "respondent",
        "appellant",
        "accused",
        "complainant",
    ],
    "ISSUES": [
        "issue",
        "question",
        "whether",
        "point for consideration",
    ],
    "ARGUMENTS": [
        "argued",
        "submitted",
        "contended",
        "learned counsel",
        "submission",
    ],
    "REASONING": [
        "court observed",
        "we find",
        "therefore",
        "analysis",
        "reason",
        "considered",
    ],
    "HOLDING": [
        "held",
        "we hold",
        "it is held",
        "principle laid down",
    ],
    "DECISION": [
        "allowed",
        "dismissed",
        "disposed",
        "ordered",
        "appeal is",
        "petition is",
    ],
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_paragraphs(text: str) -> list[dict]:
    paragraphs = []
    cursor = 0

    for part in re.split(r"\n\s*\n", text):
        paragraph = part.strip()

        if not paragraph:
            cursor += len(part)
            continue

        start = text.find(paragraph, cursor)
        end = start + len(paragraph)

        paragraphs.append(
            {
                "text": paragraph,
                "start_offset": start,
                "end_offset": end,
            }
        )

        cursor = end

    return paragraphs


def classify_role(text: str, fallback_role: str) -> tuple[str, float]:
    lower = text.lower()

    scores = {}

    for role, keywords in ROLE_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in lower:
                score += 1

        scores[role] = score

    best_role = max(scores, key=scores.get)
    best_score = scores[best_role]

    if best_score == 0:
        return fallback_role, 0.50

    confidence = min(0.60 + (best_score * 0.10), 0.95)

    return best_role, round(confidence, 2)


def build_role_segments(section_rows: list[dict]) -> list[dict]:
    output = []

    for section in section_rows:
        section_type = section.get("section_type", "UNKNOWN")
        section_text = section.get("section_text", "")
        paragraphs = split_paragraphs(section_text)

        for index, paragraph in enumerate(paragraphs, start=1):
            role, confidence = classify_role(paragraph["text"], section_type)

            output.append(
                {
                    "role_segment_id": f"{section['section_id']}_ROLE_{index:04d}",
                    "section_id": section["section_id"],
                    "case_id": section["case_id"],
                    "title": section.get("title", ""),
                    "court": section.get("court", ""),
                    "year": section.get("year", ""),
                    "section_type": section_type,
                    "rhetorical_role": role,
                    "confidence": confidence,
                    "start_char": section["start_char"] + paragraph["start_offset"],
                    "end_char": section["start_char"] + paragraph["end_offset"],
                    "text": paragraph["text"],
                }
            )

    return output


def main() -> None:
    sections = read_jsonl(SECTION_SEGMENTS_PATH)
    role_segments = build_role_segments(sections)

    write_jsonl(role_segments, RHETORICAL_ROLE_SEGMENTS_PATH)

    print("\nRhetorical Role Segmentation Report")
    print("=" * 70)
    print(f"Input sections: {len(sections)}")
    print(f"Role segments created: {len(role_segments)}")
    print(f"Output saved to: {RHETORICAL_ROLE_SEGMENTS_PATH}")
    print("=" * 70)

    for row in role_segments[:10]:
        print(
            f"{row['role_segment_id']} | "
            f"{row['rhetorical_role']} | "
            f"confidence={row['confidence']}"
        )


if __name__ == "__main__":
    main()
