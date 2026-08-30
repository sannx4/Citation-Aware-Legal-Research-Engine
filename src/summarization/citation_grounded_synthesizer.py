import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    RAG_ANSWER_PATH,
    CITATION_GROUNDED_ANSWER_PATH,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def synthesize_markdown(data: dict) -> str:
    lines = []

    lines.append("# Citation-Grounded Legal Research Answer")
    lines.append("")
    lines.append(f"**Query:** {data['query']}")
    lines.append("")
    lines.append("## Short Answer")
    lines.append("")
    lines.append(data["answer"])
    lines.append("")
    lines.append("## Legal Concepts Detected")
    lines.append("")

    for concept in data.get("concepts", []):
        lines.append(f"- {concept}")

    lines.append("")
    lines.append("## Statutes / Constitutional Provisions")
    lines.append("")

    for statute in data.get("statutes", []):
        lines.append(f"- {statute}")

    lines.append("")
    lines.append("## Supporting Cases")
    lines.append("")

    for item in data.get("evidence", []):
        lines.append(f"### {item['case']}")
        lines.append("")
        lines.append(f"- **Case ID:** {item['case_id']}")
        lines.append(f"- **Court:** {item['court']}")
        lines.append(f"- **Year:** {item['year']}")
        lines.append(f"- **Why ranked:** {item['why_ranked']}")
        lines.append(f"- **Evidence snippet:** {item['snippet']}")
        lines.append("")

    lines.append("## Disclaimer")
    lines.append("")
    lines.append(data.get("disclaimer", "Research assistance only. Not legal advice."))

    return "\n".join(lines)


def main() -> None:
    data = load_json(RAG_ANSWER_PATH)

    markdown = synthesize_markdown(data)

    CITATION_GROUNDED_ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CITATION_GROUNDED_ANSWER_PATH.open("w", encoding="utf-8") as file:
        file.write(markdown)

    print("\nCitation-Grounded Answer Synthesis Report")
    print("=" * 70)
    print(f"Saved to: {CITATION_GROUNDED_ANSWER_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()