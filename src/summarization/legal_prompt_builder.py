import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import EVIDENCE_PACK_PATH, LEGAL_PROMPT_PATH


def load_pack(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_prompt(pack: dict) -> str:
    evidence_blocks = []

    for index, item in enumerate(pack["evidence"], start=1):
        evidence_blocks.append(
            f"""
Authority {index}
Case: {item["case"]}
Case ID: {item["case_id"]}
Court: {item["court"]}
Year: {item["year"]}
Score: {item["final_score"]}
Why ranked: {item["why_ranked"]}
Evidence snippet: {item["snippet"]}
""".strip()
        )

    evidence_text = "\n\n".join(evidence_blocks)

    prompt = f"""
You are a legal research assistant for Indian law.

Task:
Write a citation-grounded legal research memorandum.

Question:
{pack["query"]}

Detected intent:
{pack["intent"]}

Detected legal concepts:
{", ".join(pack["concepts"])}

Detected statutes / constitutional provisions:
{", ".join(pack["statutes"])}

Likely precedent cases:
{", ".join(pack["likely_cases"])}

Retrieved legal evidence:
{evidence_text}

Instructions:
1. Use only the supplied evidence.
2. Do not invent cases, statutes, facts, or holdings.
3. If evidence is insufficient, say so clearly.
4. Do not give personal legal advice.
5. Write in this structure:
   - Issue
   - Relevant Law
   - Analysis
   - Conclusion
   - Sources Used

Important:
Every legal claim must be grounded in the retrieved evidence.
""".strip()

    return prompt


def save_prompt(prompt: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(prompt)


def main() -> None:
    pack = load_pack(EVIDENCE_PACK_PATH)
    prompt = build_prompt(pack)

    save_prompt(prompt, LEGAL_PROMPT_PATH)

    print("\nPhase 5.2 Legal Prompt Builder Report")
    print("=" * 70)
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Saved to: {LEGAL_PROMPT_PATH}")
    print("=" * 70)

    print(prompt[:1200])
    print("\n... prompt truncated in console ...")


if __name__ == "__main__":
    main()