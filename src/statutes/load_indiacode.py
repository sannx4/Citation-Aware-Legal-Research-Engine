import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import RAW_STATUTES_DIR, STATUTES_JSONL_PATH


SAMPLE_STATUTES = [
    {
        "statute_id": "CONSTITUTION_OF_INDIA",
        "act_name": "Constitution of India",
        "short_name": "Constitution",
        "source": "manual_seed",
        "sections": [
            {
                "section_id": "ARTICLE_14",
                "section_type": "ARTICLE",
                "section_number": "14",
                "heading": "Equality before law",
                "text": (
                    "The State shall not deny to any person equality before the law "
                    "or the equal protection of the laws within the territory of India."
                ),
            },
            {
                "section_id": "ARTICLE_19",
                "section_type": "ARTICLE",
                "section_number": "19",
                "heading": "Protection of certain rights regarding freedom of speech etc.",
                "text": (
                    "Article 19 protects certain freedoms, subject to reasonable restrictions "
                    "provided under the Constitution."
                ),
            },
            {
                "section_id": "ARTICLE_21",
                "section_type": "ARTICLE",
                "section_number": "21",
                "heading": "Protection of life and personal liberty",
                "text": (
                    "No person shall be deprived of his life or personal liberty "
                    "except according to procedure established by law."
                ),
            },
        ],
    },
    {
        "statute_id": "IPC",
        "act_name": "Indian Penal Code, 1860",
        "short_name": "IPC",
        "source": "manual_seed",
        "sections": [
            {
                "section_id": "IPC_302",
                "section_type": "SECTION",
                "section_number": "302",
                "heading": "Punishment for murder",
                "text": (
                    "Whoever commits murder shall be punished with death, or imprisonment "
                    "for life, and shall also be liable to fine."
                ),
            },
            {
                "section_id": "IPC_420",
                "section_type": "SECTION",
                "section_number": "420",
                "heading": "Cheating and dishonestly inducing delivery of property",
                "text": (
                    "Whoever cheats and thereby dishonestly induces the person deceived "
                    "to deliver any property shall be punished as provided by law."
                ),
            },
        ],
    },
    {
        "statute_id": "INDIAN_CONTRACT_ACT",
        "act_name": "Indian Contract Act, 1872",
        "short_name": "Contract Act",
        "source": "manual_seed",
        "sections": [
            {
                "section_id": "CONTRACT_10",
                "section_type": "SECTION",
                "section_number": "10",
                "heading": "What agreements are contracts",
                "text": (
                    "All agreements are contracts if they are made by the free consent "
                    "of parties competent to contract, for a lawful consideration and "
                    "with a lawful object, and are not hereby expressly declared to be void."
                ),
            }
        ],
    },
]


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    RAW_STATUTES_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(SAMPLE_STATUTES, STATUTES_JSONL_PATH)

    print("\nStatute Loader Report")
    print("=" * 70)
    print(f"Statutes loaded: {len(SAMPLE_STATUTES)}")
    print(f"Output saved to: {STATUTES_JSONL_PATH}")
    print("=" * 70)

    for statute in SAMPLE_STATUTES:
        print(f"{statute['statute_id']} | sections={len(statute['sections'])}")


if __name__ == "__main__":
    main()
