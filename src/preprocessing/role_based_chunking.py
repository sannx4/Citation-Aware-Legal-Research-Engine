import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    RHETORICAL_ROLE_SEGMENTS_PATH,
    ROLE_BASED_CHUNKS_PATH,
    ROLE_CHUNKING_REPORT_PATH,
)

from src.preprocessing.legal_chunker import split_paragraph_aware


ROLE_WEIGHTS = {
    "FACTS": 0.70,
    "ISSUES": 0.90,
    "ARGUMENTS": 0.80,
    "REASONING": 1.00,
    "HOLDING": 1.00,
    "DECISION": 0.95,
    "UNKNOWN": 0.50,
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}. Run rhetorical_role_segmenter.py first."
        )

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


def safe_role(role: str) -> str:
    role = str(role).upper().strip()

    if not role:
        return "UNKNOWN"

    return role.replace(" ", "_")


def make_chunk_id(case_id: str, role: str, index: int) -> str:
    return f"{case_id}_{safe_role(role)}_{index:04d}"


def build_role_based_chunks(
    role_segments: list[dict],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict]:

    chunks = []
    counters = {}

    for segment in role_segments:
        case_id = str(segment.get("case_id", "UNKNOWN_CASE"))
        role = safe_role(segment.get("rhetorical_role", "UNKNOWN"))
        text = segment.get("text", "")

        if not text.strip():
            continue

        local_chunks = split_paragraph_aware(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for local_chunk in local_chunks:
            key = (case_id, role)
            counters[key] = counters.get(key, 0) + 1

            chunk_index = counters[key]
            chunk_id = make_chunk_id(case_id, role, chunk_index)

            global_start = int(segment.get("start_char", 0)) + local_chunk["start_offset"]
            global_end = int(segment.get("start_char", 0)) + local_chunk["end_offset"]

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "case_id": case_id,
                    "title": segment.get("title", ""),
                    "court": segment.get("court", ""),
                    "year": segment.get("year", ""),
                    "role": role,
                    "section_type": segment.get("section_type", ""),
                    "role_confidence": segment.get("confidence", 0.0),
                    "role_weight": ROLE_WEIGHTS.get(role, 0.50),
                    "source_role_segment_id": segment.get("role_segment_id", ""),
                    "chunk_index_within_role": chunk_index,
                    "chunk_text": local_chunk["chunk_text"],
                    "start_char": global_start,
                    "end_char": global_end,
                    "chunk_size": local_chunk["chunk_size"],
                }
            )

    return chunks


def write_report(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = {}

    for chunk in chunks:
        role = chunk["role"]

        if role not in summary:
            summary[role] = {
                "role": role,
                "chunk_count": 0,
                "avg_chunk_size": 0,
                "role_weight": chunk["role_weight"],
            }

        summary[role]["chunk_count"] += 1
        summary[role]["avg_chunk_size"] += chunk["chunk_size"]

    rows = []

    for role, data in summary.items():
        count = data["chunk_count"]

        rows.append(
            {
                "role": role,
                "chunk_count": count,
                "avg_chunk_size": round(data["avg_chunk_size"] / count, 2),
                "role_weight": data["role_weight"],
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "role",
                "chunk_count",
                "avg_chunk_size",
                "role_weight",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    role_segments = read_jsonl(RHETORICAL_ROLE_SEGMENTS_PATH)

    chunks = build_role_based_chunks(
        role_segments=role_segments,
        chunk_size=900,
        overlap=120,
    )

    write_jsonl(chunks, ROLE_BASED_CHUNKS_PATH)
    write_report(chunks, ROLE_CHUNKING_REPORT_PATH)

    print("\nGap 14 Rhetorical Role Chunking Report")
    print("=" * 70)
    print(f"Input role segments: {len(role_segments)}")
    print(f"Role-based chunks created: {len(chunks)}")
    print(f"Chunks saved to: {ROLE_BASED_CHUNKS_PATH}")
    print(f"Report saved to: {ROLE_CHUNKING_REPORT_PATH}")
    print("=" * 70)

    for chunk in chunks[:10]:
        print(
            f"{chunk['chunk_id']} | "
            f"role={chunk['role']} | "
            f"weight={chunk['role_weight']} | "
            f"chars={chunk['start_char']}-{chunk['end_char']}"
        )


if __name__ == "__main__":
    main()
