import csv
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CASE_METADATA_PATH,
    NORMALIZED_CITATIONS_PATH,
    NODES_PATH,
    EDGES_PATH,
)


def create_case_nodes(metadata: pd.DataFrame) -> list[dict]:
    nodes = []

    for _, row in metadata.iterrows():
        nodes.append(
            {
                "node_id": row["case_id"],
                "node_type": "CASE",
                "label": row["title"],
                "source": "case_metadata",
            }
        )

    return nodes


def create_citation_nodes(normalized: pd.DataFrame) -> list[dict]:
    nodes = []
    seen = set()

    for _, row in normalized.iterrows():
        canonical_id = row["canonical_id"]

        if canonical_id in seen:
            continue

        seen.add(canonical_id)

        nodes.append(
            {
                "node_id": canonical_id,
                "node_type": row["mention_type"],
                "label": row["canonical_name"],
                "source": "normalized_citations",
            }
        )

    return nodes


def create_edges(normalized: pd.DataFrame) -> list[dict]:
    edges = []

    for index, row in normalized.iterrows():
        source_case_id = row["source_case_id"]
        target_id = row["canonical_id"]
        mention_type = row["mention_type"]

        if mention_type == "CASE":
            edge_type = "CITES_CASE"
        elif mention_type in {"ARTICLE", "SECTION"}:
            edge_type = "MENTIONS_STATUTE"
        else:
            edge_type = "MENTIONS_ENTITY"

        edges.append(
            {
                "edge_id": f"EDGE_{index + 1:05d}",
                "source": source_case_id,
                "target": target_id,
                "edge_type": edge_type,
                "confidence": row["confidence"],
                "chunk_id": row["chunk_id"],
                "original_mention": row["original_mention"],
            }
        )

    return edges


def write_csv(rows: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main() -> None:
    if not CASE_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {CASE_METADATA_PATH}")

    if not NORMALIZED_CITATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing normalized citations file: {NORMALIZED_CITATIONS_PATH}"
        )

    metadata = pd.read_csv(CASE_METADATA_PATH)
    normalized = pd.read_csv(NORMALIZED_CITATIONS_PATH)

    case_nodes = create_case_nodes(metadata)
    citation_nodes = create_citation_nodes(normalized)
    all_nodes = case_nodes + citation_nodes

    edges = create_edges(normalized)

    write_csv(
        all_nodes,
        NODES_PATH,
        ["node_id", "node_type", "label", "source"],
    )

    write_csv(
        edges,
        EDGES_PATH,
        [
            "edge_id",
            "source",
            "target",
            "edge_type",
            "confidence",
            "chunk_id",
            "original_mention",
        ],
    )

    print("\nDay 11 Citation Edge List Report")
    print("=" * 70)
    print(f"Case nodes: {len(case_nodes)}")
    print(f"Citation/statute nodes: {len(citation_nodes)}")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Total edges: {len(edges)}")
    print(f"Nodes saved to: {NODES_PATH}")
    print(f"Edges saved to: {EDGES_PATH}")
    print("=" * 70)

    for edge in edges[:10]:
        print(
            f"{edge['source']} -> {edge['target']} "
            f"({edge['edge_type']})"
        )


if __name__ == "__main__":
    main()