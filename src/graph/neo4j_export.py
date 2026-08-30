import csv
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    LEGAL_KG_NODES_PATH,
    LEGAL_KG_EDGES_PATH,
    NEO4J_NODES_EXPORT_PATH,
    NEO4J_EDGES_EXPORT_PATH,
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def export_nodes(nodes: pd.DataFrame) -> list[dict]:
    rows = []

    for _, row in nodes.iterrows():
        node_id = row.get("node_id", "")
        node_type = row.get("node_type", "")
        label = row.get("label", "")

        rows.append(
            {
                "id:ID": node_id,
                ":LABEL": node_type,
                "name": label,
                "source": row.get("source", ""),
            }
        )

    return rows


def export_edges(edges: pd.DataFrame) -> list[dict]:
    rows = []

    for _, row in edges.iterrows():
        rows.append(
            {
                ":START_ID": row.get("source", ""),
                ":END_ID": row.get("target", ""),
                ":TYPE": row.get("edge_type", ""),
                "edge_id": row.get("edge_id", ""),
                "confidence": row.get("confidence", ""),
                "source_file": row.get("source_file", ""),
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    nodes = read_csv(LEGAL_KG_NODES_PATH)
    edges = read_csv(LEGAL_KG_EDGES_PATH)

    neo4j_nodes = export_nodes(nodes)
    neo4j_edges = export_edges(edges)

    write_csv(neo4j_nodes, NEO4J_NODES_EXPORT_PATH)
    write_csv(neo4j_edges, NEO4J_EDGES_EXPORT_PATH)

    print("\nNeo4j Export Report")
    print("=" * 70)
    print(f"Neo4j nodes exported: {len(neo4j_nodes)}")
    print(f"Neo4j edges exported: {len(neo4j_edges)}")
    print(f"Nodes export saved to: {NEO4J_NODES_EXPORT_PATH}")
    print(f"Edges export saved to: {NEO4J_EDGES_EXPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
