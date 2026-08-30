import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import LEGAL_KG_EDGES_PATH


CITATION_EDGE_TYPES = {
    "CASE_CITES_CASE",
    "CASE_FOLLOWS_CASE",
    "CASE_OVERRULES_CASE",
    "CASE_DISTINGUISHES_CASE",
}


def load_kg_edges() -> pd.DataFrame:
    if not LEGAL_KG_EDGES_PATH.exists():
        raise FileNotFoundError(
            f"Legal KG edges not found: {LEGAL_KG_EDGES_PATH}. "
            "Run: python src/graph/legal_kg_builder.py"
        )

    return pd.read_csv(LEGAL_KG_EDGES_PATH)


def get_outgoing_neighbors(case_id: str, edges: pd.DataFrame) -> list[dict]:
    rows = edges[
        (edges["source"].astype(str) == case_id)
        & (edges["edge_type"].astype(str).isin(CITATION_EDGE_TYPES))
    ]

    neighbors = []

    for _, row in rows.iterrows():
        neighbors.append(
            {
                "neighbor_case_id": str(row["target"]),
                "direction": "outgoing",
                "edge_type": str(row["edge_type"]),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
            }
        )

    return neighbors


def get_incoming_neighbors(case_id: str, edges: pd.DataFrame) -> list[dict]:
    rows = edges[
        (edges["target"].astype(str) == case_id)
        & (edges["edge_type"].astype(str).isin(CITATION_EDGE_TYPES))
    ]

    neighbors = []

    for _, row in rows.iterrows():
        neighbors.append(
            {
                "neighbor_case_id": str(row["source"]),
                "direction": "incoming",
                "edge_type": str(row["edge_type"]),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
            }
        )

    return neighbors


def treatment_boost(edge_type: str) -> float:
    edge_type = str(edge_type).upper()

    if edge_type == "CASE_FOLLOWS_CASE":
        return 0.25

    if edge_type == "CASE_CITES_CASE":
        return 0.18

    if edge_type == "CASE_DISTINGUISHES_CASE":
        return 0.08

    if edge_type == "CASE_OVERRULES_CASE":
        return -0.20

    return 0.05


def expand_cases_with_citation_neighbors(
    seed_case_ids: list[str],
    max_neighbors_per_case: int = 5,
) -> list[dict]:
    edges = load_kg_edges()
    expanded = []

    seen = set()

    for seed_case_id in seed_case_ids:
        neighbors = []
        neighbors.extend(get_outgoing_neighbors(seed_case_id, edges))
        neighbors.extend(get_incoming_neighbors(seed_case_id, edges))

        neighbors = sorted(
            neighbors,
            key=lambda row: row["confidence"],
            reverse=True,
        )[:max_neighbors_per_case]

        for neighbor in neighbors:
            neighbor_id = neighbor["neighbor_case_id"]

            key = (seed_case_id, neighbor_id, neighbor["edge_type"])

            if key in seen:
                continue

            seen.add(key)

            expanded.append(
                {
                    "seed_case_id": seed_case_id,
                    "neighbor_case_id": neighbor_id,
                    "direction": neighbor["direction"],
                    "edge_type": neighbor["edge_type"],
                    "edge_confidence": neighbor["confidence"],
                    "graph_boost": treatment_boost(neighbor["edge_type"]),
                }
            )

    return expanded





