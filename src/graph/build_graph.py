import csv
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    NODES_PATH,
    EDGES_PATH,
    GRAPH_PATH,
    REPORTS_DIR,
)


GRAPH_METRICS_PATH = REPORTS_DIR / "graph_metrics.csv"


def load_nodes(nodes_path: Path) -> pd.DataFrame:
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")

    return pd.read_csv(nodes_path)


def load_edges(edges_path: Path) -> pd.DataFrame:
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")

    return pd.read_csv(edges_path)


def build_directed_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> nx.DiGraph:

    graph = nx.DiGraph()

    for _, row in nodes.iterrows():
        graph.add_node(
            row["node_id"],
            node_type=row["node_type"],
            label=row["label"],
            source=row["source"],
        )

    for _, row in edges.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            edge_id=row["edge_id"],
            edge_type=row["edge_type"],
            confidence=float(row["confidence"]),
            chunk_id=row["chunk_id"],
            original_mention=row["original_mention"],
        )

    return graph


def compute_graph_metrics(graph: nx.DiGraph) -> list[dict]:
    pagerank_scores = nx.pagerank(graph) if graph.number_of_nodes() > 0 else {}

    metrics = []

    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]

        metrics.append(
            {
                "node_id": node_id,
                "label": node_data.get("label", ""),
                "node_type": node_data.get("node_type", ""),
                "in_degree": graph.in_degree(node_id),
                "out_degree": graph.out_degree(node_id),
                "pagerank": pagerank_scores.get(node_id, 0.0),
            }
        )

    metrics.sort(
        key=lambda row: (
            row["pagerank"],
            row["in_degree"],
        ),
        reverse=True,
    )

    return metrics


def write_metrics_csv(
    metrics: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "node_id",
        "label",
        "node_type",
        "in_degree",
        "out_degree",
        "pagerank",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in metrics:
            writer.writerow(row)


def export_graphml(
    graph: nx.DiGraph,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)


def main() -> None:
    nodes = load_nodes(NODES_PATH)
    edges = load_edges(EDGES_PATH)

    graph = build_directed_graph(nodes, edges)

    metrics = compute_graph_metrics(graph)

    export_graphml(graph, GRAPH_PATH)
    write_metrics_csv(metrics, GRAPH_METRICS_PATH)

    print("\nDay 12 NetworkX Citation Graph Report")
    print("=" * 70)
    print(f"Nodes loaded: {graph.number_of_nodes()}")
    print(f"Edges loaded: {graph.number_of_edges()}")
    print(f"Graph saved to: {GRAPH_PATH}")
    print(f"Metrics saved to: {GRAPH_METRICS_PATH}")
    print("=" * 70)

    print("Top authority nodes:")
    for row in metrics[:10]:
        print(
            f"{row['node_id']} | "
            f"in_degree={row['in_degree']} | "
            f"pagerank={row['pagerank']:.6f} | "
            f"{row['label']}"
        )


if __name__ == "__main__":
    main()