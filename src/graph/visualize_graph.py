import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import GRAPH_PATH, REPORTS_DIR


OUTPUT_IMAGE_PATH = REPORTS_DIR / "citation_graph_top30.png"
INTERPRETATION_PATH = REPORTS_DIR / "graph_interpretation.md"
GRAPH_METRICS_PATH = REPORTS_DIR / "graph_metrics.csv"


def load_graph(graph_path: Path) -> nx.DiGraph:
    if not graph_path.exists():
        raise FileNotFoundError(
            f"GraphML file not found: {graph_path}. "
            "Run Day 12 first: python src/graph/build_graph.py"
        )

    graph = nx.read_graphml(graph_path)
    return graph


def load_top_nodes(metrics_path: Path, top_n: int = 30) -> list[str]:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Graph metrics file not found: {metrics_path}. "
            "Run Day 12 first."
        )

    metrics = pd.read_csv(metrics_path)

    if metrics.empty:
        return []

    metrics = metrics.sort_values(
        by=["pagerank", "in_degree"],
        ascending=False,
    )

    return metrics.head(top_n)["node_id"].tolist()


def build_readable_subgraph(
    graph: nx.DiGraph,
    top_nodes: list[str],
) -> nx.DiGraph:
    existing_nodes = [
        node for node in top_nodes
        if node in graph.nodes
    ]

    subgraph = graph.subgraph(existing_nodes).copy()

    return subgraph


def get_node_labels(graph: nx.DiGraph) -> dict:
    labels = {}

    for node_id, data in graph.nodes(data=True):
        label = data.get("label", node_id)

        if len(str(label)) > 28:
            label = str(label)[:25] + "..."

        labels[node_id] = label

    return labels


def get_node_sizes(graph: nx.DiGraph) -> list[int]:
    sizes = []

    for node in graph.nodes:
        in_degree = graph.in_degree(node)
        size = 800 + (in_degree * 300)
        sizes.append(size)

    return sizes


def get_node_colors(graph: nx.DiGraph) -> list[str]:
    colors = []

    for _, data in graph.nodes(data=True):
        node_type = data.get("node_type", "")

        if node_type == "CASE":
            colors.append("lightblue")
        elif node_type == "ARTICLE":
            colors.append("lightgreen")
        elif node_type == "SECTION":
            colors.append("lightyellow")
        else:
            colors.append("lightgray")

    return colors


def visualize_graph(graph: nx.DiGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if graph.number_of_nodes() == 0:
        raise ValueError("Subgraph has no nodes to visualize.")

    plt.figure(figsize=(18, 12))

    layout = nx.spring_layout(
        graph,
        seed=42,
        k=1.2,
        iterations=100,
    )

    labels = get_node_labels(graph)
    node_sizes = get_node_sizes(graph)
    node_colors = get_node_colors(graph)

    nx.draw_networkx_nodes(
        graph,
        layout,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
    )

    nx.draw_networkx_edges(
        graph,
        layout,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        width=1.5,
        alpha=0.6,
    )

    nx.draw_networkx_labels(
        graph,
        layout,
        labels=labels,
        font_size=8,
    )

    plt.title(
        "Top 30 Citation Graph Nodes",
        fontsize=18,
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_interpretation(
    graph: nx.DiGraph,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_by_indegree = sorted(
        graph.nodes,
        key=lambda node: graph.in_degree(node),
        reverse=True,
    )[:10]

    lines = [
        "# Day 13 Graph Interpretation",
        "",
        "## What this visualization shows",
        "",
        "This image shows the top citation graph nodes selected from graph metrics.",
        "Nodes represent cases, constitutional articles, or statutory sections.",
        "Directed arrows represent citation or mention relationships.",
        "",
        "## Important authority nodes",
        "",
    ]

    for node in top_by_indegree:
        label = graph.nodes[node].get("label", node)
        node_type = graph.nodes[node].get("node_type", "")
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)

        lines.append(
            f"- `{node}` | {label} | type={node_type} | "
            f"in_degree={in_degree} | out_degree={out_degree}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Nodes with high in-degree are important because many cases point to them.",
            "In legal research, these are likely to be influential precedents or frequently cited statutes.",
            "",
            "## Limitation",
            "",
            "This visualization only shows the top 30 nodes, so it is a simplified view of the full citation graph.",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    graph = load_graph(GRAPH_PATH)
    top_nodes = load_top_nodes(GRAPH_METRICS_PATH, top_n=30)
    subgraph = build_readable_subgraph(graph, top_nodes)

    visualize_graph(subgraph, OUTPUT_IMAGE_PATH)
    write_interpretation(subgraph, INTERPRETATION_PATH)

    print("\nDay 13 Graph Visualization Report")
    print("=" * 70)
    print(f"Full graph nodes: {graph.number_of_nodes()}")
    print(f"Full graph edges: {graph.number_of_edges()}")
    print(f"Visualized nodes: {subgraph.number_of_nodes()}")
    print(f"Visualized edges: {subgraph.number_of_edges()}")
    print(f"PNG saved to: {OUTPUT_IMAGE_PATH}")
    print(f"Interpretation saved to: {INTERPRETATION_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()