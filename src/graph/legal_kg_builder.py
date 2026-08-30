import csv
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CASE_METADATA_PATH,
    STATUTE_LINKS_PATH,
    LEGAL_CONCEPTS_PATH,
    CASE_OUTCOMES_PATH,
    CITATION_TREATMENTS_PATH,
    CITATION_RESOLUTION_PATH,
    LEGAL_KG_NODES_PATH,
    LEGAL_KG_EDGES_PATH,
    LEGAL_KG_GRAPHML_PATH,
)

from src.graph.kg_schema import make_safe_id


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_node(nodes: dict, node_id: str, node_type: str, label: str, source: str, **props) -> None:
    if not node_id:
        return

    if node_id in nodes:
        nodes[node_id].update({k: v for k, v in props.items() if v})
        return

    nodes[node_id] = {
        "node_id": node_id,
        "node_type": node_type,
        "label": label,
        "source": source,
        **props,
    }


def add_edge(edges: list, source: str, target: str, edge_type: str, confidence: float, source_file: str, **props) -> None:
    if not source or not target:
        return

    edges.append(
        {
            "edge_id": f"KG_EDGE_{len(edges) + 1:06d}",
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "confidence": confidence,
            "source_file": source_file,
            **props,
        }
    )


def build_case_and_court_nodes(metadata: pd.DataFrame, nodes: dict, edges: list) -> None:
    for _, row in metadata.iterrows():
        case_id = str(row.get("case_id", ""))
        title = str(row.get("title", ""))
        court = str(row.get("court", ""))
        year = str(row.get("year", ""))

        add_node(nodes, case_id, "CASE", title, "case_metadata", court=court, year=year)

        court_id = "COURT_" + make_safe_id(court)
        add_node(nodes, court_id, "COURT", court, "case_metadata")

        add_edge(edges, case_id, court_id, "CASE_DECIDED_BY_COURT", 1.0, "case_metadata.csv")


def build_statute_nodes_and_edges(statute_links: pd.DataFrame, nodes: dict, edges: list) -> None:
    for _, row in statute_links.iterrows():
        case_id = str(row.get("case_id", ""))
        mention_type = str(row.get("mention_type", "")).upper()
        section_id = str(row.get("section_id", ""))
        statute_id = str(row.get("statute_id", ""))
        act_name = str(row.get("act_name", ""))
        heading = str(row.get("section_heading", ""))
        full_text = str(row.get("section_full_text", ""))
        confidence = float(row.get("match_confidence", 0.0) or 0.0)

        if not section_id:
            continue

        act_node_id = "ACT_" + make_safe_id(statute_id or act_name)
        section_node_type = "ARTICLE" if mention_type == "ARTICLE" else "SECTION"

        add_node(nodes, act_node_id, "ACT", act_name, "statute_links")

        add_node(
            nodes,
            section_id,
            section_node_type,
            heading or section_id,
            "statute_links",
            full_text=full_text,
            statute_id=statute_id,
        )

        if section_node_type == "ARTICLE":
            add_edge(
                edges,
                case_id,
                section_id,
                "CASE_MENTIONS_ARTICLE",
                confidence,
                "statute_links.csv",
                mention_text=row.get("mention_text", ""),
            )
            add_edge(edges, section_id, act_node_id, "ARTICLE_BELONGS_TO_ACT", 1.0, "statute_links.csv")
        else:
            add_edge(
                edges,
                case_id,
                section_id,
                "CASE_INTERPRETS_SECTION",
                confidence,
                "statute_links.csv",
                mention_text=row.get("mention_text", ""),
            )
            add_edge(edges, section_id, act_node_id, "SECTION_BELONGS_TO_ACT", 1.0, "statute_links.csv")


def build_concept_nodes_and_edges(legal_concepts: pd.DataFrame, nodes: dict, edges: list) -> None:
    for _, row in legal_concepts.iterrows():
        source_type = str(row.get("source_type", ""))
        source_id = str(row.get("source_id", ""))
        concept = str(row.get("concept", ""))
        confidence = float(row.get("confidence", 0.0) or 0.0)

        if source_type != "CASE":
            continue

        concept_id = "CONCEPT_" + make_safe_id(concept)

        add_node(nodes, concept_id, "LEGAL_CONCEPT", concept, "legal_concepts")

        add_edge(
            edges,
            source_id,
            concept_id,
            "CASE_HAS_CONCEPT",
            confidence,
            "legal_concepts.csv",
            matched_keywords=row.get("matched_keywords", ""),
        )


def build_outcome_nodes_and_edges(case_outcomes: pd.DataFrame, nodes: dict, edges: list) -> None:
    for _, row in case_outcomes.iterrows():
        case_id = str(row.get("case_id", ""))
        outcome = str(row.get("outcome", ""))
        holding_text = str(row.get("holding_text", ""))
        outcome_confidence = float(row.get("outcome_confidence", 0.0) or 0.0)
        holding_confidence = float(row.get("holding_confidence", 0.0) or 0.0)

        if outcome and outcome != "UNKNOWN":
            outcome_id = "OUTCOME_" + make_safe_id(outcome)

            add_node(nodes, outcome_id, "OUTCOME", outcome, "case_outcomes")

            add_edge(
                edges,
                case_id,
                outcome_id,
                "CASE_HAS_OUTCOME",
                outcome_confidence,
                "case_outcomes.csv",
                evidence=row.get("outcome_evidence", ""),
            )

        if holding_text:
            holding_id = "HOLDING_" + make_safe_id(case_id)

            add_node(
                nodes,
                holding_id,
                "HOLDING",
                f"Holding for {case_id}",
                "case_outcomes",
                holding_text=holding_text,
            )

            add_edge(edges, case_id, holding_id, "CASE_HAS_HOLDING", holding_confidence, "case_outcomes.csv")


def treatment_edge_type(treatment: str) -> str:
    treatment = str(treatment).upper()

    if treatment == "FOLLOWED":
        return "CASE_FOLLOWS_CASE"

    if treatment == "OVERRULED":
        return "CASE_OVERRULES_CASE"

    if treatment == "DISTINGUISHED":
        return "CASE_DISTINGUISHES_CASE"

    return "CASE_HAS_CITATION_TREATMENT"


def load_citation_resolution_map() -> dict:
    if not CITATION_RESOLUTION_PATH.exists():
        return {}

    resolution = pd.read_csv(CITATION_RESOLUTION_PATH)
    mapping = {}

    for _, row in resolution.iterrows():
        cited_text = str(row.get("cited_case_text", ""))
        resolved_case_id = str(row.get("resolved_case_id", ""))

        if cited_text and resolved_case_id:
            mapping[cited_text] = resolved_case_id

    return mapping


def build_citation_treatment_nodes_and_edges(citation_treatments: pd.DataFrame, nodes: dict, edges: list) -> None:
    citation_resolution_map = load_citation_resolution_map()

    for _, row in citation_treatments.iterrows():
        case_id = str(row.get("case_id", ""))
        cited_case_text = str(row.get("cited_case_text", ""))
        treatment = str(row.get("treatment", ""))
        polarity = str(row.get("polarity", ""))
        confidence = float(row.get("confidence", 0.0) or 0.0)

        if not case_id or not cited_case_text:
            continue

        cited_node_id = citation_resolution_map.get(
            cited_case_text,
            "CASE_REF_" + make_safe_id(cited_case_text),
        )

        treatment_node_id = "TREATMENT_" + make_safe_id(treatment)

        add_node(nodes, cited_node_id, "CASE", cited_case_text, "citation_resolution")

        add_node(
            nodes,
            treatment_node_id,
            "CITATION_TREATMENT",
            treatment,
            "citation_treatments",
            polarity=polarity,
        )

        add_edge(
            edges,
            case_id,
            cited_node_id,
            "CASE_CITES_CASE",
            confidence,
            "citation_treatments.csv",
            treatment=treatment,
            polarity=polarity,
            cited_case_text=cited_case_text,
        )

        add_edge(
            edges,
            case_id,
            cited_node_id,
            treatment_edge_type(treatment),
            confidence,
            "citation_treatments.csv",
            treatment=treatment,
            polarity=polarity,
            cited_case_text=cited_case_text,
        )

        add_edge(
            edges,
            case_id,
            treatment_node_id,
            "CASE_HAS_CITATION_TREATMENT",
            confidence,
            "citation_treatments.csv",
            cited_case_text=cited_case_text,
        )


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_graph(nodes: dict, edges: list) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for node_id, node in nodes.items():
        graph.add_node(node_id, **node)

    for edge in edges:
        graph.add_edge(edge["source"], edge["target"], key=edge["edge_id"], **edge)

    return graph


def main() -> None:
    metadata = read_csv_if_exists(CASE_METADATA_PATH)
    statute_links = read_csv_if_exists(STATUTE_LINKS_PATH)
    legal_concepts = read_csv_if_exists(LEGAL_CONCEPTS_PATH)
    case_outcomes = read_csv_if_exists(CASE_OUTCOMES_PATH)
    citation_treatments = read_csv_if_exists(CITATION_TREATMENTS_PATH)

    nodes = {}
    edges = []

    build_case_and_court_nodes(metadata, nodes, edges)
    build_statute_nodes_and_edges(statute_links, nodes, edges)
    build_concept_nodes_and_edges(legal_concepts, nodes, edges)
    build_outcome_nodes_and_edges(case_outcomes, nodes, edges)
    build_citation_treatment_nodes_and_edges(citation_treatments, nodes, edges)

    node_rows = list(nodes.values())
    edge_rows = edges

    write_csv(node_rows, LEGAL_KG_NODES_PATH)
    write_csv(edge_rows, LEGAL_KG_EDGES_PATH)

    graph = build_graph(nodes, edges)
    LEGAL_KG_GRAPHML_PATH.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, LEGAL_KG_GRAPHML_PATH)

    print("\nFull Legal Knowledge Graph Report")
    print("=" * 70)
    print(f"Nodes created: {len(node_rows)}")
    print(f"Edges created: {len(edge_rows)}")
    print(f"Nodes saved to: {LEGAL_KG_NODES_PATH}")
    print(f"Edges saved to: {LEGAL_KG_EDGES_PATH}")
    print(f"GraphML saved to: {LEGAL_KG_GRAPHML_PATH}")
    print("=" * 70)

    for edge in edge_rows[:15]:
        print(f"{edge['source']} --{edge['edge_type']}--> {edge['target']}")


if __name__ == "__main__":
    main()
