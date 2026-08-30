import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    BM25_INDEX_PATH,
    GRAPH_AUGMENTED_RESULTS_PATH,
    LEGAL_KG_NODES_PATH,
)

from src.query.query_analyzer import analyze_query
from src.retrieval.citation_expansion import expand_cases_with_citation_neighbors


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_bm25_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {path}. Run BM25 indexing first."
        )

    with path.open("rb") as file:
        return pickle.load(file)


def rebuild_bm25_from_payload(payload: dict) -> tuple[BM25Okapi, list[dict]]:
    if "metadata" in payload:
        metadata = payload["metadata"]
    elif "chunks" in payload:
        metadata = payload["chunks"]
    elif "documents" in payload:
        metadata = payload["documents"]
    else:
        raise KeyError(
            f"BM25 payload does not contain metadata/chunks/documents. "
            f"Available keys: {list(payload.keys())}"
        )

    if "tokenized_corpus" in payload:
        tokenized_corpus = payload["tokenized_corpus"]
    elif "tokens" in payload:
        tokenized_corpus = payload["tokens"]
    elif "tokenized_documents" in payload:
        tokenized_corpus = payload["tokenized_documents"]
    else:
        tokenized_corpus = [
            tokenize(
                item.get("chunk_text", "")
                or item.get("text", "")
                or item.get("cleaned_text", "")
                or item.get("snippet", "")
            )
            for item in metadata
        ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, metadata



def bm25_initial_search(query: str, top_k: int = 5) -> list[dict]:
    payload = load_bm25_payload(BM25_INDEX_PATH)
    bm25, metadata = rebuild_bm25_from_payload(payload)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )[:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        item = metadata[idx]

        results.append(
            {
                "rank": rank,
                "case_id": str(item.get("case_id", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "title": str(item.get("title", "")),
                "court": str(item.get("court", "")),
                "year": str(item.get("year", "")),
                "snippet": " ".join(str(item.get("chunk_text", "")).split())[:250],
                "base_score": float(scores[idx]),
                "source": "initial_bm25",
            }
        )

    return results


def load_case_node_labels() -> dict:
    if not LEGAL_KG_NODES_PATH.exists():
        return {}

    nodes = pd.read_csv(LEGAL_KG_NODES_PATH)

    labels = {}

    for _, row in nodes.iterrows():
        node_id = str(row.get("node_id", ""))
        labels[node_id] = {
            "title": str(row.get("label", "")),
            "node_type": str(row.get("node_type", "")),
        }

    return labels


def merge_initial_and_neighbors(
    initial_results: list[dict],
    neighbor_rows: list[dict],
) -> list[dict]:
    case_labels = load_case_node_labels()
    merged = {}

    for row in initial_results:
        case_id = row["case_id"]

        if case_id not in merged:
            merged[case_id] = {
                "case_id": case_id,
                "title": row["title"],
                "court": row["court"],
                "year": row["year"],
                "best_chunk_id": row["chunk_id"],
                "snippet": row["snippet"],
                "base_score": row["base_score"],
                "graph_boost": 0.0,
                "final_score": row["base_score"],
                "source": "initial",
                "graph_reason": "",
            }
        else:
            merged[case_id]["base_score"] = max(
                merged[case_id]["base_score"],
                row["base_score"],
            )

    for row in neighbor_rows:
        neighbor_id = row["neighbor_case_id"]

        if neighbor_id not in merged:
            label_info = case_labels.get(neighbor_id, {})

            merged[neighbor_id] = {
                "case_id": neighbor_id,
                "title": label_info.get("title", neighbor_id),
                "court": "",
                "year": "",
                "best_chunk_id": "",
                "snippet": "",
                "base_score": 0.0,
                "graph_boost": row["graph_boost"],
                "final_score": row["graph_boost"],
                "source": "citation_neighbor",
                "graph_reason": (
                    f"{row['seed_case_id']} --{row['edge_type']}--> "
                    f"{neighbor_id} ({row['direction']})"
                ),
            }
        else:
            merged[neighbor_id]["graph_boost"] += row["graph_boost"]
            merged[neighbor_id]["graph_reason"] += (
                f" | {row['seed_case_id']} --{row['edge_type']}--> {neighbor_id}"
            )

    for case_id, row in merged.items():
        row["final_score"] = row["base_score"] + row["graph_boost"]

    ranked = sorted(
        merged.values(),
        key=lambda row: row["final_score"],
        reverse=True,
    )

    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        row["base_score"] = round(float(row["base_score"]), 4)
        row["graph_boost"] = round(float(row["graph_boost"]), 4)
        row["final_score"] = round(float(row["final_score"]), 4)

    return ranked


def write_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "case_id",
        "title",
        "court",
        "year",
        "best_chunk_id",
        "base_score",
        "graph_boost",
        "final_score",
        "source",
        "graph_reason",
        "snippet",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def graph_augmented_search(query: str, top_k: int = 5, neighbor_k: int = 5) -> list[dict]:
    query_analysis = analyze_query(query)
    rewritten_query = query_analysis["rewritten_query"]

    initial_results = bm25_initial_search(rewritten_query, top_k=top_k)

    seed_case_ids = sorted(set(row["case_id"] for row in initial_results))

    neighbor_rows = expand_cases_with_citation_neighbors(
        seed_case_ids=seed_case_ids,
        max_neighbors_per_case=neighbor_k,
    )

    ranked = merge_initial_and_neighbors(initial_results, neighbor_rows)

    write_results(ranked, GRAPH_AUGMENTED_RESULTS_PATH)

    print("\nGap 19/20 Citation Neighborhood Expansion Report")
    print("=" * 70)
    print(f"Original query: {query}")
    print(f"Rewritten query: {rewritten_query}")
    print(f"Initial cases found: {len(seed_case_ids)}")
    print(f"Citation neighbors added: {len(neighbor_rows)}")
    print(f"Final candidates: {len(ranked)}")
    print(f"Output saved to: {GRAPH_AUGMENTED_RESULTS_PATH}")
    print("=" * 70)

    for row in ranked[:10]:
        print(
            f"Rank {row['rank']} | {row['case_id']} | "
            f"final={row['final_score']} | "
            f"base={row['base_score']} | "
            f"graph={row['graph_boost']} | "
            f"{row['title']}"
        )

        if row["graph_reason"]:
            print(f"  graph_reason: {row['graph_reason']}")

    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph-augmented legal retrieval using citation neighborhood expansion."
    )

    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbor-k", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graph_augmented_search(
        query=args.query,
        top_k=args.top_k,
        neighbor_k=args.neighbor_k,
    )


if __name__ == "__main__":
    main()
