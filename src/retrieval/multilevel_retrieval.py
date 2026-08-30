import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CASE_CHUNKS_PATH,
    CLEAN_CASES_PATH,
    STATUTE_MENTIONS_PATH,
    BM25_INDEX_PATH,
    EDGES_PATH,
    REPORTS_DIR,
)

MULTILEVEL_RETRIEVAL_RESULTS_PATH = (
    REPORTS_DIR / "multilevel_retrieval_results.csv"
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def normalize_scores(items: list[dict], score_key: str) -> list[dict]:
    if not items:
        return items

    scores = [float(item.get(score_key, 0.0)) for item in items]
    min_score = min(scores)
    max_score = max(scores)

    for item in items:
        raw = float(item.get(score_key, 0.0))

        if max_score == min_score:
            item[f"{score_key}_norm"] = 1.0
        else:
            item[f"{score_key}_norm"] = (raw - min_score) / (max_score - min_score)

    return items


def build_bm25(texts: list[str]) -> BM25Okapi:
    tokenized = [tokenize(text) for text in texts]
    return BM25Okapi(tokenized)


def retrieve_case_level(query: str, top_k: int = 5) -> list[dict]:
    cases = read_jsonl(CLEAN_CASES_PATH)

    if not cases:
        return []

    texts = [
        case.get("cleaned_text")
        or case.get("raw_text")
        or case.get("text")
        or ""
        for case in cases
    ]

    bm25 = build_bm25(texts)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results = []

    for rank, index in enumerate(ranked, start=1):
        case = cases[index]
        text = texts[index]

        results.append(
            {
                "level": "case",
                "rank": rank,
                "case_id": str(case.get("case_id", "")),
                "title": str(case.get("title", "")),
                "court": str(case.get("court", "")),
                "year": str(case.get("year", "")),
                "evidence_id": str(case.get("case_id", "")),
                "raw_score": float(scores[index]),
                "snippet": " ".join(text.split())[:350],
            }
        )

    return normalize_scores(results, "raw_score")


def retrieve_chunk_level(query: str, top_k: int = 5) -> list[dict]:
    chunks = read_jsonl(CASE_CHUNKS_PATH)

    if not chunks:
        return []

    texts = [chunk.get("chunk_text", "") for chunk in chunks]

    bm25 = build_bm25(texts)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results = []

    for rank, index in enumerate(ranked, start=1):
        chunk = chunks[index]

        results.append(
            {
                "level": "chunk",
                "rank": rank,
                "case_id": str(chunk.get("case_id", "")),
                "title": str(chunk.get("title", "")),
                "court": str(chunk.get("court", "")),
                "year": str(chunk.get("year", "")),
                "evidence_id": str(chunk.get("chunk_id", "")),
                "raw_score": float(scores[index]),
                "snippet": " ".join(chunk.get("chunk_text", "").split())[:350],
            }
        )

    return normalize_scores(results, "raw_score")


def split_paragraphs(chunks: list[dict]) -> list[dict]:
    paragraphs = []

    for chunk in chunks:
        text = chunk.get("chunk_text", "")
        parts = re.split(r"\n\s*\n|(?<=\.)\s{2,}", text)

        para_index = 1

        for part in parts:
            part = " ".join(part.split())

            if len(part) < 40:
                continue

            paragraphs.append(
                {
                    "paragraph_id": f"{chunk.get('chunk_id')}_P{para_index:03d}",
                    "case_id": chunk.get("case_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "title": chunk.get("title", ""),
                    "court": chunk.get("court", ""),
                    "year": chunk.get("year", ""),
                    "paragraph_text": part,
                }
            )

            para_index += 1

    return paragraphs


def retrieve_paragraph_level(query: str, top_k: int = 5) -> list[dict]:
    chunks = read_jsonl(CASE_CHUNKS_PATH)
    paragraphs = split_paragraphs(chunks)

    if not paragraphs:
        return []

    texts = [para["paragraph_text"] for para in paragraphs]

    bm25 = build_bm25(texts)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results = []

    for rank, index in enumerate(ranked, start=1):
        para = paragraphs[index]

        results.append(
            {
                "level": "paragraph",
                "rank": rank,
                "case_id": str(para.get("case_id", "")),
                "title": str(para.get("title", "")),
                "court": str(para.get("court", "")),
                "year": str(para.get("year", "")),
                "evidence_id": str(para.get("paragraph_id", "")),
                "raw_score": float(scores[index]),
                "snippet": para.get("paragraph_text", "")[:350],
            }
        )

    return normalize_scores(results, "raw_score")


def retrieve_statute_level(query: str, top_k: int = 5) -> list[dict]:
    if not STATUTE_MENTIONS_PATH.exists():
        return []

    statutes = pd.read_csv(STATUTE_MENTIONS_PATH)

    if statutes.empty:
        return []

    query_tokens = set(tokenize(query))
    rows = []

    for _, row in statutes.iterrows():
        mention_text = str(row.get("mention_text", ""))
        reference_number = str(row.get("reference_number", ""))
        mention_type = str(row.get("mention_type", ""))

        searchable = f"{mention_type} {mention_text} {reference_number}"
        overlap = len(query_tokens & set(tokenize(searchable)))

        if overlap == 0:
            continue

        rows.append(
            {
                "level": "statute",
                "rank": 0,
                "case_id": str(row.get("case_id", "")),
                "title": "",
                "court": "",
                "year": "",
                "evidence_id": str(row.get("chunk_id", "")),
                "raw_score": float(overlap),
                "snippet": f"{mention_type}: {mention_text}",
            }
        )

    rows = sorted(rows, key=lambda item: item["raw_score"], reverse=True)[:top_k]

    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank

    return normalize_scores(rows, "raw_score")


def retrieve_citation_neighbors(seed_case_ids: list[str], top_k: int = 5) -> list[dict]:
    if not EDGES_PATH.exists():
        return []

    edges = pd.read_csv(EDGES_PATH)

    if edges.empty:
        return []

    neighbors = []
    seen = set()

    for case_id in seed_case_ids:
        outgoing = edges[edges["source"].astype(str) == case_id]
        incoming = edges[edges["target"].astype(str) == case_id]

        for _, row in outgoing.iterrows():
            neighbor_id = str(row.get("target", ""))

            if not neighbor_id or neighbor_id in seen:
                continue

            seen.add(neighbor_id)

            neighbors.append(
                {
                    "level": "citation_neighbor",
                    "rank": 0,
                    "case_id": neighbor_id,
                    "title": "",
                    "court": "",
                    "year": "",
                    "evidence_id": neighbor_id,
                    "raw_score": float(row.get("confidence", 0.5) or 0.5),
                    "snippet": f"{case_id} cites {neighbor_id}",
                }
            )

        for _, row in incoming.iterrows():
            neighbor_id = str(row.get("source", ""))

            if not neighbor_id or neighbor_id in seen:
                continue

            seen.add(neighbor_id)

            neighbors.append(
                {
                    "level": "citation_neighbor",
                    "rank": 0,
                    "case_id": neighbor_id,
                    "title": "",
                    "court": "",
                    "year": "",
                    "evidence_id": neighbor_id,
                    "raw_score": float(row.get("confidence", 0.5) or 0.5),
                    "snippet": f"{neighbor_id} cites {case_id}",
                }
            )

    neighbors = sorted(
        neighbors,
        key=lambda item: item["raw_score"],
        reverse=True,
    )[:top_k]

    for rank, item in enumerate(neighbors, start=1):
        item["rank"] = rank

    return normalize_scores(neighbors, "raw_score")


def fuse_results(
    case_results: list[dict],
    chunk_results: list[dict],
    paragraph_results: list[dict],
    statute_results: list[dict],
    neighbor_results: list[dict],
) -> list[dict]:

    weights = {
        "case": 0.25,
        "chunk": 0.25,
        "paragraph": 0.25,
        "statute": 0.15,
        "citation_neighbor": 0.10,
    }

    all_rows = (
        case_results
        + chunk_results
        + paragraph_results
        + statute_results
        + neighbor_results
    )

    fused = {}

    for row in all_rows:
        case_id = row.get("case_id", "")

        if not case_id:
            continue

        level = row["level"]
        score = row.get("raw_score_norm", 0.0)
        weighted_score = weights.get(level, 0.0) * score

        if case_id not in fused:
            fused[case_id] = {
                "case_id": case_id,
                "title": row.get("title", ""),
                "court": row.get("court", ""),
                "year": row.get("year", ""),
                "fusion_score": 0.0,
                "best_case": "",
                "best_paragraph": "",
                "related_statute": "",
                "citation_neighbors": [],
                "evidence_sources": [],
            }

        fused[case_id]["fusion_score"] += weighted_score
        fused[case_id]["evidence_sources"].append(level)

        if level == "case" and not fused[case_id]["best_case"]:
            fused[case_id]["best_case"] = row.get("snippet", "")

        if level in {"chunk", "paragraph"}:
            current = fused[case_id]["best_paragraph"]

            if not current or len(row.get("snippet", "")) > len(current):
                fused[case_id]["best_paragraph"] = row.get("snippet", "")

        if level == "statute" and not fused[case_id]["related_statute"]:
            fused[case_id]["related_statute"] = row.get("snippet", "")

        if level == "citation_neighbor":
            fused[case_id]["citation_neighbors"].append(row.get("snippet", ""))

    ranked = sorted(
        fused.values(),
        key=lambda item: item["fusion_score"],
        reverse=True,
    )

    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["fusion_score"] = round(row["fusion_score"], 4)
        row["evidence_sources"] = sorted(set(row["evidence_sources"]))
        row["citation_neighbors"] = row["citation_neighbors"][:5]

    return ranked


def write_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "case_id",
        "title",
        "court",
        "year",
        "fusion_score",
        "evidence_sources",
        "best_case",
        "best_paragraph",
        "related_statute",
        "citation_neighbors",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    **row,
                    "evidence_sources": ";".join(row.get("evidence_sources", [])),
                    "citation_neighbors": " | ".join(row.get("citation_neighbors", [])),
                }
            )


def multilevel_retrieval(
    query: str,
    top_k: int = 5,
) -> list[dict]:

    case_results = retrieve_case_level(query, top_k)
    chunk_results = retrieve_chunk_level(query, top_k)
    paragraph_results = retrieve_paragraph_level(query, top_k)
    statute_results = retrieve_statute_level(query, top_k)

    seed_case_ids = sorted(
        {
            row["case_id"]
            for row in case_results + chunk_results + paragraph_results
            if row.get("case_id")
        }
    )

    neighbor_results = retrieve_citation_neighbors(seed_case_ids, top_k)

    fused = fuse_results(
        case_results=case_results,
        chunk_results=chunk_results,
        paragraph_results=paragraph_results,
        statute_results=statute_results,
        neighbor_results=neighbor_results,
    )

    fused = fused[:top_k]

    write_results(fused, MULTILEVEL_RETRIEVAL_RESULTS_PATH)

    print("\nGap 19 Multi-Level Retrieval Fusion Report")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"Case-level results: {len(case_results)}")
    print(f"Chunk-level results: {len(chunk_results)}")
    print(f"Paragraph-level results: {len(paragraph_results)}")
    print(f"Statute-level results: {len(statute_results)}")
    print(f"Citation-neighbor results: {len(neighbor_results)}")
    print(f"Final fused cases: {len(fused)}")
    print(f"Output saved to: {MULTILEVEL_RETRIEVAL_RESULTS_PATH}")
    print("=" * 70)

    for row in fused:
        print(
            f"\nRank {row['rank']} | {row['case_id']} | "
            f"score={row['fusion_score']} | {row['title']}"
        )
        print(f"Sources: {', '.join(row['evidence_sources'])}")

        if row["related_statute"]:
            print(f"Related statute: {row['related_statute']}")

        if row["best_paragraph"]:
            print(f"Best paragraph: {row['best_paragraph'][:250]}...")

        if row["citation_neighbors"]:
            print("Citation neighbors:")
            for neighbor in row["citation_neighbors"]:
                print(f"  - {neighbor}")

    return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-level legal retrieval fusion."
    )

    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    multilevel_retrieval(
        query=args.query,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
