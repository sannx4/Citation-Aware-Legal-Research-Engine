import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CASE_CHUNKS_PATH,
    EMBEDDING_BENCHMARK_QUERIES_PATH,
    EMBEDDING_BENCHMARK_QRELS_PATH,
    EMBEDDING_MODEL_COMPARISON_PATH,
)


@dataclass
class ModelSpec:
    name: str
    model_id: str
    backend: str
    query_prefix: str = ""
    passage_prefix: str = ""
    enabled: bool = True
    note: str = ""


def get_model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="all-MiniLM-L6-v2",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            backend="sentence_transformers",
        ),
        ModelSpec(
            name="BGE-small-en-v1.5",
            model_id="BAAI/bge-small-en-v1.5",
            backend="sentence_transformers",
        ),
        ModelSpec(
            name="E5-small-v2",
            model_id="intfloat/e5-small-v2",
            backend="sentence_transformers",
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        ModelSpec(
            name="LegalBERT",
            model_id="nlpaueb/legal-bert-base-uncased",
            backend="hf_mean_pooling",
        ),
        ModelSpec(
            name="InLegalBERT",
            model_id="law-ai/InLegalBERT",
            backend="hf_mean_pooling",
        ),
    ]

    domain_model_path = ROOT_DIR / "models" / "legal_biencoder"

    if domain_model_path.exists():
        specs.append(
            ModelSpec(
                name="domain-finetuned",
                model_id=str(domain_model_path),
                backend="sentence_transformers",
            )
        )
    else:
        specs.append(
            ModelSpec(
                name="domain-finetuned",
                model_id=str(domain_model_path),
                backend="missing_local_model",
                enabled=False,
                note="Skipped because models/legal_biencoder does not exist yet.",
            )
        )

    return specs


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_chunks() -> tuple[list[str], list[dict]]:
    chunks = read_jsonl(CASE_CHUNKS_PATH)

    texts = []
    metadata = []

    for chunk in chunks:
        text = str(chunk.get("chunk_text", "")).strip()

        if not text:
            continue

        texts.append(text)
        metadata.append(
            {
                "case_id": chunk.get("case_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "title": chunk.get("title", ""),
                "court": chunk.get("court", ""),
                "year": chunk.get("year", ""),
            }
        )

    if not texts:
        raise ValueError("No chunk text found. Run chunking first.")

    return texts, metadata


def ensure_default_eval_data() -> None:
    EMBEDDING_BENCHMARK_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not EMBEDDING_BENCHMARK_QUERIES_PATH.exists():
        query_rows = [
            {
                "query_id": "Q001",
                "query": "Article 21 privacy",
            },
            {
                "query_id": "Q002",
                "query": "fair procedure under Article 21",
            },
            {
                "query_id": "Q003",
                "query": "preventive detention personal liberty",
            },
            {
                "query_id": "Q004",
                "query": "right to privacy fundamental right",
            },
            {
                "query_id": "Q005",
                "query": "passport impounding personal liberty",
            },
        ]

        pd.DataFrame(query_rows).to_csv(
            EMBEDDING_BENCHMARK_QUERIES_PATH,
            index=False,
        )

    if not EMBEDDING_BENCHMARK_QRELS_PATH.exists():
        qrels_rows = [
            {"query_id": "Q001", "case_id": "CASE_001", "relevance": 3},
            {"query_id": "Q001", "case_id": "CASE_002", "relevance": 2},
            {"query_id": "Q002", "case_id": "CASE_002", "relevance": 3},
            {"query_id": "Q002", "case_id": "CASE_001", "relevance": 2},
            {"query_id": "Q002", "case_id": "CASE_003", "relevance": 1},
            {"query_id": "Q003", "case_id": "CASE_003", "relevance": 3},
            {"query_id": "Q003", "case_id": "CASE_002", "relevance": 2},
            {"query_id": "Q004", "case_id": "CASE_001", "relevance": 3},
            {"query_id": "Q005", "case_id": "CASE_002", "relevance": 3},
        ]

        pd.DataFrame(qrels_rows).to_csv(
            EMBEDDING_BENCHMARK_QRELS_PATH,
            index=False,
        )


def load_eval_data() -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    ensure_default_eval_data()

    queries = pd.read_csv(EMBEDDING_BENCHMARK_QUERIES_PATH)
    qrels_df = pd.read_csv(EMBEDDING_BENCHMARK_QRELS_PATH)

    qrels: dict[str, dict[str, int]] = {}

    for _, row in qrels_df.iterrows():
        query_id = str(row["query_id"])
        case_id = str(row["case_id"])
        relevance = int(row["relevance"])

        qrels.setdefault(query_id, {})
        qrels[query_id][case_id] = relevance

    return queries, qrels


def cleanup_memory() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def encode_with_sentence_transformers(
    spec: ModelSpec,
    texts: list[str],
    is_query: bool,
    local_only: bool,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model_kwargs = {}

    if local_only:
        model_kwargs["local_files_only"] = True

    model = SentenceTransformer(
        spec.model_id,
        **model_kwargs,
    )

    prefix = spec.query_prefix if is_query else spec.passage_prefix
    encoded_texts = [prefix + text for text in texts]

    embeddings = model.encode(
        encoded_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    del model
    cleanup_memory()

    return embeddings


def mean_pooling(last_hidden_state, attention_mask):
    import torch

    input_mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(last_hidden_state * input_mask, dim=1)

    counts = torch.clamp(
        input_mask.sum(dim=1),
        min=1e-9,
    )

    return summed / counts


def encode_with_hf_mean_pooling(
    spec: ModelSpec,
    texts: list[str],
    is_query: bool,
    local_only: bool,
    batch_size: int,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        local_files_only=local_only,
    )

    model = AutoModel.from_pretrained(
        spec.model_id,
        local_files_only=local_only,
    )

    model.eval()

    prefix = spec.query_prefix if is_query else spec.passage_prefix
    encoded_texts = [prefix + text for text in texts]

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(encoded_texts), batch_size):
            batch = encoded_texts[start:start + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )

            output = model(**encoded)
            embeddings = mean_pooling(
                output.last_hidden_state,
                encoded["attention_mask"],
            )

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1,
            )

            all_embeddings.append(
                embeddings.cpu().numpy().astype("float32")
            )

    del model
    del tokenizer
    cleanup_memory()

    return np.vstack(all_embeddings).astype("float32")


def encode_texts(
    spec: ModelSpec,
    texts: list[str],
    is_query: bool,
    local_only: bool,
    batch_size: int,
) -> np.ndarray:
    if spec.backend == "sentence_transformers":
        return encode_with_sentence_transformers(
            spec=spec,
            texts=texts,
            is_query=is_query,
            local_only=local_only,
            batch_size=batch_size,
        )

    if spec.backend == "hf_mean_pooling":
        return encode_with_hf_mean_pooling(
            spec=spec,
            texts=texts,
            is_query=is_query,
            local_only=local_only,
            batch_size=batch_size,
        )

    raise ValueError(f"Unsupported backend: {spec.backend}")


def rank_cases(
    query_vector: np.ndarray,
    doc_embeddings: np.ndarray,
    metadata: list[dict],
) -> list[dict]:
    scores = doc_embeddings @ query_vector

    sorted_indices = np.argsort(scores)[::-1]

    seen_cases = set()
    ranked_cases = []

    for index in sorted_indices:
        case_id = metadata[index]["case_id"]

        if case_id in seen_cases:
            continue

        seen_cases.add(case_id)

        ranked_cases.append(
            {
                "case_id": case_id,
                "chunk_id": metadata[index]["chunk_id"],
                "title": metadata[index]["title"],
                "score": float(scores[index]),
            }
        )

    return ranked_cases


def recall_at_k(
    ranked_cases: list[dict],
    relevant_cases: dict[str, int],
    k: int,
) -> float:
    if not relevant_cases:
        return 0.0

    top_cases = {
        row["case_id"]
        for row in ranked_cases[:k]
    }

    retrieved_relevant = sum(
        1 for case_id in relevant_cases
        if case_id in top_cases
    )

    return retrieved_relevant / len(relevant_cases)


def mrr_score(
    ranked_cases: list[dict],
    relevant_cases: dict[str, int],
) -> float:
    for rank, row in enumerate(ranked_cases, start=1):
        if row["case_id"] in relevant_cases:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    ranked_cases: list[dict],
    relevant_cases: dict[str, int],
    k: int,
) -> float:
    score = 0.0

    for index, row in enumerate(ranked_cases[:k], start=1):
        relevance = relevant_cases.get(row["case_id"], 0)

        if relevance <= 0:
            continue

        score += (2**relevance - 1) / math.log2(index + 1)

    return score


def ndcg_at_k(
    ranked_cases: list[dict],
    relevant_cases: dict[str, int],
    k: int,
) -> float:
    dcg = dcg_at_k(
        ranked_cases=ranked_cases,
        relevant_cases=relevant_cases,
        k=k,
    )

    ideal_relevances = sorted(
        relevant_cases.values(),
        reverse=True,
    )[:k]

    ideal_score = 0.0

    for index, relevance in enumerate(ideal_relevances, start=1):
        ideal_score += (2**relevance - 1) / math.log2(index + 1)

    if ideal_score == 0:
        return 0.0

    return dcg / ideal_score


def evaluate_model(
    spec: ModelSpec,
    chunk_texts: list[str],
    metadata: list[dict],
    queries_df: pd.DataFrame,
    qrels: dict[str, dict[str, int]],
    top_k: int,
    local_only: bool,
    batch_size: int,
) -> dict:
    if not spec.enabled:
        return {
            "model_name": spec.name,
            "model_id": spec.model_id,
            "backend": spec.backend,
            "status": "skipped",
            "error": spec.note,
            "chunks_indexed": len(chunk_texts),
            "queries": len(queries_df),
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
            "embedding_build_time_sec": 0.0,
            "avg_query_latency_ms": 0.0,
            "index_size_mb": 0.0,
        }

    start_time = time.perf_counter()

    doc_embeddings = encode_texts(
        spec=spec,
        texts=chunk_texts,
        is_query=False,
        local_only=local_only,
        batch_size=batch_size,
    )

    embedding_build_time = time.perf_counter() - start_time
    index_size_mb = doc_embeddings.nbytes / (1024 * 1024)

    query_texts = [
        str(row["query"])
        for _, row in queries_df.iterrows()
    ]

    query_start = time.perf_counter()

    query_embeddings = encode_texts(
        spec=spec,
        texts=query_texts,
        is_query=True,
        local_only=local_only,
        batch_size=batch_size,
    )

    query_rankings = []

    for query_index, (_, query_row) in enumerate(queries_df.iterrows()):
        query_vector = query_embeddings[query_index]
        ranked_cases = rank_cases(
            query_vector=query_vector,
            doc_embeddings=doc_embeddings,
            metadata=metadata,
        )

        query_rankings.append(
            {
                "query_id": str(query_row["query_id"]),
                "ranked_cases": ranked_cases,
            }
        )

    total_query_time = time.perf_counter() - query_start
    avg_query_latency_ms = (total_query_time / len(queries_df)) * 1000

    recall_5_values = []
    recall_10_values = []
    mrr_values = []
    ndcg_10_values = []

    for item in query_rankings:
        query_id = item["query_id"]
        ranked_cases = item["ranked_cases"]
        relevant_cases = qrels.get(query_id, {})

        recall_5_values.append(
            recall_at_k(
                ranked_cases=ranked_cases,
                relevant_cases=relevant_cases,
                k=5,
            )
        )

        recall_10_values.append(
            recall_at_k(
                ranked_cases=ranked_cases,
                relevant_cases=relevant_cases,
                k=10,
            )
        )

        mrr_values.append(
            mrr_score(
                ranked_cases=ranked_cases,
                relevant_cases=relevant_cases,
            )
        )

        ndcg_10_values.append(
            ndcg_at_k(
                ranked_cases=ranked_cases,
                relevant_cases=relevant_cases,
                k=10,
            )
        )

    del doc_embeddings
    del query_embeddings
    cleanup_memory()

    return {
        "model_name": spec.name,
        "model_id": spec.model_id,
        "backend": spec.backend,
        "status": "ok",
        "error": "",
        "chunks_indexed": len(chunk_texts),
        "queries": len(queries_df),
        "recall_at_5": round(float(np.mean(recall_5_values)), 4),
        "recall_at_10": round(float(np.mean(recall_10_values)), 4),
        "mrr": round(float(np.mean(mrr_values)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg_10_values)), 4),
        "embedding_build_time_sec": round(float(embedding_build_time), 4),
        "avg_query_latency_ms": round(float(avg_query_latency_ms), 4),
        "index_size_mb": round(float(index_size_mb), 4),
    }


def write_report(rows: list[dict]) -> None:
    EMBEDDING_MODEL_COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model_name",
        "model_id",
        "backend",
        "status",
        "error",
        "chunks_indexed",
        "queries",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "embedding_build_time_sec",
        "avg_query_latency_ms",
        "index_size_mb",
    ]

    with EMBEDDING_MODEL_COMPARISON_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(rows: list[dict]) -> None:
    print("\nGap 15 Domain-Specific Embedding Benchmark Report")
    print("=" * 70)

    for row in rows:
        print(
            f"{row['model_name']} | "
            f"status={row['status']} | "
            f"R@5={row['recall_at_5']} | "
            f"R@10={row['recall_at_10']} | "
            f"MRR={row['mrr']} | "
            f"nDCG@10={row['ndcg_at_10']} | "
            f"latency={row['avg_query_latency_ms']} ms | "
            f"index={row['index_size_mb']} MB"
        )

        if row["status"] != "ok":
            print(f"  Error: {row['error']}")

    print("=" * 70)
    print(f"Report saved to: {EMBEDDING_MODEL_COMPARISON_PATH}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models for legal retrieval."
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Optional model names to run. Example: "
            "--models all-MiniLM-L6-v2 InLegalBERT"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only locally cached HuggingFace models.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunk_texts, metadata = load_chunks()
    queries_df, qrels = load_eval_data()

    specs = get_model_specs()

    if args.models:
        wanted = set(args.models)

        specs = [
            spec for spec in specs
            if spec.name in wanted
        ]

        if not specs:
            raise ValueError(
                f"No matching models found for: {sorted(wanted)}"
            )

    rows = []

    for spec in specs:
        print(f"\nRunning benchmark for: {spec.name}")
        print("-" * 70)

        try:
            row = evaluate_model(
                spec=spec,
                chunk_texts=chunk_texts,
                metadata=metadata,
                queries_df=queries_df,
                qrels=qrels,
                top_k=args.top_k,
                local_only=args.local_only,
                batch_size=args.batch_size,
            )

        except Exception as error:
            row = {
                "model_name": spec.name,
                "model_id": spec.model_id,
                "backend": spec.backend,
                "status": "failed",
                "error": str(error),
                "chunks_indexed": len(chunk_texts),
                "queries": len(queries_df),
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "mrr": 0.0,
                "ndcg_at_10": 0.0,
                "embedding_build_time_sec": 0.0,
                "avg_query_latency_ms": 0.0,
                "index_size_mb": 0.0,
            }

        rows.append(row)
        cleanup_memory()

    write_report(rows)
    print_report(rows)


if __name__ == "__main__":
    main()