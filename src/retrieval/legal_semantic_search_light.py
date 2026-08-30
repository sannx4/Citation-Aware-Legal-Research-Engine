import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    LEGAL_EMBEDDING_MODEL_NAME,
    LEGAL_EMBEDDINGS_PATH,
    LEGAL_EMBEDDING_METADATA_PATH,
)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask, dim=1) / torch.clamp(
        input_mask.sum(dim=1),
        min=1e-9,
    )


def encode_query(query: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        LEGAL_EMBEDDING_MODEL_NAME,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        LEGAL_EMBEDDING_MODEL_NAME,
        local_files_only=True,
    )

    model.eval()

    with torch.no_grad():
        encoded = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        output = model(**encoded)
        embedding = mean_pooling(output, encoded["attention_mask"])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy().astype("float32")[0]


def load_metadata():
    with LEGAL_EMBEDDING_METADATA_PATH.open("rb") as file:
        return pickle.load(file)


def search(query: str, top_k: int):
    embeddings = np.load(LEGAL_EMBEDDINGS_PATH).astype("float32")
    metadata = load_metadata()

    query_vector = encode_query(query)

    scores = embeddings @ query_vector

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        item = metadata[idx]

        results.append(
            {
                "rank": rank,
                "score": round(float(scores[idx]), 4),
                "case_id": item["case_id"],
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "court": item["court"],
                "year": item["year"],
                "snippet": " ".join(item["chunk_text"].split())[:300],
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, args.top_k)

    print("\nPhase 3 Lightweight LegalBERT Semantic Search")
    print("=" * 70)

    for row in results:
        print(
            f"\nRank {row['rank']} | score={row['score']} | "
            f"{row['case_id']} | {row['title']}"
        )
        print(row["snippet"])


if __name__ == "__main__":
    main()