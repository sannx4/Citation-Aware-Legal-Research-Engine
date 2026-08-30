import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    CASE_CHUNKS_PATH,
    LEGAL_EMBEDDING_MODEL_NAME,
    LEGAL_EMBEDDINGS_PATH,
    LEGAL_EMBEDDING_METADATA_PATH,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask, dim=1) / torch.clamp(
        input_mask.sum(dim=1),
        min=1e-9,
    )


def encode_texts(texts: list[str], batch_size: int = 8) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(LEGAL_EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(LEGAL_EMBEDDING_MODEL_NAME)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            output = model(**encoded)
            embeddings = mean_pooling(output, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype("float32")


def main() -> None:
    chunks = read_jsonl(CASE_CHUNKS_PATH)

    texts = []
    metadata = []

    for chunk in chunks:
        text = chunk.get("chunk_text", "")
        texts.append(text)

        metadata.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "case_id": chunk.get("case_id", ""),
                "title": chunk.get("title", ""),
                "court": chunk.get("court", ""),
                "year": chunk.get("year", ""),
                "chunk_text": text,
            }
        )

    embeddings = encode_texts(texts)

    LEGAL_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(LEGAL_EMBEDDINGS_PATH, embeddings)

    with LEGAL_EMBEDDING_METADATA_PATH.open("wb") as file:
        pickle.dump(metadata, file)

    print("\nPhase 3 LegalBERT Embedding Report")
    print("=" * 70)
    print(f"Model: {LEGAL_EMBEDDING_MODEL_NAME}")
    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {LEGAL_EMBEDDINGS_PATH}")
    print(f"Metadata saved to: {LEGAL_EMBEDDING_METADATA_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()