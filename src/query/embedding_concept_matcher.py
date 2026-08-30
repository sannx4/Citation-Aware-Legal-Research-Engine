import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    QUERY_CONCEPT_EMBEDDINGS_PATH,
    QUERY_CONCEPT_LABELS_PATH,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


LEGAL_CONCEPT_BANK = {
    "privacy": [
        "right to privacy",
        "private life",
        "personal data",
        "informational privacy",
        "confidential communication",
        "phone privacy",
        "digital privacy",
    ],
    "digital surveillance": [
        "phone monitoring",
        "device search",
        "whatsapp messages",
        "email access",
        "digital tracking",
        "electronic surveillance",
        "government surveillance",
    ],
    "personal liberty": [
        "personal liberty",
        "life and liberty",
        "freedom from arbitrary action",
        "bodily liberty",
        "Article 21 liberty",
    ],
    "search and seizure": [
        "search without warrant",
        "seizure of phone",
        "police search",
        "device inspection",
        "raid",
        "warrantless search",
    ],
    "due process": [
        "fair procedure",
        "procedure established by law",
        "just fair and reasonable procedure",
        "due process",
    ],
    "natural justice": [
        "fair hearing",
        "notice before action",
        "opportunity of hearing",
        "audi alteram partem",
    ],
    "arbitrariness": [
        "arbitrary state action",
        "manifest arbitrariness",
        "unreasonable government action",
    ],
    "reasonable restriction": [
        "reasonable restriction",
        "public order restriction",
        "restriction on rights",
    ],
    "mens rea": [
        "criminal intent",
        "guilty mind",
        "intention to commit offence",
        "knowledge of offence",
    ],
    "burden of proof": [
        "burden of proof",
        "onus of proof",
        "prove beyond reasonable doubt",
        "standard of proof",
    ],
}


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def flatten_concept_bank() -> tuple[list[str], list[str]]:
    labels = []
    texts = []

    for concept, examples in LEGAL_CONCEPT_BANK.items():
        for example in examples:
            labels.append(concept)
            texts.append(example)

    return labels, texts


def build_concept_index() -> None:
    model = load_model()
    labels, texts = flatten_concept_bank()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    QUERY_CONCEPT_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(QUERY_CONCEPT_EMBEDDINGS_PATH, embeddings)

    with QUERY_CONCEPT_LABELS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "labels": labels,
                "texts": texts,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nQuery Concept Embedding Index Built")
    print("=" * 70)
    print(f"Concept examples indexed: {len(texts)}")
    print(f"Embeddings saved to: {QUERY_CONCEPT_EMBEDDINGS_PATH}")
    print(f"Labels saved to: {QUERY_CONCEPT_LABELS_PATH}")
    print("=" * 70)


def load_concept_index() -> tuple[np.ndarray, list[str], list[str]]:
    if not QUERY_CONCEPT_EMBEDDINGS_PATH.exists() or not QUERY_CONCEPT_LABELS_PATH.exists():
        build_concept_index()

    embeddings = np.load(QUERY_CONCEPT_EMBEDDINGS_PATH)

    with QUERY_CONCEPT_LABELS_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return embeddings, payload["labels"], payload["texts"]


def cosine_scores(query_embedding: np.ndarray, concept_embeddings: np.ndarray) -> np.ndarray:
    return np.dot(concept_embeddings, query_embedding.T).reshape(-1)


def find_embedding_concepts(
    query: str,
    threshold: float = 0.35,
    top_k: int = 5,
) -> list[dict]:
    model = load_model()
    concept_embeddings, labels, texts = load_concept_index()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = cosine_scores(query_embedding, concept_embeddings)

    rows = []

    for label, text, score in zip(labels, texts, scores):
        if float(score) >= threshold:
            rows.append(
                {
                    "concept": label,
                    "matched_example": text,
                    "score": round(float(score), 4),
                }
            )

    rows.sort(key=lambda row: row["score"], reverse=True)

    best_by_concept = {}

    for row in rows:
        concept = row["concept"]

        if concept not in best_by_concept:
            best_by_concept[concept] = row

    return list(best_by_concept.values())[:top_k]


if __name__ == "__main__":
    build_concept_index()

    test_query = "Can government read my WhatsApp messages?"

    print("\nTest Query")
    print("=" * 70)
    print(test_query)

    results = find_embedding_concepts(test_query)

    for row in results:
        print(
            f"{row['concept']} | "
            f"score={row['score']} | "
            f"matched={row['matched_example']}"
        )
