# Day 19 Retrieval Evaluation Report

## What was evaluated

This evaluation compares BM25 keyword search, FAISS semantic search, and Hybrid retrieval.

## Metrics

- Recall@k checks whether the correct case appears in the top-k results.
- MRR checks how early the first correct result appears.

## Summary

- BM25: Recall@1=0.7917, Recall@3=1.0, Recall@5=1.0, MRR=0.9375
- FAISS: Recall@1=0.7917, Recall@3=1.0, Recall@5=1.0, MRR=0.9375
- HYBRID: Recall@1=0.7917, Recall@3=1.0, Recall@5=1.0, MRR=0.9375

## Limitation

This evaluation uses a small manually labeled query set. Larger evaluation data is needed for reliable conclusions.