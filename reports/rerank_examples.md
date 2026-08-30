# Day 20 Reranking Report

Query: `Article 21 privacy`

## What this report shows

Hybrid search first retrieves candidate chunks. The cross-encoder then reranks those chunks by reading the query and chunk together.

## Before reranking

### Hybrid Rank 1: K.S. Puttaswamy v. Union of India

- Case ID: `CASE_001`
- Chunk ID: `CASE_001_CHUNK_0001`
- Hybrid score: `1.0`

### Hybrid Rank 2: Maneka Gandhi v. Union of India

- Case ID: `CASE_002`
- Chunk ID: `CASE_002_CHUNK_0001`
- Hybrid score: `0.1548`

### Hybrid Rank 3: A.K. Gopalan v. State of Madras

- Case ID: `CASE_003`
- Chunk ID: `CASE_003_CHUNK_0001`
- Hybrid score: `0.0123`


## After reranking

### Rerank 1: K.S. Puttaswamy v. Union of India

- Case ID: `CASE_001`
- Chunk ID: `CASE_001_CHUNK_0001`
- Hybrid score: `1.0`
- Rerank score: `4.8376`

Snippet: K.S. Puttaswamy v. Union of India Court: Supreme Court of India Year: 2017 Facts: The case concerned whether privacy is protected as a fundamental right. Issues: Whether the right to privacy is protected under Article 21 of the Constitution of India....

### Rerank 2: Maneka Gandhi v. Union of India

- Case ID: `CASE_002`
- Chunk ID: `CASE_002_CHUNK_0001`
- Hybrid score: `0.1548`
- Rerank score: `-1.1366`

Snippet: Maneka Gandhi v. Union of India Court: Supreme Court of India Year: 1978 Facts: The petitioner challenged the impounding of her passport. Issues: Whether procedure established by law under Article 21 must be fair, just, and reasonable. Arguments: The...

### Rerank 3: A.K. Gopalan v. State of Madras

- Case ID: `CASE_003`
- Chunk ID: `CASE_003_CHUNK_0001`
- Hybrid score: `0.0123`
- Rerank score: `-11.2661`

Snippet: A.K. Gopalan v. State of Madras Court: Supreme Court of India Year: 1950 Facts: The petitioner challenged preventive detention. Issues: Whether preventive detention violated fundamental rights. Arguments: The petitioner argued that detention affected...

## Limitation

Cross-encoder reranking is slower than BM25 or FAISS because it scores each query-chunk pair separately.