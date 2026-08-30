# Day 18 Hybrid Search Comparison

Query: `Article 21 privacy`

## What this report shows

This report compares combined BM25 keyword retrieval and FAISS semantic retrieval.
BM25 helps with exact legal terms. FAISS helps with meaning-based search.

## Results

### Rank 1: K.S. Puttaswamy v. Union of India

- Case ID: `CASE_001`
- Chunk ID: `CASE_001_CHUNK_0001`
- BM25 score: `1.0`
- FAISS score: `1.0`
- Hybrid score: `1.0`

Snippet: K.S. Puttaswamy v. Union of India Court: Supreme Court of India Year: 2017 Facts: The case concerned whether privacy is protected as a fundamental right. Issues: Whether the right to privacy is protected under Article 21 of the Constitution of India....

### Rank 2: Maneka Gandhi v. Union of India

- Case ID: `CASE_002`
- Chunk ID: `CASE_002_CHUNK_0001`
- BM25 score: `0.0`
- FAISS score: `0.3097`
- Hybrid score: `0.1548`

Snippet: Maneka Gandhi v. Union of India Court: Supreme Court of India Year: 1978 Facts: The petitioner challenged the impounding of her passport. Issues: Whether procedure established by law under Article 21 must be fair, just, and reasonable. Arguments: The...

### Rank 3: A.K. Gopalan v. State of Madras

- Case ID: `CASE_003`
- Chunk ID: `CASE_003_CHUNK_0001`
- BM25 score: `0.0246`
- FAISS score: `0.0`
- Hybrid score: `0.0123`

Snippet: A.K. Gopalan v. State of Madras Court: Supreme Court of India Year: 1950 Facts: The petitioner challenged preventive detention. Issues: Whether preventive detention violated fundamental rights. Arguments: The petitioner argued that detention affected...

## Limitation

The hybrid score depends on the alpha weight. A small corpus may produce unstable ranking.