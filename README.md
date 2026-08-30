# Citation-Aware Legal Research Engine for Indian Case Law

An AI-powered Indian legal research engine that retrieves, ranks, explains, and generates citation-grounded legal research answers using legal document ingestion, citation extraction, semantic search, citation graphs, authority ranking, and RAG-based legal memo generation.

---

## Project Overview

The **Citation-Aware Legal Research Engine** is a LegalTech AI system designed for Indian case law research. It processes court judgments, extracts legal citations and statute references, builds citation relationships between cases, retrieves relevant precedents, ranks them using legal authority signals, and generates grounded legal research memos using retrieved evidence.

Traditional keyword search is not enough for legal research because judgments are long, legal language is complex, and important precedents may not contain the exact search words. This project combines **BM25 keyword retrieval**, **semantic vector search**, **citation graph reasoning**, **authority ranking**, and **RAG-based legal memo generation** to make legal search more accurate, explainable, and research-ready.

---

## One-Line Summary

An AI-powered legal research engine that retrieves, ranks, explains, and summarizes Indian legal judgments using semantic search, citation graphs, legal authority ranking, and retrieval-augmented generation.

---

## Key Features

* **Legal document ingestion pipeline** for Indian court judgments and statute-related sources.
* **PDF and text extraction** with support for raw case files, metadata mapping, and extraction reports.
* **Text preprocessing pipeline** for cleaning, chunking, deduplication, and preparing legal documents for retrieval.
* **Statute and Article extraction** for references such as Article 21, Article 14, Section 302, and other legal provisions.
* **Case citation extraction** for detecting legal references like `K.S. Puttaswamy v. Union of India`.
* **Citation normalization and graph construction** to represent relationships between cases.
* **BM25 keyword search** for exact legal term matching.
* **Semantic embedding search** using transformer-based embeddings.
* **FAISS/vector search** for fast similarity-based legal retrieval.
* **Hybrid retrieval** combining sparse keyword search and dense semantic search.
* **Citation-neighborhood expansion** to discover related cited/citing cases.
* **Legal authority ranking** using court hierarchy, bench strength, PageRank, recency, and precedent validity.
* **Citation-aware ranking** combining hybrid retrieval, reranking, authority score, statute match, and recency validity.
* **Explainable ranking** showing why a case was selected.
* **Legal query understanding** for rewriting natural language queries into legally meaningful concepts.
* **Evidence pack generation** for grounded answer generation.
* **RAG legal prompt builder** for constructing source-grounded prompts.
* **Local LLM legal memo generation** using Ollama and Llama 3.2.
* **Embedding model benchmarking** using Recall@5, Recall@10, MRR, nDCG, latency, and index size.
* **Production ingestion utilities** including PDF integrity checks, duplicate detection, hash indexing, resumable downloads, checkpointed downloads, error classification, and crawler rate limiting.

---

## Why This Project Matters

Legal research requires more than ordinary text search. Lawyers, students, and researchers must understand:

* which cases are relevant,
* which cases are legally authoritative,
* whether a case cites or relies on another case,
* which statutes or constitutional articles are involved,
* whether the answer is grounded in actual evidence,
* and why a result was ranked highly.

This project is built to solve those problems by combining **Legal NLP**, **Information Retrieval**, **Graph Analytics**, and **LLM-based RAG** into one end-to-end legal research system.

---

## System Architecture

```text
Raw Judgments / PDFs / Statutes
        ↓
Corpus Manifest + Metadata
        ↓
PDF/Text Extraction
        ↓
Text Cleaning
        ↓
Chunking
        ↓
Statute / Article Extraction
        ↓
Case Citation Extraction
        ↓
Citation Normalization
        ↓
Citation Graph Construction
        ↓
BM25 Search
        ↓
Semantic Embeddings
        ↓
FAISS / Vector Search
        ↓
Hybrid Retrieval
        ↓
Citation-Neighborhood Expansion
        ↓
Legal Authority Ranking
        ↓
Citation-Aware Ranking
        ↓
Explainable Ranking
        ↓
Evidence Pack Builder
        ↓
Legal Prompt Builder
        ↓
Local LLM / Ollama
        ↓
Citation-Grounded Legal Memo
```

---

## Repository Structure

```text
legal-research-engine/
│
├── app/
│   └── Streamlit / frontend demo files
│
├── data/
│   ├── raw/
│   │   ├── sample_cases/
│   │   ├── manifest.csv
│   │   ├── seed_urls.csv
│   │   └── case_metadata.csv
│   │
│   ├── interim/
│   │   └── raw extracted legal text
│   │
│   ├── processed/
│   │   ├── clean_cases.jsonl
│   │   ├── case_chunks.jsonl
│   │   ├── statute_mentions.csv
│   │   ├── case_citations.csv
│   │   ├── normalized_citations.csv
│   │   ├── nodes.csv
│   │   ├── edges.csv
│   │   ├── citation_graph.graphml
│   │   └── legal knowledge graph outputs
│   │
│   └── indexes/
│       ├── BM25 indexes
│       ├── FAISS indexes
│       └── embedding files
│
├── reports/
│   ├── retrieval reports
│   ├── ranking reports
│   ├── graph reports
│   ├── citation evaluation reports
│   ├── evidence packs
│   ├── legal prompts
│   └── generated legal memos
│
├── src/
│   ├── ingestion/
│   │   ├── crawlers/
│   │   ├── pdf_extract.py
│   │   ├── pdf_integrity_check.py
│   │   ├── duplicate_detector.py
│   │   ├── hash_index.py
│   │   ├── error_classifier.py
│   │   ├── resume_downloads.py
│   │   ├── checkpoint_downloader.py
│   │   └── rate_limiter.py
│   │
│   ├── preprocessing/
│   │   ├── clean_text.py
│   │   ├── chunk_cases.py
│   │   └── deduplicate.py
│   │
│   ├── extraction/
│   │   ├── patterns.py
│   │   ├── extract_statutes.py
│   │   ├── extract_case_citations.py
│   │   ├── normalize_citations.py
│   │   ├── citation_treatment.py
│   │   └── legal_concepts.py
│   │
│   ├── graph/
│   │   ├── build_edges.py
│   │   ├── build_graph.py
│   │   ├── visualize_graph.py
│   │   └── legal_kg_builder.py
│   │
│   ├── retrieval/
│   │   ├── bm25_search.py
│   │   ├── embed_chunks.py
│   │   ├── faiss_index.py
│   │   ├── hybrid_search.py
│   │   ├── legal_semantic_search_light.py
│   │   ├── citation_expansion.py
│   │   ├── graph_augmented_retrieval.py
│   │   └── multilevel_retrieval.py
│   │
│   ├── ranking/
│   │   ├── court_hierarchy.py
│   │   ├── bench_strength.py
│   │   ├── precedent_validity.py
│   │   ├── authority_score.py
│   │   └── citation_aware_ranker.py
│   │
│   ├── explainability/
│   │   └── ranking_explainer.py
│   │
│   ├── query/
│   │   ├── query_understanding.py
│   │   └── kg_llm_reasoner.py
│   │
│   ├── summarization/
│   │   ├── evidence_pack_builder.py
│   │   ├── legal_prompt_builder.py
│   │   ├── rag_answer_generator.py
│   │   ├── citation_grounded_synthesizer.py
│   │   └── legal_memo_generator.py
│   │
│   ├── experiments/
│   │   └── embedding_benchmark.py
│   │
│   ├── evaluation/
│   │   └── retrieval / citation evaluation scripts
│   │
│   └── config.py
│
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

### Core Language

* Python

### Data Handling

* Pandas
* NumPy
* CSV
* JSON
* JSONL
* Pickle

### PDF and Document Processing

* PyMuPDF / Fitz
* pdfplumber
* PDF integrity validation
* File hash validation
* Metadata extraction

### Web Crawling and Ingestion

* Requests
* Playwright
* Human-assisted eCourts crawler
* Manifest-based downloader
* Resume-aware downloader
* Checkpointed downloader
* Rate-limited crawling

### NLP and Embeddings

* Hugging Face Transformers
* SentenceTransformers
* PyTorch
* LegalBERT
* InLegalBERT
* all-MiniLM-L6-v2
* BGE
* E5

### Search and Retrieval

* BM25
* rank-bm25
* FAISS
* Semantic search
* Hybrid retrieval
* Multi-level retrieval
* Citation-neighborhood expansion

### Graph Analytics

* NetworkX
* PageRank
* Centrality scores
* Citation graph
* Legal knowledge graph

### Ranking

* Hybrid ranking
* Citation-aware ranking
* Legal authority ranking
* Court hierarchy scoring
* Bench strength scoring
* Recency scoring
* Precedent validity scoring

### RAG and LLM

* Evidence pack builder
* Legal prompt builder
* Citation-grounded answer generation
* Ollama
* Llama 3.2 3B

### Evaluation

* Precision
* Recall
* F1
* Recall@5
* Recall@10
* MRR
* nDCG
* Latency
* Index size

### Development and Deployment

* Git
* GitHub
* Python virtual environment
* requirements.txt
* FastAPI
* Streamlit
* Docker-ready structure

---

## Current Project Status

The project currently includes a working end-to-end legal research pipeline for a sample Indian legal corpus.

Completed modules include:

* Project architecture and repository setup
* Sample legal corpus setup
* Raw text/PDF extraction
* Text cleaning
* Legal chunking
* Deduplication
* Statute and Article extraction
* Case citation extraction
* Citation normalization
* Citation graph construction
* Citation graph metrics
* Citation evaluation
* BM25 keyword retrieval
* Embedding generation
* FAISS semantic retrieval
* Hybrid search
* LegalBERT/InLegalBERT semantic search
* Graph-augmented retrieval
* Multi-level retrieval fusion
* Legal authority ranking
* Citation-aware ranking
* Explainable ranking
* Query understanding
* KG-style legal reasoning
* Evidence pack generation
* Legal prompt generation
* Ollama-based legal memo generation
* Embedding benchmark module
* Production ingestion hardening modules

---

## Example Query

```text
Can police check my phone without permission?
```

The system processes this query by identifying legal concepts such as:

```text
privacy
personal data
digital surveillance
consent
due process
search and seizure
state action
Article 21
```

Then it retrieves and ranks relevant cases such as:

```text
K.S. Puttaswamy v. Union of India
Maneka Gandhi v. Union of India
A.K. Gopalan v. State of Madras
```

Finally, it generates a citation-grounded legal memo using retrieved evidence.

---

## Example Output

```text
Issue:
Can police check my phone without permission?

Relevant Law:
Article 21 of the Constitution of India, including the right to life,
personal liberty, and privacy.

Analysis:
The strongest retrieved precedent is K.S. Puttaswamy v. Union of India,
where the Supreme Court recognized privacy as a fundamental right under
Article 21. Maneka Gandhi v. Union of India supports the requirement of
fair, just, and reasonable procedure when personal liberty is restricted.

Conclusion:
Based on the retrieved evidence, phone searches without consent or proper
legal procedure may raise Article 21 privacy concerns. However, the system
does not provide legal advice and the answer depends on the exact facts,
lawful authority, and procedural safeguards.
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sannx4/Citation-Aware-Legal-Research-Engine.git
cd Citation-Aware-Legal-Research-Engine
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Create Project Directories

```bash
python src/config.py
```

### Load Sample Cases

```bash
python src/ingestion/load_samples.py
```

### Extract Raw Text

```bash
python src/ingestion/pdf_extract.py --metadata data/raw/case_metadata.csv
```

### Clean Text

```bash
python src/preprocessing/clean_text.py
```

### Chunk Cases

```bash
python src/preprocessing/chunk_cases.py --chunk-size 900 --overlap 120
```

### Extract Statute Mentions

```bash
python src/extraction/extract_statutes.py
```

### Extract Case Citations

```bash
python src/extraction/extract_case_citations.py
```

### Normalize Citations

```bash
python src/extraction/normalize_citations.py
```

### Build Graph Edges

```bash
python src/graph/build_edges.py
```

### Build Citation Graph

```bash
python src/graph/build_graph.py
```

### Run BM25 Search

```bash
python src/retrieval/bm25_search.py "Article 21 privacy"
```

### Generate Embeddings

```bash
python src/retrieval/embed_chunks.py
```

### Build and Search FAISS Index

```bash
python src/retrieval/faiss_index.py "right to privacy" --rebuild
```

### Run Hybrid Search

```bash
python src/retrieval/hybrid_search.py "Article 21 privacy"
```

### Run Legal Authority Ranking

```bash
python src/ranking/authority_score.py
```

### Run Citation-Aware Ranking

```bash
python src/ranking/citation_aware_ranker.py "Article 21 privacy" --top-k 5
```

### Run Explainable Ranking

```bash
python src/explainability/ranking_explainer.py "Article 21 privacy" --top-k 5
```

### Run Query Understanding

```bash
python src/query/query_understanding.py "Can police check my phone without permission?"
```

### Build Evidence Pack

```bash
python src/summarization/evidence_pack_builder.py "Can police check my phone without permission?"
```

### Build Legal Prompt

```bash
python src/summarization/legal_prompt_builder.py
```

### Generate Legal Memo with Ollama

```bash
python src/summarization/legal_memo_generator.py
```

---

## Ollama Setup

This project supports local LLM generation using Ollama.

### Install Ollama

Download and install Ollama from the official website.

### Pull a Lightweight Model

```bash
ollama pull llama3.2:3b
```

### Test the Model

```bash
ollama run llama3.2:3b "Explain Article 21 in simple terms"
```

### Generate Legal Memo

```bash
python src/summarization/legal_memo_generator.py
```

The generated memo is saved in:

```text
reports/legal_memo.md
```

---

## Embedding Benchmark

The project includes an embedding benchmark module to compare generic and legal-domain embedding models.

Models compared:

```text
all-MiniLM-L6-v2
BGE
E5
LegalBERT
InLegalBERT
domain-finetuned model
```

Metrics:

```text
Recall@5
Recall@10
MRR
nDCG
Latency
Index size
```

Run benchmark:

```bash
python src/experiments/embedding_benchmark.py --batch-size 2
```

Run selected models only:

```bash
python src/experiments/embedding_benchmark.py --models all-MiniLM-L6-v2 InLegalBERT --batch-size 2
```

Output:

```text
reports/embedding_model_comparison.csv
```

---

## Production Ingestion Modules

The project also includes production-grade ingestion utilities.

### PDF Integrity Check

Checks whether downloaded PDFs are valid, non-empty, readable, and not HTML/error pages saved as PDF.

```bash
python src/ingestion/pdf_integrity_check.py
```

### Duplicate Detection

Detects duplicate documents using URL, metadata, and file hash.

```bash
python src/ingestion/duplicate_detector.py
```

### Hash Indexing

Tracks file hashes to avoid reprocessing unchanged documents.

```bash
python src/ingestion/hash_index.py
```

### Error Classification

Classifies failed downloads and PDF validation issues.

```bash
python src/ingestion/error_classifier.py
```

### Resume-Aware Downloads

Tracks completed, failed, and pending downloads.

```bash
python src/ingestion/resume_downloads.py
```

### Checkpointed Downloads

Uses `.part` files and atomic rename to prevent corrupted partial files.

```bash
python src/ingestion/checkpoint_downloader.py
```

---

## Core Modules Explained

### 1. Ingestion Module

Collects legal documents, reads metadata, downloads files, validates PDFs, and prepares raw case documents.

### 2. Preprocessing Module

Cleans extracted legal text, removes formatting noise, chunks long judgments, and prepares text for retrieval.

### 3. Extraction Module

Extracts statute references, constitutional articles, case citations, legal concepts, and citation treatments.

### 4. Graph Module

Builds citation graphs and legal knowledge graphs from extracted relationships.

### 5. Retrieval Module

Retrieves relevant cases using BM25, semantic embeddings, FAISS, hybrid search, and graph-augmented retrieval.

### 6. Ranking Module

Ranks retrieved cases using legal relevance, citation authority, statute matching, court hierarchy, and precedent validity.

### 7. Explainability Module

Explains why a result ranked highly, including matched provisions, semantic relevance, court authority, and citation signals.

### 8. Query Understanding Module

Converts user queries into legal concepts, statutes, likely cases, and rewritten legal search queries.

### 9. Summarization / RAG Module

Builds evidence packs, constructs legal prompts, and generates citation-grounded legal memos using local LLMs.

### 10. Evaluation Module

Evaluates retrieval, citation extraction, ranking, and embedding performance using standard IR metrics.

---

## Example Reports Generated

```text
reports/citation_evaluation_report.csv
reports/citation_error_analysis.md
reports/hybrid_search_results.csv
reports/citation_aware_ranking_results.csv
reports/explainable_ranking_results.json
reports/explainable_ranking_results.csv
reports/query_understanding_result.json
reports/kg_reasoning_result.json
reports/evidence_pack.json
reports/legal_prompt.txt
reports/legal_memo.md
reports/embedding_model_comparison.csv
reports/pdf_integrity_report.csv
```

---

## Research Value

This project is stronger than a basic chatbot or CRUD application because it combines:

* Legal document processing
* Information retrieval
* Citation graph reasoning
* Domain-specific embeddings
* Legal authority ranking
* Explainable AI
* Retrieval-augmented generation
* Local LLM inference
* Evaluation metrics
* Production ingestion hardening

It demonstrates practical AI engineering skills across NLP, search systems, graph analytics, RAG, and legal-domain reasoning.

---

## Current Limitations

* The current working corpus is still small compared to a full production legal search system.
* Some modules are prototype-level and need larger-scale validation.
* Legal memo generation depends on the quality of retrieved evidence and the local LLM.
* The system should not be treated as a substitute for a lawyer.
* More cases, statutes, expert-labeled queries, and evaluation benchmarks are needed for research-grade reliability.

---

## Future Improvements

* Scale corpus from sample judgments to 1,000+ and then 10,000+ judgments.
* Add full Supreme Court, High Court, and IndiaCode ingestion.
* Build a production database layer using PostgreSQL or SQLite.
* Add Neo4j support for large legal knowledge graphs.
* Fine-tune a legal bi-encoder using citation pairs.
* Fine-tune a legal cross-encoder reranker.
* Add rhetorical-role segmentation for facts, issues, reasoning, and holding.
* Add statute full-text retrieval.
* Add hallucination detection and claim-evidence verification.
* Add temporal validity and overruling detection.
* Add legal memo quality evaluation.
* Build FastAPI endpoints for search and summarization.
* Build Streamlit/React interface for interactive legal research.

---

## Project Roadmap

```text
Phase 1: Corpus and Preprocessing
- Sample cases
- Metadata
- Extraction
- Cleaning
- Chunking
- Deduplication

Phase 2: Citation Intelligence
- Article extraction
- Section extraction
- Case citation extraction
- Citation normalization
- Citation graph

Phase 3: Search and Retrieval
- BM25
- Embeddings
- FAISS
- Hybrid retrieval
- Graph-augmented retrieval

Phase 4: Ranking and Explainability
- Authority ranking
- Citation-aware ranking
- Explainable search
- Query understanding

Phase 5: RAG and Legal Memo Generation
- Evidence pack
- Legal prompt
- Ollama LLM memo
- Citation-grounded answer generation

Phase 6: Research-Grade Scaling
- Large corpus ingestion
- Legal embedding benchmark
- Bi-encoder training
- Cross-encoder training
- Evaluation and ablation studies
```

---

## Disclaimer

This project is for **legal research assistance only**. It does not provide legal advice. Generated outputs must be verified by qualified legal professionals before use in any legal, academic, or professional setting.

---

## Author

**Sanjay**

Project: **Citation-Aware Legal Research Engine for Indian Case Law**

GitHub: `sannx4`

---

## License

This repository is intended for educational, research, and portfolio purposes. Add a license file before production or public reuse.
