from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

REPORTS_DIR = BASE_DIR / "reports"

# Day 1-21 existing paths
RAW_CASES_DIR = RAW_DATA_DIR / "sample_cases"
CASE_METADATA_PATH = RAW_DATA_DIR / "case_metadata.csv"

RAW_EXTRACTED_CASES_PATH = INTERIM_DATA_DIR / "raw_extracted_cases.jsonl"
CLEAN_CASES_PATH = PROCESSED_DATA_DIR / "clean_cases.jsonl"
CASE_CHUNKS_PATH = PROCESSED_DATA_DIR / "case_chunks.jsonl"

STATUTE_MENTIONS_PATH = PROCESSED_DATA_DIR / "statute_mentions.csv"
CASE_CITATIONS_PATH = PROCESSED_DATA_DIR / "case_citations.csv"
NORMALIZED_CITATIONS_PATH = PROCESSED_DATA_DIR / "normalized_citations.csv"

NODES_PATH = PROCESSED_DATA_DIR / "nodes.csv"
EDGES_PATH = PROCESSED_DATA_DIR / "edges.csv"
GRAPH_PATH = PROCESSED_DATA_DIR / "citation_graph.graphml"

BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
EMBEDDINGS_PATH = INDEX_DIR / "chunk_embeddings.npy"

# Production-level ingestion paths
CRAWLERS_DIR = BASE_DIR / "src" / "ingestion" / "crawlers"

MANIFEST_PATH = RAW_DATA_DIR / "manifest.csv"
DOWNLOAD_LOGS_PATH = RAW_DATA_DIR / "download_logs.csv"
FAILED_DOWNLOADS_PATH = RAW_DATA_DIR / "failed_downloads.csv"

RAW_STATUTES_DIR = RAW_DATA_DIR / "statutes"

CORPUS_VERSIONS_DIR = DATA_DIR / "corpus_versions"

FAILED_DOWNLOADS_CLASSIFIED_PATH = RAW_CASES_DIR / "failed_downloads_classfied.csv"

PDF_INTEGRITY_REPORT_PATH = REPORTS_DIR / "pdf_integrity_report.csv"
FAILED_DOWNLOADS_CLASSIFIED_PATH = RAW_DATA_DIR / "failed_downloads_classified.csv"

DOWNLOAD_STATE_PATH = RAW_DATA_DIR / "download_state.json"
DOWNLOAD_CHECKPOINTS_PATH = RAW_DATA_DIR / "download_checkpoints.csv"

SOURCE_SCHEMA_CHANGES_PATH = REPORTS_DIR / "source_schema_changes.md"
CRAWL_RATE_REPORT_PATH = REPORTS_DIR / "crawl_rate_report.csv"

SEED_URLS_PATH = RAW_DATA_DIR / "seed_urls.csv"

DUPLICATE_DOWNLOADS_PATH = RAW_CASES_DIR / "duplicate_downloads.csv"
DEDUPED_MANIFEST_PATH = (
    RAW_DATA_DIR / "manifest_deduped.csv"
)

FILE_HASH_INDEX_PATH = RAW_CASES_DIR / "file_hash_index.csv"

DOWNLOAD_STATE_PATH = RAW_DATA_DIR / "download_state.json"

PARTIAL_DOWNLOADS_DIR = RAW_DATA_DIR / "partial_downloads"
DOWNLOAD_CHECKPOINTS_PATH = RAW_DATA_DIR / "download_checkpoints.csv"

CRAWL_RATE_REPORT_PATH = REPORTS_DIR / "crawl_rate_report.csv"

CRAWL_COMPLIANCE_REPORT_PATH = REPORTS_DIR / "crawl_compliance_report.md"

SOURCE_SCHEMA_CHANGES_PATH = REPORTS_DIR / "source_schema_changes.md"

OCR_TEXT_DIR = PROCESSED_DATA_DIR / "ocr_text"
OCR_QUALITY_REPORT_PATH = REPORTS_DIR / "ocr_quality_report.csv"
OCR_EXTRACTED_TEXT_PATH = INTERIM_DATA_DIR / "ocr_extracted_cases.jsonl"

ADVANCED_CLEAN_CASES_PATH = PROCESSED_DATA_DIR / "advanced_clean_cases.jsonl"
SECTION_SEGMENTS_PATH = PROCESSED_DATA_DIR / "section_segments.jsonl"
RHETORICAL_ROLE_SEGMENTS_PATH = PROCESSED_DATA_DIR / "rhetorical_role_segments.jsonl"
PII_MASKED_CASES_PATH = PROCESSED_DATA_DIR / "pii_masked_cases.jsonl"
ADVANCED_PREPROCESSING_REPORT_PATH = REPORTS_DIR / "advanced_preprocessing_report.csv"

ROLE_BASED_CHUNKS_PATH = PROCESSED_DATA_DIR / "role_based_chunks.jsonl"
ROLE_CHUNKING_REPORT_PATH = REPORTS_DIR / "role_chunking_report.csv"


CASE_OUTCOMES_PATH = PROCESSED_DATA_DIR / "case_outcomes.csv"

LEGAL_CONCEPTS_PATH = PROCESSED_DATA_DIR / "legal_concepts.csv"

RAW_STATUTES_DIR = RAW_DATA_DIR / "statutes"
STATUTES_JSONL_PATH = PROCESSED_DATA_DIR / "statutes.jsonl"
STATUTE_SECTIONS_PATH = PROCESSED_DATA_DIR / "statute_sections.csv"
STATUTE_LINKS_PATH = PROCESSED_DATA_DIR / "statute_links.csv"

CITATION_TREATMENTS_PATH = PROCESSED_DATA_DIR / "citation_treatments.csv"
ADVANCED_CASE_CITATIONS_PATH = PROCESSED_DATA_DIR / "advanced_case_citations.csv"
CITATION_CONTEXTS_PATH = PROCESSED_DATA_DIR / "citation_contexts.csv"

LEGAL_KG_NODES_PATH = PROCESSED_DATA_DIR / "legal_kg_nodes.csv"
LEGAL_KG_EDGES_PATH = PROCESSED_DATA_DIR / "legal_kg_edges.csv"
LEGAL_KG_GRAPHML_PATH = PROCESSED_DATA_DIR / "legal_knowledge_graph.graphml"
NEO4J_NODES_EXPORT_PATH = PROCESSED_DATA_DIR / "neo4j_nodes.csv"
NEO4J_EDGES_EXPORT_PATH = PROCESSED_DATA_DIR / "neo4j_edges.csv"

QUERY_ANALYSIS_REPORT_PATH = REPORTS_DIR / "query_analysis_report.csv"
QUERY_CONCEPT_EMBEDDINGS_PATH = INDEX_DIR / "query_concept_embeddings.npy"
QUERY_CONCEPT_LABELS_PATH = INDEX_DIR / "query_concept_labels.json"

GRAPH_AUGMENTED_RESULTS_PATH = REPORTS_DIR / "graph_augmented_retrieval_results.csv"

CITATION_RESOLUTION_PATH = PROCESSED_DATA_DIR / "citation_resolution.csv"

MULTILEVEL_RETRIEVAL_RESULTS_PATH = REPORTS_DIR / "multilevel_retrieval_results.csv"

AUTHORITY_SCORES_PATH = (
    PROCESSED_DATA_DIR / "authority_scores.csv"
)

STATUTE_MENTIONS_PATH = (
    PROCESSED_DATA_DIR / "statute_mentions.csv"

) 

CITATION_TREATMENTS_PATH = PROCESSED_DATA_DIR / "citation_treatments.csv"

CITATION_AWARE_RANKING_PATH = (
    REPORTS_DIR / "citation_aware_ranking_results.csv"
)

HYBRID_SEARCH_RESULTS_PATH = (
    REPORTS_DIR / "hybrid_search_results.csv"
)

RERANKED_RESULTS_PATH = (
    REPORTS_DIR / "reranked_results.csv"
)

EXPLAINABLE_RANKING_PATH = REPORTS_DIR / "explainable_ranking_results.json"
EXPLAINABLE_RANKING_CSV_PATH = REPORTS_DIR / "explainable_ranking_results.csv"

QUERY_UNDERSTANDING_PATH = REPORTS_DIR / "query_understanding_result.json"

LEGAL_EMBEDDING_MODEL_NAME = "law-ai/InLegalBERT"

LEGAL_EMBEDDINGS_PATH = INDEX_DIR / "legal_chunk_embeddings.npy"
LEGAL_EMBEDDING_METADATA_PATH = INDEX_DIR / "legal_chunk_embedding_metadata.pkl"
LEGAL_FAISS_INDEX_PATH = INDEX_DIR / "legal_faiss.index"

KG_REASONING_RESULT_PATH = REPORTS_DIR / "kg_reasoning_result.json"
RAG_ANSWER_PATH = REPORTS_DIR / "rag_answer.json"
CITATION_GROUNDED_ANSWER_PATH = REPORTS_DIR / "citation_grounded_answer.md"

EVIDENCE_PACK_PATH = REPORTS_DIR / "evidence_pack.json"

LEGAL_PROMPT_PATH = REPORTS_DIR / "legal_prompt.txt"

OLLAMA_EXE_PATH = r"C:\Users\ADMIN\AppData\Local\Programs\Ollama\ollama.exe"
OLLAMA_MODEL_NAME = "llama3.2:3b"
LEGAL_MEMO_PATH = REPORTS_DIR / "legal_memo.md"

EVAL_DATA_DIR = DATA_DIR / "eval"

EMBEDDING_BENCHMARK_QUERIES_PATH = (
    EVAL_DATA_DIR / "embedding_benchmark_queries.csv"
)

EMBEDDING_BENCHMARK_QRELS_PATH = (
    EVAL_DATA_DIR / "embedding_benchmark_qrels.csv"
)

EMBEDDING_MODEL_COMPARISON_PATH = (
    REPORTS_DIR / "embedding_model_comparison.csv"
)

CORPUS_REGISTRY_PATH = (
    RAW_DATA_DIR / "corpus_registry.csv"
)

CORPUS_VERSIONS_DIR = DATA_DIR / "corpus_versions"

CORPUS_STATS_PATH = (
    REPORTS_DIR / "corpus_stats.json"
)

CORPUS_SCALE_REPORT_PATH = REPORTS_DIR / "corpus_scale_report.md"

def create_project_directories() -> None:
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        INDEX_DIR,
        RAW_CASES_DIR,
        RAW_STATUTES_DIR,
        REPORTS_DIR,
        CORPUS_VERSIONS_DIR,
        CRAWLERS_DIR,
        PARTIAL_DOWNLOADS_DIR,
        EVAL_DATA_DIR,
        CORPUS_VERSIONS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_directories()
    print("Project directories created successfully.")