# Citation-Aware Legal Research Engine for Indian Case Law

## Project Overview

This project is an AI-powered legal research engine for Indian case law. It is designed to ingest court judgments and statutes, clean and preprocess legal text, extract citations, build a citation graph, retrieve relevant precedents, rerank results, and generate evidence-grounded legal summaries.

## Problem Statement

Legal research is difficult because judgments are long, citations are complex, and keyword search often misses semantically relevant precedents. This project solves that problem using retrieval systems, citation graphs, and retrieval-augmented generation.

## Core Architecture

Raw judgments/statutes  
→ text extraction  
→ preprocessing and cleaning  
→ chunking  
→ citation/statute extraction  
→ citation graph construction  
→ BM25 and dense retrieval  
→ reranking  
→ grounded summary generation  
→ FastAPI/Streamlit demo

## Main Modules

- `src/ingestion`: PDF and text loading
- `src/preprocessing`: cleaning, chunking, deduplication
- `src/extraction`: statute and case citation extraction
- `src/graph`: citation graph creation and metrics
- `src/retrieval`: BM25, embeddings, FAISS, hybrid search
- `src/reranking`: cross-encoder reranking
- `src/summarization`: RAG-based answer generation
- `src/api`: FastAPI backend
- `app`: Streamlit frontend
- `reports`: outputs, screenshots, and evaluation notes

## Day 1 Status

Completed:
- Created project folder structure
- Added dependency list
- Added central configuration file
- Added initial README
- Prepared architecture foundation

## Disclaimer

This system is for legal research assistance only. It does not provide legal advice.