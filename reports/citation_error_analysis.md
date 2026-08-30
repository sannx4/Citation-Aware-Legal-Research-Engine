# Day 14 Citation Extraction Error Analysis

## Summary

- True Positives: 3
- False Positives: 3
- False Negatives: 0
- Precision: 0.5000
- Recall: 1.0000
- F1: 0.6667

## False Positives

These are citations predicted by the system but not present in the gold labels.

- CASE_001 | CASE | CASE_REF_ks_puttaswamy_v_union_of_india
- CASE_002 | CASE | CASE_REF_maneka_gandhi_v_union_of_india
- CASE_003 | CASE | CASE_REF_ak_gopalan_v_state_of_madras

## False Negatives

These are citations present in gold labels but missed by the system.

- None

## Interpretation

- High precision means the extractor avoids fake citations.
- High recall means the extractor finds most real citations.
- False positives usually come from regex over-matching.
- False negatives usually come from missing citation patterns.

## Limitation

This evaluation depends on manually created gold labels. If the gold file is incomplete, the metric will be misleading.