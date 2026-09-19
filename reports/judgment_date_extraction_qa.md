# Judgment Date Extraction QA

Version: `1.0.2`

## Important semantic rule

The existing document `year` field is not assumed to be the judgment date year. In this corpus it can represent the case/filing year. Temporal reasoning must use `decision_date`.

## Metrics

- **date_extractor_version**: 1.0.2
- **documents_processed**: 56
- **exact_dates_extracted**: 56
- **missing_dates**: 0
- **document_text_dates**: 48
- **source_url_fallback_dates**: 8
- **needs_review**: 8
- **text_url_agreements**: 48
- **text_url_disagreements**: 0
- **metadata_year_differs_from_decision_year**: 31

## Review queue

### ECOURTS_000003 — Gauhati High Court

- Selected date: `2026-09-15`
- Source: `source_url_filename`
- Label: `SOURCE_URL_FILENAME`
- Confidence: `0.78`
- Reason: `source_url_fallback_requires_review`
- Source URL date: `2026-09-15`

### ECOURTS_000005 — Gauhati High Court

- Selected date: `2026-09-15`
- Source: `source_url_filename`
- Label: `SOURCE_URL_FILENAME`
- Confidence: `0.78`
- Reason: `source_url_fallback_requires_review`
- Source URL date: `2026-09-15`

### ECOURTS_000036 — Indian High Court

- Selected date: `2026-06-08`
- Source: `source_url_filename`
- Label: `SOURCE_URL_FILENAME`
- Confidence: `0.78`
- Reason: `source_url_fallback_requires_review`
- Source URL date: `2026-06-08`

### ECOURTS_000042 — Indian High Court

- Selected date: `2026-04-21`
- Source: `source_url_filename`
- Label: `SOURCE_URL_FILENAME`
- Confidence: `0.78`
- Reason: `source_url_fallback_requires_review`
- Source URL date: `2026-04-21`

### ECOURTS_000046 — Indian High Court

- Selected date: `2026-03-31`
- Source: `source_url_filename`
- Label: `SOURCE_URL_FILENAME`
- Confidence: `0.78`
- Reason: `source_url_fallback_requires_review`
- Source URL date: `2026-03-31`

### ECOURTS_000049 — Indian High Court

- Selected date: `2026-03-17`
- Source: `source_url_filename`
- Label: `SOURCE_URL_FILENAME`
- Confidence: `0.78`
- Reason: `source_url_fallback_requires_review`
- Source URL date: `2026-03-17`

### ECOURTS_000052 — Indian High Court

- Selected date: `2026-02-20`
- Source: `source_url_filename`
- Label: `SOURCE_URL_FILENAME`
- Confidence: `0.78`
- Reason: `source_url_fallback_requires_review`
- Source URL date: `2026-02-20`

### ECOURTS_000064 — Indian High Court

- Selected date: `2025-12-19`
- Source: `source_url_filename`
- Label: `SOURCE_URL_FILENAME`
- Confidence: `0.78`
- Reason: `source_url_fallback_requires_review`
- Source URL date: `2025-12-19`
