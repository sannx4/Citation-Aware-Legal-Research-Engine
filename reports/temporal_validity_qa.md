# Temporal Validity QA

Version: `1.0.1`

## Semantics

This module reports **corpus-observed temporal treatment status**. It does not claim that an authority is globally good law.

Only `COURT` treatment events may change temporal status. Party submissions, quoted authorities and unknown actors are retained as evidence but ignored for status changes.

Temporal event dates come from the dedicated judgment-date extractor whenever available. Generic document `year` values are not used as event dates because they may represent filing or case-number years.

## Metrics

- **temporal_validity_version**: 1.0.1
- **treatment_rows_input**: 1138
- **treatment_events_built**: 1138
- **authorities_with_timelines**: 664
- **exact_dated_events**: 1138
- **year_only_dated_events**: 0
- **undated_events**: 0
- **events_dated_by_judgment_date_extractor**: 1138
- **events_with_document_text_dates**: 1005
- **events_with_source_url_fallback_dates**: 133
- **events_with_reviewable_dates**: 133
- **judicial_treatment_events**: 1
- **positive_judicial_events**: 1
- **status_changing_judicial_events**: 0
- **party_or_nonjudicial_negative_claims_ignored**: 8
- **temporal_leakage_violations**: 0
- **authorities_with_qa_issues**: 0
- **authorities_needing_review**: 3
- **events_followed**: 1
- **events_relied_on**: 65
- **events_approved**: 0
- **events_distinguished**: 7
- **events_doubted**: 0
- **events_disapproved**: 0
- **events_overruled**: 1
- **events_referred_to**: 1064
- **status_no_observed_negative_treatment**: 664
- **status_distinguished**: 0
- **status_doubted**: 0
- **status_disapproved**: 0
- **status_overruled**: 0

## Authorities requiring review

### CASE_2902AF5EBC74 — Gujarat State Road Transp rabhatbhai and another, 1987 others v. Sukhwinder Singh a claimants is with regard to as stated to be 26 years of age a aimants asserted his income to bunal noticed that there was ome and

- Canonical authority resolver marked this authority for review.

### CASE_64E93696A16D — Ex.A-4) and pass Counsel for the Petitioner: K v. RAGHU VEER Counsel for the Respondent Y N ANJANEYACHARYULU The Court made the following order

- Canonical authority resolver marked this authority for review.

### CASE_AFE87DD42607 — Smti Neemita Chowhai Age: Wife of Nabam Takar Hina permanent resident of Village Nampong PO and PS Namsai District Namsai Arunachal Pradesh v. The State of AP represented by the PP of AP Advocate for the Petitioner

- Canonical authority resolver marked this authority for review.

## Corpus-scope warning

Status is inferred only from treatment events observed in the currently indexed corpus. Absence of negative treatment does not prove that an authority is globally good law.
