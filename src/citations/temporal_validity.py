from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RESOLVED_TREATMENTS_PATH = (
    PROCESSED_DIR / "citation_treatments_resolved.jsonl"
)

AUTHORITY_REGISTRY_PATH = (
    PROCESSED_DIR / "canonical_citation_registry.jsonl"
)

JUDGMENT_DATES_PATH = (
    PROCESSED_DIR / "judgment_dates.jsonl"
)

RAW_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "raw" / "manifest_deduped.csv"
)

DOCUMENTS_DIR = (
    PROCESSED_DIR / "documents"
)

TIMELINE_OUTPUT_PATH = (
    PROCESSED_DIR / "authority_treatment_timeline.jsonl"
)

STATUS_OUTPUT_PATH = (
    PROCESSED_DIR / "authority_temporal_status.jsonl"
)

REPORT_PATH = (
    REPORTS_DIR / "temporal_validity_report.csv"
)

QA_PATH = (
    REPORTS_DIR / "temporal_validity_qa.md"
)


TEMPORAL_VALIDITY_VERSION = "1.0.1"


# ============================================================
# Semantics
# ============================================================

POSITIVE_TREATMENTS = {
    "RELIED_ON",
    "FOLLOWED",
    "APPROVED",
}

NEGATIVE_TREATMENTS = {
    "DISTINGUISHED",
    "DOUBTED",
    "DISAPPROVED",
    "OVERRULED",
}

STRONG_TREATMENTS = (
    POSITIVE_TREATMENTS
    | NEGATIVE_TREATMENTS
)

NEUTRAL_TREATMENTS = {
    "REFERRED_TO",
}

# IMPORTANT:
# This is NOT an assertion that the authority is globally "good law".
# It is only a corpus-observed temporal state.
TREATMENT_TO_STATUS = {
    "DISTINGUISHED": "DISTINGUISHED",
    "DOUBTED": "DOUBTED",
    "DISAPPROVED": "DISAPPROVED",
    "OVERRULED": "OVERRULED",
}

STATUS_SEVERITY = {
    "NO_OBSERVED_NEGATIVE_TREATMENT": 0,
    "DISTINGUISHED": 1,
    "DOUBTED": 2,
    "DISAPPROVED": 3,
    "OVERRULED": 4,
}

DATE_FIELDS = (
    "decision_date",
    "judgment_date",
    "date_of_judgment",
    "judgement_date",
    "date_of_judgement",
    "decided_on",
    "decision_on",
    "order_date",
    "date",
)

YEAR_FIELDS = (
    "year",
    "decision_year",
    "judgment_year",
    "judgement_year",
)

DOCUMENT_ID_FIELDS = (
    "document_id",
    "doc_id",
    "id",
    "case_id",
)

CORPUS_SCOPE_WARNING = (
    "Status is inferred only from treatment events observed in the "
    "currently indexed corpus. Absence of negative treatment does not "
    "prove that an authority is globally good law."
)


# ============================================================
# Generic helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSONL file not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )

            rows.append(value)

    return rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_non_empty(
    data: dict[str, Any],
    fields: tuple[str, ...],
) -> Any:
    for field in fields:
        value = data.get(field)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return None


# ============================================================
# Date parsing
# ============================================================

def year_to_conservative_date(year: int) -> date | None:
    """
    A year-only treatment is conservatively made effective on
    31 December of that year.

    This avoids leaking an event into earlier dates within the
    same year when the exact judgment date is unavailable.
    """

    if year < 1800 or year > 2200:
        return None

    return date(year, 12, 31)


def parse_date_value(
    value: Any,
) -> tuple[date | None, str, str | None]:
    """
    Returns:
        parsed/effective date
        precision: EXACT | YEAR | MISSING | INVALID
        original string
    """

    if value is None:
        return None, "MISSING", None

    if isinstance(value, datetime):
        return value.date(), "EXACT", value.isoformat()

    if isinstance(value, date):
        return value, "EXACT", value.isoformat()

    if isinstance(value, int):
        parsed = year_to_conservative_date(value)

        if parsed is not None:
            return parsed, "YEAR", str(value)

        return None, "INVALID", str(value)

    raw = normalize_spaces(str(value))

    if not raw:
        return None, "MISSING", None

    if re.fullmatch(r"\d{4}", raw):
        parsed = year_to_conservative_date(int(raw))

        if parsed is not None:
            return parsed, "YEAR", raw

        return None, "INVALID", raw

    # ISO date / datetime.
    iso_candidate = raw.replace("Z", "+00:00")

    try:
        parsed_dt = datetime.fromisoformat(iso_candidate)
        return parsed_dt.date(), "EXACT", raw
    except ValueError:
        pass

    try:
        parsed_date = date.fromisoformat(raw)
        return parsed_date, "EXACT", raw
    except ValueError:
        pass

    formats = (
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
    )

    for fmt in formats:
        try:
            return (
                datetime.strptime(raw, fmt).date(),
                "EXACT",
                raw,
            )
        except ValueError:
            continue

    # Extract a date embedded inside a larger metadata string.
    embedded_patterns = (
        (
            r"\b(\d{4}-\d{2}-\d{2})\b",
            "%Y-%m-%d",
        ),
        (
            r"\b(\d{2}/\d{2}/\d{4})\b",
            "%d/%m/%Y",
        ),
        (
            r"\b(\d{2}-\d{2}-\d{4})\b",
            "%d-%m-%Y",
        ),
    )

    for pattern, fmt in embedded_patterns:
        match = re.search(pattern, raw)

        if not match:
            continue

        try:
            return (
                datetime.strptime(
                    match.group(1),
                    fmt,
                ).date(),
                "EXACT",
                raw,
            )
        except ValueError:
            continue

    return None, "INVALID", raw


# ============================================================
# Document metadata index
# ============================================================

def merge_metadata_record(
    index: dict[str, dict[str, Any]],
    document_id: str,
    record: dict[str, Any],
) -> None:
    document_id = normalize_spaces(document_id)

    if not document_id:
        return

    existing = index.setdefault(
        document_id,
        {},
    )

    # Prefer already-known non-empty values, except an exact date
    # is allowed to replace a year-only date.
    for key, value in record.items():
        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        if key not in existing:
            existing[key] = value
            continue

        if key in DATE_FIELDS:
            old_date, old_precision, _ = parse_date_value(
                existing[key]
            )
            new_date, new_precision, _ = parse_date_value(
                value
            )

            if (
                new_date is not None
                and new_precision == "EXACT"
                and old_precision != "EXACT"
            ):
                existing[key] = value


def load_manifest_metadata() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    if not RAW_MANIFEST_PATH.exists():
        return index

    with RAW_MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            document_id = first_non_empty(
                row,
                DOCUMENT_ID_FIELDS,
            )

            if document_id is None:
                continue

            merge_metadata_record(
                index,
                str(document_id),
                dict(row),
            )

    return index


def load_document_metadata() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    if not DOCUMENTS_DIR.exists():
        return index

    for path in DOCUMENTS_DIR.rglob("*.json"):
        try:
            value = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            continue

        if not isinstance(value, dict):
            continue

        document_id = first_non_empty(
            value,
            DOCUMENT_ID_FIELDS,
        )

        if document_id is None:
            document_id = path.stem

        merge_metadata_record(
            index,
            str(document_id),
            value,
        )

    for path in DOCUMENTS_DIR.rglob("*.jsonl"):
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                for line in file:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(value, dict):
                        continue

                    document_id = first_non_empty(
                        value,
                        DOCUMENT_ID_FIELDS,
                    )

                    if document_id is None:
                        continue

                    merge_metadata_record(
                        index,
                        str(document_id),
                        value,
                    )
        except (OSError, UnicodeDecodeError):
            continue

    return index


def build_document_metadata_index() -> (
    dict[str, dict[str, Any]]
):
    index = load_manifest_metadata()

    document_rows = load_document_metadata()

    for document_id, record in document_rows.items():
        merge_metadata_record(
            index,
            document_id,
            record,
        )

    return index


# ============================================================
# Extracted judgment-date index
# ============================================================

def load_judgment_date_index(
    path: Path = JUDGMENT_DATES_PATH,
) -> dict[str, dict[str, Any]]:
    """
    Load the dedicated judgment-date extraction output.

    The generic document `year` field is intentionally NOT used
    as a temporal event date because it may represent the filing/
    case-number year rather than the decision year.
    """

    rows = read_jsonl(path)

    index: dict[str, dict[str, Any]] = {}

    for row in rows:
        document_id = normalize_spaces(
            str(
                row.get(
                    "document_id",
                    "",
                )
            )
        )

        if not document_id:
            continue

        index[document_id] = row

    return index


# ============================================================
# Event-date resolution
# ============================================================

def resolve_event_date(
    row: dict[str, Any],
    document_metadata: dict[str, dict[str, Any]],
    judgment_dates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Resolve the date of the CITING judgment.

    Priority:
      1. Dedicated judgment_date_extractor output.
      2. Any exact date already present in the treatment row.
      3. Any exact date explicitly present in document metadata.

    v1.0.1 deliberately removes generic year-only fallback because
    the corpus `year` field may be the filing/case-number year.
    """

    document_id = normalize_spaces(
        str(
            row.get(
                "document_id",
                "",
            )
        )
    )

    extracted = judgment_dates.get(
        document_id,
        {},
    )

    extracted_value = extracted.get(
        "decision_date"
    )

    if extracted_value not in (
        None,
        "",
    ):
        parsed, precision, raw = parse_date_value(
            extracted_value
        )

        if (
            parsed is not None
            and precision == "EXACT"
        ):
            return {
                "event_date": (
                    parsed.isoformat()
                ),

                "event_date_precision": (
                    "EXACT"
                ),

                "event_date_source": (
                    "judgment_date_extractor"
                ),

                "event_date_raw": (
                    extracted.get(
                        "raw_date"
                    )
                    or raw
                ),

                "event_date_conservative": (
                    False
                ),

                "event_date_confidence": (
                    safe_float(
                        extracted.get(
                            "confidence"
                        ),
                        0.0,
                    )
                ),

                "event_date_needs_review": (
                    bool(
                        extracted.get(
                            "needs_review",
                            False,
                        )
                    )
                ),

                "event_date_evidence_source": (
                    extracted.get(
                        "date_source"
                    )
                ),

                "event_date_label": (
                    extracted.get(
                        "date_label"
                    )
                ),

                "event_date_selection_reason": (
                    extracted.get(
                        "selection_reason"
                    )
                ),

                "event_date_extractor_version": (
                    extracted.get(
                        "date_extractor_version"
                    )
                ),

                "date_parse_issues": [],
            }

    metadata = document_metadata.get(
        document_id,
        {},
    )

    candidates: list[
        tuple[str, Any]
    ] = []

    for field in DATE_FIELDS:
        if row.get(field) not in (
            None,
            "",
        ):
            candidates.append(
                (
                    f"treatment.{field}",
                    row.get(field),
                )
            )

    for field in DATE_FIELDS:
        if metadata.get(field) not in (
            None,
            "",
        ):
            candidates.append(
                (
                    f"metadata.{field}",
                    metadata.get(field),
                )
            )

    invalid_values: list[str] = []

    for source, value in candidates:
        parsed, precision, raw = parse_date_value(
            value
        )

        if parsed is None:
            if (
                precision == "INVALID"
                and raw
            ):
                invalid_values.append(
                    f"{source}={raw}"
                )

            continue

        # Exact only. Never reinterpret a bare year as a date.
        if precision == "EXACT":
            return {
                "event_date": (
                    parsed.isoformat()
                ),

                "event_date_precision": (
                    "EXACT"
                ),

                "event_date_source": (
                    source
                ),

                "event_date_raw": (
                    raw
                ),

                "event_date_conservative": (
                    False
                ),

                "event_date_confidence": (
                    None
                ),

                "event_date_needs_review": (
                    False
                ),

                "event_date_evidence_source": (
                    source
                ),

                "event_date_label": (
                    None
                ),

                "event_date_selection_reason": (
                    "exact_date_fallback"
                ),

                "event_date_extractor_version": (
                    None
                ),

                "date_parse_issues": (
                    invalid_values
                ),
            }

    return {
        "event_date": None,

        "event_date_precision": (
            "INVALID"
            if invalid_values
            else "MISSING"
        ),

        "event_date_source": None,

        "event_date_raw": None,

        "event_date_conservative": (
            False
        ),

        "event_date_confidence": (
            None
        ),

        "event_date_needs_review": (
            True
        ),

        "event_date_evidence_source": (
            None
        ),

        "event_date_label": (
            None
        ),

        "event_date_selection_reason": (
            "no_exact_decision_date_available"
        ),

        "event_date_extractor_version": (
            None
        ),

        "date_parse_issues": (
            invalid_values
        ),
    }


# ============================================================
# Registry helpers
# ============================================================

def build_registry_index(
    registry: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case_id")): row
        for row in registry
        if row.get("case_id")
    }


def build_local_document_to_case_id(
    registry: list[dict[str, Any]],
) -> dict[str, str]:
    output: dict[str, str] = {}

    for row in registry:
        document_id = row.get(
            "local_document_id"
        )

        case_id = row.get(
            "case_id"
        )

        if document_id and case_id:
            output[str(document_id)] = str(case_id)

    return output


# ============================================================
# Treatment-event construction
# ============================================================

def treatment_status_effect(
    treatment: str,
    actor: str,
) -> str:
    treatment = treatment.upper()
    actor = actor.upper()

    if actor != "COURT":
        return "NO_STATUS_CHANGE_NON_JUDICIAL"

    if treatment in NEGATIVE_TREATMENTS:
        return TREATMENT_TO_STATUS[treatment]

    if treatment in POSITIVE_TREATMENTS:
        return "POSITIVE_JUDICIAL_TREATMENT"

    return "NO_STATUS_CHANGE"


def build_event(
    row: dict[str, Any],
    registry_index: dict[str, dict[str, Any]],
    local_document_to_case_id: dict[str, str],
    document_metadata: dict[str, dict[str, Any]],
    judgment_dates: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    target_case_id = row.get(
        "target_case_id"
    )

    if not target_case_id:
        return None

    target_case_id = str(
        target_case_id
    )

    authority = registry_index.get(
        target_case_id,
        {},
    )

    treatment = str(
        row.get(
            "treatment",
            "REFERRED_TO",
        )
    ).upper()

    actor = str(
        row.get(
            "treatment_actor",
            row.get(
                "actor",
                "UNKNOWN",
            ),
        )
    ).upper()

    document_id = normalize_spaces(
        str(
            row.get(
                "document_id",
                "",
            )
        )
    )

    event_date = resolve_event_date(
        row,
        document_metadata,
        judgment_dates,
    )

    is_judicial_treatment = (
        actor == "COURT"
        and treatment in STRONG_TREATMENTS
    )

    is_status_changing = (
        actor == "COURT"
        and treatment in NEGATIVE_TREATMENTS
    )

    is_positive_judicial = (
        actor == "COURT"
        and treatment in POSITIVE_TREATMENTS
    )

    event_needs_review = bool(
        row.get(
            "resolution_needs_review",
            False,
        )
        or event_date.get(
            "event_date_needs_review",
            False,
        )
    )

    # A missing/invalid date matters most when the event could
    # change temporal legal status.
    if (
        is_status_changing
        and event_date[
            "event_date"
        ]
        is None
    ):
        event_needs_review = True

    return {
        "treatment_id": row.get(
            "treatment_id"
        ),

        "target_case_id": (
            target_case_id
        ),

        "target_case_name": (
            row.get(
                "target_case_name"
            )
            or authority.get(
                "canonical_name"
            )
        ),

        "target_reporters": (
            row.get(
                "target_reporters",
                authority.get(
                    "reporter_citations",
                    [],
                ),
            )
        ),

        "citing_document_id": (
            document_id
        ),

        "citing_case_id": (
            local_document_to_case_id.get(
                document_id
            )
        ),

        "citing_court": row.get(
            "court"
        ),

        **event_date,

        "treatment": treatment,

        "treatment_actor": actor,

        "treatment_scope": row.get(
            "treatment_scope"
        ),

        "is_judicial_treatment": (
            is_judicial_treatment
        ),

        "is_status_changing": (
            is_status_changing
        ),

        "is_positive_judicial_treatment": (
            is_positive_judicial
        ),

        "status_effect": (
            treatment_status_effect(
                treatment,
                actor,
            )
        ),

        "confidence": round(
            safe_float(
                row.get(
                    "confidence"
                ),
                0.0,
            ),
            3,
        ),

        "trigger": row.get(
            "trigger"
        ),

        "section_type": row.get(
            "section_type"
        ),

        "chunk_id": row.get(
            "chunk_id"
        ),

        "page_start": row.get(
            "page_start"
        ),

        "page_end": row.get(
            "page_end"
        ),

        "citation_text": row.get(
            "citation_text"
        ),

        "citation_sentence": row.get(
            "citation_sentence"
        ),

        "resolution_basis": row.get(
            "resolution_basis"
        ),

        "resolution_confidence": (
            row.get(
                "resolution_confidence"
            )
        ),

        "resolution_needs_review": (
            row.get(
                "resolution_needs_review",
                False,
            )
        ),

        "detector_version": row.get(
            "detector_version"
        ),

        "resolver_version": row.get(
            "resolver_version"
        ),

        "needs_review": (
            event_needs_review
        ),
    }


# ============================================================
# Timeline QA
# ============================================================

def event_sort_key(
    event: dict[str, Any],
) -> tuple[str, str]:
    return (
        str(
            event.get(
                "event_date"
            )
            or "9999-12-31"
        ),
        str(
            event.get(
                "treatment_id"
            )
            or ""
        ),
    )


def detect_timeline_conflicts(
    dated_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Different status-changing labels on the same effective date
    # --------------------------------------------------------

    by_date: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for event in dated_events:
        if event.get(
            "is_status_changing"
        ):
            by_date[
                str(event["event_date"])
            ].append(
                event
            )

    for event_date, events in by_date.items():
        labels = sorted(
            {
                str(
                    event.get(
                        "treatment"
                    )
                )
                for event in events
            }
        )

        if len(labels) > 1:
            issues.append(
                {
                    "type": (
                        "CONFLICTING_STATUS_CHANGES_SAME_DATE"
                    ),
                    "event_date": (
                        event_date
                    ),
                    "treatments": labels,
                }
            )

    # --------------------------------------------------------
    # Positive judicial treatment after an observed overruling
    # --------------------------------------------------------

    overruled_date: str | None = None

    for event in dated_events:
        if (
            event.get(
                "is_status_changing"
            )
            and event.get(
                "treatment"
            )
            == "OVERRULED"
        ):
            overruled_date = str(
                event.get(
                    "event_date"
                )
            )

        elif (
            overruled_date is not None
            and event.get(
                "is_positive_judicial_treatment"
            )
        ):
            issues.append(
                {
                    "type": (
                        "POSITIVE_TREATMENT_AFTER_OBSERVED_OVERRULING"
                    ),
                    "overruled_date": (
                        overruled_date
                    ),
                    "later_event_date": (
                        event.get(
                            "event_date"
                        )
                    ),
                    "later_treatment": (
                        event.get(
                            "treatment"
                        )
                    ),
                    "treatment_id": (
                        event.get(
                            "treatment_id"
                        )
                    ),
                }
            )

    return issues


# ============================================================
# Point-in-time status
# ============================================================

def get_status_as_of(
    timeline: dict[str, Any],
    as_of: str | date,
) -> dict[str, Any]:
    if isinstance(as_of, str):
        parsed_as_of, precision, _ = (
            parse_date_value(
                as_of
            )
        )

        if (
            parsed_as_of is None
            or precision != "EXACT"
        ):
            raise ValueError(
                "--as-of must be an exact date "
                "in YYYY-MM-DD format."
            )

        as_of_date = parsed_as_of

    else:
        as_of_date = as_of

    eligible: list[
        dict[str, Any]
    ] = []

    later: list[
        dict[str, Any]
    ] = []

    for event in timeline.get(
        "dated_events",
        [],
    ):
        raw_date = event.get(
            "event_date"
        )

        if not raw_date:
            continue

        event_date = date.fromisoformat(
            str(raw_date)
        )

        if event_date <= as_of_date:
            eligible.append(event)
        else:
            later.append(event)

    status_changing = [
        event
        for event in eligible
        if event.get(
            "is_status_changing"
        )
    ]

    basis_event: dict[str, Any] | None = None

    status = (
        "NO_OBSERVED_NEGATIVE_TREATMENT"
    )

    if status_changing:
        # Conservative v1 rule:
        # the strongest observed judicial negative treatment
        # remains effective. A later FOLLOWED event cannot
        # silently undo an OVERRULED event.
        basis_event = max(
            status_changing,
            key=lambda event: (
                STATUS_SEVERITY[
                    TREATMENT_TO_STATUS[
                        str(
                            event.get(
                                "treatment"
                            )
                        )
                    ]
                ],
                str(
                    event.get(
                        "event_date"
                    )
                ),
            ),
        )

        status = TREATMENT_TO_STATUS[
            str(
                basis_event.get(
                    "treatment"
                )
            )
        ]

    positive_judicial_events = [
        event
        for event in eligible
        if event.get(
            "is_positive_judicial_treatment"
        )
    ]

    neutral_reference_count = sum(
        1
        for event in eligible
        if event.get(
            "treatment"
        )
        == "REFERRED_TO"
    )

    ignored_non_judicial_negative = [
        event
        for event in eligible
        if (
            event.get(
                "treatment"
            )
            in NEGATIVE_TREATMENTS
            and event.get(
                "treatment_actor"
            )
            != "COURT"
        )
    ]

    return {
        "authority_id": timeline.get(
            "authority_id"
        ),

        "canonical_name": timeline.get(
            "canonical_name"
        ),

        "as_of": as_of_date.isoformat(),

        "status": status,

        "status_basis": (
            "CORPUS_OBSERVED_JUDICIAL_TREATMENT"
        ),

        "basis_event": basis_event,

        "basis_confidence": (
            basis_event.get(
                "confidence"
            )
            if basis_event
            else None
        ),

        "confidence_scope": (
            "Treatment-detection confidence only; "
            "not probability that the authority is globally valid."
        ),

        "has_positive_judicial_treatment": bool(
            positive_judicial_events
        ),

        "positive_judicial_treatment_count": len(
            positive_judicial_events
        ),

        "neutral_reference_count": (
            neutral_reference_count
        ),

        "non_judicial_negative_claims_ignored": len(
            ignored_non_judicial_negative
        ),

        "eligible_dated_events": len(
            eligible
        ),

        "later_events_excluded": len(
            later
        ),

        "undated_events_excluded": len(
            timeline.get(
                "undated_events",
                [],
            )
        ),

        "timeline_needs_review": timeline.get(
            "needs_review",
            False,
        ),

        "reviewable_date_event_count": timeline.get(
            "reviewable_date_event_count",
            0,
        ),

        "reviewable_status_change_date_count": timeline.get(
            "reviewable_status_change_date_count",
            0,
        ),

        "corpus_scope_warning": (
            CORPUS_SCOPE_WARNING
        ),

        "temporal_validity_version": (
            TEMPORAL_VALIDITY_VERSION
        ),
    }


# ============================================================
# Timeline construction
# ============================================================

def build_timelines(
    treatment_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    document_metadata: dict[str, dict[str, Any]],
    judgment_dates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    registry_index = build_registry_index(
        registry
    )

    local_document_to_case_id = (
        build_local_document_to_case_id(
            registry
        )
    )

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in treatment_rows:
        event = build_event(
            row=row,
            registry_index=registry_index,
            local_document_to_case_id=(
                local_document_to_case_id
            ),
            document_metadata=document_metadata,
            judgment_dates=judgment_dates,
        )

        if event is None:
            continue

        grouped[
            str(
                event[
                    "target_case_id"
                ]
            )
        ].append(event)

    output: list[
        dict[str, Any]
    ] = []

    # Include every canonical authority, even if a future pipeline
    # produces an authority without a treatment row.
    for authority in sorted(
        registry,
        key=lambda row: str(
            row.get(
                "case_id",
                "",
            )
        ),
    ):
        authority_id = str(
            authority.get(
                "case_id",
                "",
            )
        )

        events = grouped.get(
            authority_id,
            [],
        )

        dated_events = sorted(
            [
                event
                for event in events
                if event.get(
                    "event_date"
                )
            ],
            key=event_sort_key,
        )

        undated_events = [
            event
            for event in events
            if not event.get(
                "event_date"
            )
        ]

        qa_issues = detect_timeline_conflicts(
            dated_events
        )

        missing_status_change_dates = sum(
            1
            for event in undated_events
            if event.get(
                "is_status_changing"
            )
        )

        if missing_status_change_dates:
            qa_issues.append(
                {
                    "type": (
                        "UNDATED_STATUS_CHANGING_EVENT"
                    ),
                    "count": (
                        missing_status_change_dates
                    ),
                }
            )

        reviewable_status_change_dates = sum(
            1
            for event in dated_events
            if (
                event.get(
                    "event_date_needs_review",
                    False,
                )
                and event.get(
                    "is_status_changing",
                    False,
                )
            )
        )

        if reviewable_status_change_dates:
            qa_issues.append(
                {
                    "type": (
                        "REVIEWABLE_DATE_FOR_STATUS_CHANGING_EVENT"
                    ),
                    "count": (
                        reviewable_status_change_dates
                    ),
                }
            )

        approximate_date_count = sum(
            1
            for event in dated_events
            if event.get(
                "event_date_precision"
            )
            == "YEAR"
        )

        reviewable_date_event_count = sum(
            1
            for event in dated_events
            if event.get(
                "event_date_needs_review",
                False,
            )
        )

        reviewable_status_change_date_count = sum(
            1
            for event in dated_events
            if (
                event.get(
                    "event_date_needs_review",
                    False,
                )
                and event.get(
                    "is_status_changing",
                    False,
                )
            )
        )

        judicial_treatment_count = sum(
            1
            for event in events
            if event.get(
                "is_judicial_treatment"
            )
        )

        status_changing_count = sum(
            1
            for event in events
            if event.get(
                "is_status_changing"
            )
        )

        positive_judicial_count = sum(
            1
            for event in events
            if event.get(
                "is_positive_judicial_treatment"
            )
        )

        needs_review = bool(
            authority.get(
                "needs_review",
                False,
            )
            or qa_issues
        )

        timeline = {
            "authority_id": authority_id,

            "canonical_name": authority.get(
                "canonical_name"
            ),

            "reporter_citations": authority.get(
                "reporter_citations",
                [],
            ),

            "resolution_basis": authority.get(
                "resolution_basis"
            ),

            "resolution_confidence": authority.get(
                "resolution_confidence"
            ),

            "authority_resolution_needs_review": (
                authority.get(
                    "needs_review",
                    False,
                )
            ),

            "event_count": len(events),

            "dated_event_count": len(
                dated_events
            ),

            "undated_event_count": len(
                undated_events
            ),

            "approximate_date_count": (
                approximate_date_count
            ),

            "reviewable_date_event_count": (
                reviewable_date_event_count
            ),

            "reviewable_status_change_date_count": (
                reviewable_status_change_date_count
            ),

            "judicial_treatment_count": (
                judicial_treatment_count
            ),

            "status_changing_event_count": (
                status_changing_count
            ),

            "positive_judicial_event_count": (
                positive_judicial_count
            ),

            "dated_events": dated_events,

            "undated_events": undated_events,

            "qa_issues": qa_issues,

            "needs_review": needs_review,

            "corpus_scope_warning": (
                CORPUS_SCOPE_WARNING
            ),

            "temporal_validity_version": (
                TEMPORAL_VALIDITY_VERSION
            ),

            "generated_at": utc_now(),
        }

        output.append(timeline)

    return output


# ============================================================
# Current observed status snapshot
# ============================================================

def build_status_snapshot(
    timelines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[
        dict[str, Any]
    ] = []

    max_date = date(9999, 12, 31)

    for timeline in timelines:
        status = get_status_as_of(
            timeline,
            max_date,
        )

        status[
            "as_of"
        ] = "LATEST_OBSERVED"

        status[
            "dated_event_count"
        ] = timeline.get(
            "dated_event_count",
            0,
        )

        status[
            "undated_event_count"
        ] = timeline.get(
            "undated_event_count",
            0,
        )

        status[
            "approximate_date_count"
        ] = timeline.get(
            "approximate_date_count",
            0,
        )

        status[
            "qa_issues"
        ] = timeline.get(
            "qa_issues",
            [],
        )

        status[
            "generated_at"
        ] = utc_now()

        output.append(status)

    return output


# ============================================================
# Validation invariants
# ============================================================

def count_temporal_leakage_violations(
    timelines: list[dict[str, Any]],
) -> int:
    """
    Validate that get_status_as_of() never counts a treatment
    occurring after the requested date.
    """

    violations = 0

    for timeline in timelines:
        dated_events = timeline.get(
            "dated_events",
            [],
        )

        for event in dated_events:
            event_date_raw = event.get(
                "event_date"
            )

            if not event_date_raw:
                continue

            event_date = date.fromisoformat(
                str(event_date_raw)
            )

            if event_date <= date.min:
                continue

            query_date = event_date - timedelta(
                days=1
            )

            result = get_status_as_of(
                timeline,
                query_date,
            )

            expected_later = sum(
                1
                for candidate in dated_events
                if date.fromisoformat(
                    str(
                        candidate[
                            "event_date"
                        ]
                    )
                )
                > query_date
            )

            if (
                result[
                    "later_events_excluded"
                ]
                != expected_later
            ):
                violations += 1

    return violations


# ============================================================
# Reporting
# ============================================================

def build_metrics(
    treatment_rows: list[dict[str, Any]],
    timelines: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    events = [
        event
        for timeline in timelines
        for event in (
            timeline.get(
                "dated_events",
                [],
            )
            + timeline.get(
                "undated_events",
                [],
            )
        )
    ]

    treatment_counter = Counter(
        str(
            event.get(
                "treatment",
                "",
            )
        )
        for event in events
    )

    status_counter = Counter(
        str(
            row.get(
                "status",
                "",
            )
        )
        for row in statuses
    )

    exact_dates = sum(
        1
        for event in events
        if event.get(
            "event_date_precision"
        )
        == "EXACT"
    )

    year_only_dates = sum(
        1
        for event in events
        if event.get(
            "event_date_precision"
        )
        == "YEAR"
    )

    undated_events = sum(
        1
        for event in events
        if not event.get(
            "event_date"
        )
    )

    reviewable_date_events = sum(
        1
        for event in events
        if event.get(
            "event_date_needs_review",
            False,
        )
    )

    extractor_date_events = sum(
        1
        for event in events
        if event.get(
            "event_date_source"
        )
        == "judgment_date_extractor"
    )

    source_url_date_events = sum(
        1
        for event in events
        if event.get(
            "event_date_evidence_source"
        )
        == "source_url_filename"
    )

    document_text_date_events = sum(
        1
        for event in events
        if event.get(
            "event_date_evidence_source"
        )
        == "document_text"
    )

    judicial_treatments = sum(
        1
        for event in events
        if event.get(
            "is_judicial_treatment"
        )
    )

    positive_judicial = sum(
        1
        for event in events
        if event.get(
            "is_positive_judicial_treatment"
        )
    )

    status_changing = sum(
        1
        for event in events
        if event.get(
            "is_status_changing"
        )
    )

    party_negative_ignored = sum(
        1
        for event in events
        if (
            event.get(
                "treatment"
            )
            in NEGATIVE_TREATMENTS
            and event.get(
                "treatment_actor"
            )
            != "COURT"
        )
    )

    conflicting_authorities = sum(
        1
        for timeline in timelines
        if timeline.get(
            "qa_issues"
        )
    )

    review_authorities = sum(
        1
        for timeline in timelines
        if timeline.get(
            "needs_review"
        )
    )

    temporal_leakage = (
        count_temporal_leakage_violations(
            timelines
        )
    )

    metrics: dict[str, Any] = {
        "temporal_validity_version": (
            TEMPORAL_VALIDITY_VERSION
        ),

        "treatment_rows_input": len(
            treatment_rows
        ),

        "treatment_events_built": len(
            events
        ),

        "authorities_with_timelines": len(
            timelines
        ),

        "exact_dated_events": exact_dates,

        "year_only_dated_events": (
            year_only_dates
        ),

        "undated_events": undated_events,

        "events_dated_by_judgment_date_extractor": (
            extractor_date_events
        ),

        "events_with_document_text_dates": (
            document_text_date_events
        ),

        "events_with_source_url_fallback_dates": (
            source_url_date_events
        ),

        "events_with_reviewable_dates": (
            reviewable_date_events
        ),

        "judicial_treatment_events": (
            judicial_treatments
        ),

        "positive_judicial_events": (
            positive_judicial
        ),

        "status_changing_judicial_events": (
            status_changing
        ),

        "party_or_nonjudicial_negative_claims_ignored": (
            party_negative_ignored
        ),

        "temporal_leakage_violations": (
            temporal_leakage
        ),

        "authorities_with_qa_issues": (
            conflicting_authorities
        ),

        "authorities_needing_review": (
            review_authorities
        ),
    }

    for label in (
        "FOLLOWED",
        "RELIED_ON",
        "APPROVED",
        "DISTINGUISHED",
        "DOUBTED",
        "DISAPPROVED",
        "OVERRULED",
        "REFERRED_TO",
    ):
        metrics[
            f"events_{label.lower()}"
        ] = treatment_counter.get(
            label,
            0,
        )

    for status in (
        "NO_OBSERVED_NEGATIVE_TREATMENT",
        "DISTINGUISHED",
        "DOUBTED",
        "DISAPPROVED",
        "OVERRULED",
    ):
        metrics[
            f"status_{status.lower()}"
        ] = status_counter.get(
            status,
            0,
        )

    return metrics


def write_report(
    metrics: dict[str, Any],
) -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "value",
            ],
        )

        writer.writeheader()

        for metric, value in metrics.items():
            writer.writerow(
                {
                    "metric": metric,
                    "value": value,
                }
            )


def write_qa_report(
    timelines: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: list[str] = [
        "# Temporal Validity QA",
        "",
        f"Version: `{TEMPORAL_VALIDITY_VERSION}`",
        "",
        "## Semantics",
        "",
        (
            "This module reports **corpus-observed temporal "
            "treatment status**. It does not claim that an "
            "authority is globally good law."
        ),
        "",
        (
            "Only `COURT` treatment events may change temporal "
            "status. Party submissions, quoted authorities and "
            "unknown actors are retained as evidence but ignored "
            "for status changes."
        ),
        "",
        (
            "Temporal event dates come from the dedicated judgment-date "
            "extractor whenever available. Generic document `year` values "
            "are not used as event dates because they may represent filing "
            "or case-number years."
        ),
        "",
        "## Metrics",
        "",
    ]

    for metric, value in metrics.items():
        lines.append(
            f"- **{metric}**: {value}"
        )

    lines.extend(
        [
            "",
            "## Authorities requiring review",
            "",
        ]
    )

    review_rows = [
        timeline
        for timeline in timelines
        if timeline.get(
            "needs_review"
        )
    ]

    if not review_rows:
        lines.append(
            "No authority-level QA issues detected."
        )
    else:
        for timeline in review_rows:
            lines.append(
                (
                    f"### {timeline.get('authority_id')} — "
                    f"{timeline.get('canonical_name') or '[no canonical name]'}"
                )
            )
            lines.append("")

            if timeline.get(
                "authority_resolution_needs_review"
            ):
                lines.append(
                    "- Canonical authority resolver marked this authority for review."
                )

            for issue in timeline.get(
                "qa_issues",
                [],
            ):
                lines.append(
                    "- Temporal QA: `"
                    + str(
                        issue.get(
                            "type",
                            "UNKNOWN",
                        )
                    )
                    + "` — "
                    + json.dumps(
                        issue,
                        ensure_ascii=False,
                    )
                )

            lines.append("")

    lines.extend(
        [
            "## Corpus-scope warning",
            "",
            CORPUS_SCOPE_WARNING,
            "",
        ]
    )

    QA_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Console helpers
# ============================================================

def print_metrics(
    metrics: dict[str, Any],
) -> None:
    print()
    print("Temporal Legal Validity Report")
    print("=" * 70)

    labels = (
        (
            "Treatment events processed",
            "treatment_events_built",
        ),
        (
            "Authorities with timelines",
            "authorities_with_timelines",
        ),
        (
            "Exact dated events",
            "exact_dated_events",
        ),
        (
            "Year-only dated events",
            "year_only_dated_events",
        ),
        (
            "Undated events",
            "undated_events",
        ),
        (
            "Events dated by judgment-date extractor",
            "events_dated_by_judgment_date_extractor",
        ),
        (
            "Events with document-text dates",
            "events_with_document_text_dates",
        ),
        (
            "Events with source-URL fallback dates",
            "events_with_source_url_fallback_dates",
        ),
        (
            "Events with reviewable dates",
            "events_with_reviewable_dates",
        ),
        (
            "Judicial treatment events",
            "judicial_treatment_events",
        ),
        (
            "Positive judicial events",
            "positive_judicial_events",
        ),
        (
            "Status-changing judicial events",
            "status_changing_judicial_events",
        ),
        (
            "Non-judicial negative claims ignored",
            "party_or_nonjudicial_negative_claims_ignored",
        ),
        (
            "Temporal leakage violations",
            "temporal_leakage_violations",
        ),
        (
            "Authorities with QA issues",
            "authorities_with_qa_issues",
        ),
        (
            "Authorities needing review",
            "authorities_needing_review",
        ),
    )

    for title, key in labels:
        print(
            f"{title}: "
            f"{metrics.get(key, 0)}"
        )

    print()
    print("Observed status snapshot")
    print("-" * 70)

    for status in (
        "NO_OBSERVED_NEGATIVE_TREATMENT",
        "DISTINGUISHED",
        "DOUBTED",
        "DISAPPROVED",
        "OVERRULED",
    ):
        key = f"status_{status.lower()}"

        print(
            f"{status}: "
            f"{metrics.get(key, 0)}"
        )

    print()
    print(
        f"Timeline output: {TIMELINE_OUTPUT_PATH}"
    )
    print(
        f"Status output: {STATUS_OUTPUT_PATH}"
    )
    print(
        f"Report: {REPORT_PATH}"
    )
    print(
        f"QA report: {QA_PATH}"
    )
    print("=" * 70)


def print_case_status(
    timelines: list[dict[str, Any]],
    case_id: str,
    as_of: str | None,
) -> None:
    index = {
        str(
            timeline.get(
                "authority_id"
            )
        ): timeline
        for timeline in timelines
    }

    timeline = index.get(
        case_id
    )

    if timeline is None:
        raise ValueError(
            f"Unknown authority ID: {case_id}"
        )

    if as_of is None:
        result = get_status_as_of(
            timeline,
            date(9999, 12, 31),
        )
        result["as_of"] = (
            "LATEST_OBSERVED"
        )
    else:
        result = get_status_as_of(
            timeline,
            as_of,
        )

    print()
    print("Point-in-Time Authority Status")
    print("=" * 70)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=" * 70)


# ============================================================
# Pipeline
# ============================================================

def run_temporal_validity(
    case_id: str | None = None,
    as_of: str | None = None,
) -> None:
    treatment_rows = read_jsonl(
        RESOLVED_TREATMENTS_PATH
    )

    registry = read_jsonl(
        AUTHORITY_REGISTRY_PATH
    )

    judgment_dates = (
        load_judgment_date_index()
    )

    document_metadata = (
        build_document_metadata_index()
    )

    print()
    print("Temporal Legal Validity")
    print("=" * 70)
    print(
        "Temporal validity version: "
        f"{TEMPORAL_VALIDITY_VERSION}"
    )
    print(
        "Resolved treatment rows: "
        f"{len(treatment_rows)}"
    )
    print(
        "Canonical authorities: "
        f"{len(registry)}"
    )
    print(
        "Judgment-date rows: "
        f"{len(judgment_dates)}"
    )
    print(
        "Document metadata rows: "
        f"{len(document_metadata)}"
    )
    print("=" * 70)

    timelines = build_timelines(
        treatment_rows=treatment_rows,
        registry=registry,
        document_metadata=document_metadata,
        judgment_dates=judgment_dates,
    )

    statuses = build_status_snapshot(
        timelines
    )

    write_jsonl(
        TIMELINE_OUTPUT_PATH,
        timelines,
    )

    write_jsonl(
        STATUS_OUTPUT_PATH,
        statuses,
    )

    metrics = build_metrics(
        treatment_rows=treatment_rows,
        timelines=timelines,
        statuses=statuses,
    )

    write_report(
        metrics
    )

    write_qa_report(
        timelines,
        metrics,
    )

    print_metrics(
        metrics
    )

    if case_id:
        print_case_status(
            timelines=timelines,
            case_id=case_id,
            as_of=as_of,
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build corpus-observed temporal treatment timelines "
            "for canonical legal authorities."
        )
    )

    parser.add_argument(
        "--case-id",
        help=(
            "Optionally print temporal status for one canonical "
            "authority, e.g. CASE_ABC123."
        ),
    )

    parser.add_argument(
        "--as-of",
        help=(
            "Exact point-in-time date in YYYY-MM-DD format. "
            "Requires --case-id."
        ),
    )

    args = parser.parse_args()

    if (
        args.as_of
        and not args.case_id
    ):
        parser.error(
            "--as-of requires --case-id"
        )

    if args.as_of:
        try:
            parsed = date.fromisoformat(
                args.as_of
            )
        except ValueError:
            parser.error(
                "--as-of must use YYYY-MM-DD"
            )

        if parsed.year < 1800:
            parser.error(
                "--as-of year must be >= 1800"
            )

    return args


def main() -> None:
    args = parse_args()

    run_temporal_validity(
        case_id=args.case_id,
        as_of=args.as_of,
    )


if __name__ == "__main__":
    main()
