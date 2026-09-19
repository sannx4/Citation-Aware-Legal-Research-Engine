from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "processed" / "documents"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "judgment_dates.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "judgment_date_extraction_report.csv"
QA_PATH = PROJECT_ROOT / "reports" / "judgment_date_extraction_qa.md"

DATE_EXTRACTOR_VERSION = "1.0.2"

# Search enough of the beginning of the judgment to cover headers that
# spill into page 2, while still avoiding most body-date noise.
HEADER_SEARCH_CHARS = 12000

MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)

NUMERIC_DATE_PATTERN = (
    r"(?:"
    r"\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{4}"
    r"|"
    r"\d{4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{1,2}"
    r")"
)

TEXTUAL_DATE_PATTERN = (
    rf"(?:"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH_PATTERN}\s*,?\s+\d{{4}}"
    rf"|"
    rf"{MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s+\d{{4}}"
    rf")"
)

DAY_OF_TEXTUAL_PATTERN = (
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+DAY\s+OF\s+"
    rf"{MONTH_PATTERN}\s*,?\s+\d{{4}}"
)

ANY_DATE_PATTERN = (
    rf"(?:{NUMERIC_DATE_PATTERN}|{TEXTUAL_DATE_PATTERN}|"
    rf"{DAY_OF_TEXTUAL_PATTERN})"
)

FIRST_PAGE_DATE_PATTERN = re.compile(
    rf"(?i)\b({ANY_DATE_PATTERN})\b"
)

# Earlier item = stronger semantic signal.
# Patterns deliberately target the decision/order being extracted, not
# arbitrary "order dated ..." references in the body.
LABELED_PATTERNS: list[tuple[str, int, re.Pattern[str]]] = [
    (
        "DATE_OF_DECISION",
        100,
        re.compile(
            rf"(?is)\bdate\s+of\s+decision\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "DATE_OF_JUDGMENT",
        100,
        re.compile(
            rf"(?is)\bdate\s+of\s+judg(?:e)?ment\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "DATE_OF_ORDER_PRONOUNCED",
        100,
        re.compile(
            rf"(?is)\bdate\s+of\s+order\s+pronounced\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "JUDGMENT_PRONOUNCED_ON",
        98,
        re.compile(
            rf"(?is)\bjudg(?:e)?ment\s+pronounced\s+on\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "PRONOUNCED_ON",
        97,
        re.compile(
            rf"(?is)\bpronounced\s+on\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "DECIDED_ON",
        97,
        re.compile(
            rf"(?is)\bdecided\s+on\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "ORDER_DELIVERED_ON",
        97,
        re.compile(
            rf"(?is)\border\s+delivered\s+on\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "DELIVERED_ON",
        96,
        re.compile(
            rf"(?is)\bdelivered\s+on\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "DATED_THIS_DAY",
        96,
        re.compile(
            rf"(?is)\bdated\s+this\s+the\s+({DAY_OF_TEXTUAL_PATTERN})"
        ),
    ),
    (
        "ORDER_NUMBER_DATED",
        92,
        re.compile(
            rf"(?is)\border\s+no\.?\s*[:\-]?\s*[A-Za-z0-9./-]{{0,30}}"
            rf"\s+dated\s*[:\-]?\s*({ANY_DATE_PATTERN})"
        ),
    ),
    (
        "HEADER_DATED",
        88,
        re.compile(
            rf"(?im)^\s*dated\s*[:\-]\s*({ANY_DATE_PATTERN})\s*\.?\s*$"
        ),
    ),
]

RESERVED_PATTERN = re.compile(
    rf"(?is)\breserved\s+(?:on\s*)?[:\-]?\s*({ANY_DATE_PATTERN})"
)

SOURCE_URL_DATE_PATTERN = re.compile(
    r"_(\d{4}-\d{2}-\d{2})\.pdf(?:$|\?)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_date(raw: str) -> date | None:
    value = normalize_spaces(raw)

    value = re.sub(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+DAY\s+OF\s+",
        r"\1 ",
        value,
    )

    value = re.sub(
        r"(?i)(\d{1,2})(st|nd|rd|th)\b",
        r"\1",
        value,
    )

    value = re.sub(
        r"\s*([./-])\s*",
        r"\1",
        value,
    )

    value = value.replace(",", " ")
    value = normalize_spaces(value)

    # ISO-ish yyyy.mm.dd / yyyy/mm/dd / yyyy-mm-dd
    match = re.fullmatch(
        r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})",
        value,
    )
    if match:
        try:
            return date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        except ValueError:
            return None

    # dd.mm.yyyy / dd/mm/yyyy / dd-mm-yyyy
    match = re.fullmatch(
        r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})",
        value,
    )
    if match:
        try:
            return date(
                int(match.group(3)),
                int(match.group(2)),
                int(match.group(1)),
            )
        except ValueError:
            return None

    formats = (
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


def extract_source_url_date(source_url: str) -> date | None:
    if not source_url:
        return None

    match = SOURCE_URL_DATE_PATTERN.search(source_url.strip())
    if not match:
        return None

    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def context_window(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return normalize_spaces(text[left:right])


def collect_labeled_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for label, priority, pattern in LABELED_PATTERNS:
        for match in pattern.finditer(text):
            raw_date = match.group(1)
            parsed = parse_date(raw_date)

            if parsed is None:
                continue

            candidates.append(
                {
                    "date": parsed.isoformat(),
                    "raw_date": normalize_spaces(raw_date),
                    "label": label,
                    "priority": priority,
                    "match_start": match.start(),
                    "context": context_window(
                        text,
                        match.start(),
                        match.end(),
                    ),
                }
            )

    # Deduplicate the same semantic match while preserving strongest rule.
    best: dict[tuple[str, str], dict[str, Any]] = {}

    for candidate in candidates:
        key = (
            str(candidate["date"]),
            str(candidate["label"]),
        )

        old = best.get(key)

        if old is None or candidate["priority"] > old["priority"]:
            best[key] = candidate

    return sorted(
        best.values(),
        key=lambda row: (
            -int(row["priority"]),
            int(row["match_start"]),
        ),
    )



def first_page_text(text: str) -> str:
    """
    Keep primary judgment-date extraction on page 1.

    This prevents body phrases such as:
        "impugned judgment dated 15/12/2018"
    from being mistaken for the date of the present judgment.
    """

    marker = re.search(
        r"(?im)^---\s*PAGE\s+2\s*---\s*$",
        text,
    )

    if marker is not None:
        return text[:marker.start()]

    return text[:8000]


def collect_first_page_dates(
    text: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for match in FIRST_PAGE_DATE_PATTERN.finditer(text):
        parsed = parse_date(match.group(1))

        if parsed is None:
            continue

        rows.append(
            {
                "date": parsed.isoformat(),
                "raw_date": normalize_spaces(match.group(1)),
                "match_start": match.start(),
                "context": context_window(
                    text,
                    match.start(),
                    match.end(),
                ),
            }
        )

    unique: dict[str, dict[str, Any]] = {}

    for row in rows:
        unique.setdefault(
            str(row["date"]),
            row,
        )

    return sorted(
        unique.values(),
        key=lambda row: int(row["match_start"]),
    )

def collect_reserved_dates(text: str) -> list[str]:
    values: list[str] = []

    for match in RESERVED_PATTERN.finditer(text):
        parsed = parse_date(match.group(1))
        if parsed is not None:
            values.append(parsed.isoformat())

    return sorted(set(values))


def choose_candidate(
    candidates: list[dict[str, Any]],
    first_page_dates: list[dict[str, Any]],
    source_url_date: date | None,
) -> tuple[dict[str, Any] | None, float, bool, str]:
    """
    Selection policy:

    1. Prefer explicit present-judgment labels on page 1.
    2. If no explicit label exists, a page-1 date matching the
       eCourts URL date is accepted as document-text evidence.
    3. URL-only dates are allowed only as reviewable fallback.

    The full judgment body is intentionally not searched for
    generic "judgment dated ..." phrases.
    """

    url_iso = (
        source_url_date.isoformat()
        if source_url_date is not None
        else None
    )

    if candidates:
        # Prefer an explicit candidate that is corroborated by URL.
        if url_iso is not None:
            agreeing = [
                row
                for row in candidates
                if row["date"] == url_iso
            ]

            if agreeing:
                selected = sorted(
                    agreeing,
                    key=lambda row: (
                        -int(row["priority"]),
                        int(row["match_start"]),
                    ),
                )[0]

                return (
                    selected,
                    0.995,
                    False,
                    "explicit_text_confirmed_by_source_url",
                )

        selected = candidates[0]

        top_priority = int(selected["priority"])
        top_dates = {
            str(row["date"])
            for row in candidates
            if int(row["priority"]) >= top_priority - 2
        }

        if len(top_dates) > 1:
            return (
                selected,
                0.80,
                True,
                "conflicting_explicit_header_dates",
            )

        if (
            url_iso is not None
            and selected["date"] != url_iso
        ):
            return (
                selected,
                0.90,
                True,
                "explicit_text_source_url_disagreement",
            )

        return (
            selected,
            0.97,
            False,
            "explicit_header_text",
        )

    # No explicit label. If URL date is visibly present on page 1,
    # use the page-1 occurrence as text evidence rather than as a
    # URL-only guess.
    if url_iso is not None:
        matching_page_dates = [
            row
            for row in first_page_dates
            if row["date"] == url_iso
        ]

        if matching_page_dates:
            row = matching_page_dates[0]

            return (
                {
                    "date": row["date"],
                    "raw_date": row["raw_date"],
                    "label": "FIRST_PAGE_DATE_MATCHING_SOURCE_URL",
                    "priority": 90,
                    "match_start": row["match_start"],
                    "context": row["context"],
                },
                0.96,
                False,
                "first_page_date_confirmed_by_source_url",
            )

        return (
            {
                "date": url_iso,
                "raw_date": url_iso,
                "label": "SOURCE_URL_FILENAME",
                "priority": 70,
                "match_start": -1,
                "context": None,
            },
            0.78,
            True,
            "source_url_fallback_requires_review",
        )

    return (
        None,
        0.0,
        True,
        "no_decision_date_found",
    )

def extract_document_date(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    document_id = str(
        data.get("document_id") or path.stem
    )

    text = str(data.get("text") or "")
    header_text = first_page_text(text)

    source_url = str(
        data.get("source_url")
        or data.get("url")
        or ""
    )

    source_url_date = extract_source_url_date(source_url)

    candidates = collect_labeled_candidates(header_text)
    first_page_dates = collect_first_page_dates(header_text)
    reserved_dates = collect_reserved_dates(header_text)

    selected, confidence, needs_review, reason = choose_candidate(
        candidates,
        first_page_dates,
        source_url_date,
    )

    if selected is None:
        decision_date = None
        raw_date = None
        date_label = None
        date_source = None
        date_precision = "MISSING"
        context = None
    else:
        decision_date = selected["date"]
        raw_date = selected["raw_date"]
        date_label = selected["label"]
        context = selected.get("context")
        date_precision = "EXACT"

        if date_label == "SOURCE_URL_FILENAME":
            date_source = "source_url_filename"
        else:
            date_source = "document_text"

    url_agrees = (
        (
            decision_date == source_url_date.isoformat()
        )
        if (
            decision_date is not None
            and source_url_date is not None
            and date_source == "document_text"
        )
        else None
    )

    # If a strong textual date disagrees with URL filename date,
    # retain the text date but surface the discrepancy for review.
    if (
        decision_date is not None
        and source_url_date is not None
        and date_source == "document_text"
        and url_agrees is False
    ):
        needs_review = True
        confidence = min(confidence, 0.90)
        reason = "document_text_source_url_disagreement"

    # Compare extracted decision year with the manifest/processed `year`.
    # In this corpus, `year` often represents filing/case-number year, so
    # disagreement is informational, NOT an extraction error.
    metadata_year = data.get("year")

    return {
        "document_id": document_id,
        "title": data.get("title"),
        "court": data.get("court"),

        "decision_date": decision_date,
        "date_precision": date_precision,
        "date_source": date_source,
        "date_label": date_label,
        "raw_date": raw_date,
        "confidence": round(float(confidence), 3),
        "needs_review": bool(needs_review),
        "selection_reason": reason,

        "source_url_date": (
            source_url_date.isoformat()
            if source_url_date is not None
            else None
        ),
        "source_url_agrees": (
            bool(url_agrees)
            if (
                decision_date is not None
                and source_url_date is not None
                and date_source == "document_text"
            )
            else None
        ),

        "metadata_year": metadata_year,
        "decision_year": (
            int(decision_date[:4])
            if decision_date
            else None
        ),
        "metadata_year_differs_from_decision_year": (
            bool(
                decision_date
                and metadata_year not in (None, "")
                and int(metadata_year) != int(decision_date[:4])
            )
            if str(metadata_year).isdigit()
            else None
        ),

        "reserved_dates": reserved_dates,
        "candidate_count": len(candidates),
        "candidates": candidates[:10],
        "first_page_date_count": len(first_page_dates),
        "first_page_dates": first_page_dates[:15],
        "selected_context": context,

        "date_extractor_version": DATE_EXTRACTOR_VERSION,
        "extracted_at": utc_now(),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "date_extractor_version": DATE_EXTRACTOR_VERSION,
        "documents_processed": len(rows),
        "exact_dates_extracted": sum(
            1 for row in rows if row["date_precision"] == "EXACT"
        ),
        "missing_dates": sum(
            1 for row in rows if not row["decision_date"]
        ),
        "document_text_dates": sum(
            1 for row in rows if row["date_source"] == "document_text"
        ),
        "source_url_fallback_dates": sum(
            1 for row in rows if row["date_source"] == "source_url_filename"
        ),
        "needs_review": sum(
            1 for row in rows if row["needs_review"]
        ),
        "text_url_agreements": sum(
            1 for row in rows if row["source_url_agrees"] is True
        ),
        "text_url_disagreements": sum(
            1 for row in rows if row["source_url_agrees"] is False
        ),
        "metadata_year_differs_from_decision_year": sum(
            1
            for row in rows
            if row["metadata_year_differs_from_decision_year"] is True
        ),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "value"],
        )
        writer.writeheader()

        for key, value in metrics.items():
            writer.writerow(
                {
                    "metric": key,
                    "value": value,
                }
            )

    return metrics


def write_qa(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    QA_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Judgment Date Extraction QA",
        "",
        f"Version: `{DATE_EXTRACTOR_VERSION}`",
        "",
        "## Important semantic rule",
        "",
        (
            "The existing document `year` field is not assumed to be the "
            "judgment date year. In this corpus it can represent the case/"
            "filing year. Temporal reasoning must use `decision_date`."
        ),
        "",
        "## Metrics",
        "",
    ]

    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value}")

    lines.extend(
        [
            "",
            "## Review queue",
            "",
        ]
    )

    review_rows = [
        row
        for row in rows
        if row["needs_review"]
    ]

    if not review_rows:
        lines.append("No date-extraction rows require review.")
    else:
        for row in review_rows:
            lines.append(
                f"### {row['document_id']} — {row.get('court')}"
            )
            lines.append("")
            lines.append(
                f"- Selected date: `{row.get('decision_date')}`"
            )
            lines.append(
                f"- Source: `{row.get('date_source')}`"
            )
            lines.append(
                f"- Label: `{row.get('date_label')}`"
            )
            lines.append(
                f"- Confidence: `{row.get('confidence')}`"
            )
            lines.append(
                f"- Reason: `{row.get('selection_reason')}`"
            )
            lines.append(
                f"- Source URL date: `{row.get('source_url_date')}`"
            )

            if row.get("selected_context"):
                lines.append(
                    f"- Context: {row['selected_context']}"
                )

            lines.append("")

    QA_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def print_summary(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    print()
    print("Judgment Date Extraction")
    print("=" * 70)
    print(f"Extractor version: {DATE_EXTRACTOR_VERSION}")
    print(f"Documents processed: {metrics['documents_processed']}")
    print(f"Exact dates extracted: {metrics['exact_dates_extracted']}")
    print(f"Missing dates: {metrics['missing_dates']}")
    print(f"Document-text dates: {metrics['document_text_dates']}")
    print(
        "Source-URL fallback dates: "
        f"{metrics['source_url_fallback_dates']}"
    )
    print(f"Needs review: {metrics['needs_review']}")
    print(f"Text/URL agreements: {metrics['text_url_agreements']}")
    print(
        "Text/URL disagreements: "
        f"{metrics['text_url_disagreements']}"
    )
    print(
        "Metadata year != decision year: "
        f"{metrics['metadata_year_differs_from_decision_year']}"
    )
    print()
    print(f"Output: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"QA report: {QA_PATH}")
    print("=" * 70)

    review_rows = [row for row in rows if row["needs_review"]]

    if review_rows:
        print()
        print("Review queue")
        print("-" * 70)
        for row in review_rows:
            print(
                f"{row['document_id']} | "
                f"date={row.get('decision_date')} | "
                f"source={row.get('date_source')} | "
                f"reason={row.get('selection_reason')}"
            )


def run() -> None:
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Processed documents directory not found: {DOCUMENTS_DIR}"
        )

    files = sorted(DOCUMENTS_DIR.glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No processed document JSON files found in: {DOCUMENTS_DIR}"
        )

    rows: list[dict[str, Any]] = []

    for path in files:
        rows.append(
            extract_document_date(path)
        )

    write_jsonl(OUTPUT_PATH, rows)
    metrics = write_report(rows)
    write_qa(rows, metrics)
    print_summary(rows, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract exact judgment/decision dates from processed "
            "Indian court judgments without confusing filing-year "
            "metadata with decision dates."
        )
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    run()


if __name__ == "__main__":
    main()
