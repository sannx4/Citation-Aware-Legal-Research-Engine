from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_STATUTES_DIR = PROJECT_ROOT / "data" / "raw" / "statutes"
MANIFEST_PATH = RAW_STATUTES_DIR / "statute_manifest.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = PROCESSED_DIR / "statute_documents.jsonl"
REPORT_PATH = REPORTS_DIR / "statute_ingestion_report.csv"
QA_PATH = REPORTS_DIR / "statute_ingestion_qa.md"

STATUTE_INGESTION_VERSION = "1.0.0"

REQUIRED_COLUMNS = {
    "statute_id",
    "title",
    "year",
    "jurisdiction",
    "source",
    "file_name",
}

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".json"}
VALID_STATUSES = {"", "ACTIVE", "REPEALED", "PARTIALLY_REPEALED", "UNKNOWN"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_year(value: Any) -> int:
    raw = norm(value)
    if not re.fullmatch(r"\d{4}", raw):
        raise ValueError(f"invalid year: {raw!r}")
    year = int(raw)
    if year < 1700 or year > 2200:
        raise ValueError(f"unreasonable year: {year}")
    return year


def parse_iso_date(value: Any) -> str | None:
    raw = norm(value)
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"expected YYYY-MM-DD date, got {raw!r}") from exc


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"statute manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(
                "manifest missing required columns: " + ", ".join(sorted(missing))
            )
        return [
            {key: norm(value) for key, value in row.items()}
            for row in reader
        ]


def validate_row(row: dict[str, str], seen_ids: set[str]) -> list[str]:
    issues: list[str] = []
    statute_id = row.get("statute_id", "")
    file_name = row.get("file_name", "")

    if not statute_id:
        issues.append("missing_statute_id")
    elif statute_id in seen_ids:
        issues.append("duplicate_statute_id")

    if not row.get("title"):
        issues.append("missing_title")
    if not row.get("jurisdiction"):
        issues.append("missing_jurisdiction")
    if not row.get("source"):
        issues.append("missing_source")

    try:
        parse_year(row.get("year"))
    except ValueError:
        issues.append("invalid_year")

    if not file_name:
        issues.append("missing_file_name")
    elif Path(file_name).suffix.lower() not in SUPPORTED_SUFFIXES:
        issues.append(f"unsupported_file_type:{Path(file_name).suffix.lower() or '[none]'}")

    status = row.get("status", "").upper()
    if status not in VALID_STATUSES:
        issues.append(f"invalid_status:{status}")

    try:
        start = parse_iso_date(row.get("effective_from"))
        end = parse_iso_date(row.get("effective_to"))
        if start and end and start > end:
            issues.append("effective_from_after_effective_to")
    except ValueError:
        issues.append("invalid_effective_date")

    return issues


def extract_text(path: Path) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8"), {
            "extraction_method": "utf8_text",
            "pages": None,
        }

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            text = ""
            for key in ("text", "full_text", "content", "body"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
            if not text:
                raise ValueError(
                    "JSON statute file must contain text/full_text/content/body"
                )
        else:
            raise ValueError("JSON statute file must contain a string or object")
        return text, {"extraction_method": "json_text", "pages": None}

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required for PDF statutes. "
                "Install with: python -m pip install pymupdf"
            ) from exc

        doc = fitz.open(path)
        try:
            page_texts = [page.get_text("text") for page in doc]
        finally:
            doc.close()
        return "\n\n".join(page_texts), {
            "extraction_method": "pymupdf",
            "pages": len(page_texts),
        }

    raise ValueError(f"unsupported file type: {suffix}")


def assess_quality(text: str) -> dict[str, Any]:
    compact = norm(text)
    characters = len(text)
    words = len(compact.split()) if compact else 0
    non_space = sum(1 for c in text if not c.isspace())
    alpha = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha / non_space if non_space else 0.0

    reasons: list[str] = []
    if characters < 500:
        reasons.append("very_short_text")
    if words < 100:
        reasons.append("very_low_word_count")
    if alpha_ratio < 0.55:
        reasons.append("low_alpha_ratio")

    return {
        "characters": characters,
        "words": words,
        "alpha_ratio": round(alpha_ratio, 4),
        "quality": "review" if reasons else "good",
        "quality_reasons": reasons,
        "needs_review": bool(reasons),
    }


def build_record(row: dict[str, str]) -> dict[str, Any]:
    file_path = RAW_STATUTES_DIR / row["file_name"]
    if not file_path.exists():
        raise FileNotFoundError(f"missing statute file: {file_path}")

    text, extraction = extract_text(file_path)
    quality = assess_quality(text)

    return {
        "statute_id": row["statute_id"],
        "title": row["title"],
        "short_title": row.get("short_title") or None,
        "year": parse_year(row["year"]),
        "jurisdiction": row["jurisdiction"],
        "source": row["source"],
        "source_url": row.get("source_url") or None,
        "status": row.get("status", "").upper() or "UNKNOWN",
        "effective_from": parse_iso_date(row.get("effective_from")),
        "effective_to": parse_iso_date(row.get("effective_to")),
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "sha256": sha256_file(file_path),
        **extraction,
        **quality,
        "notes": row.get("notes") or None,
        "text": text,
        "ingestion_version": STATUTE_INGESTION_VERSION,
        "ingested_at": utc_now(),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_reports(
    successful: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    manifest_count: int,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "statute_ingestion_version": STATUTE_INGESTION_VERSION,
        "manifest_rows": manifest_count,
        "successful_documents": len(successful),
        "failed_documents": len(failed),
        "good_quality_documents": sum(
            1 for row in successful if row["quality"] == "good"
        ),
        "review_documents": sum(1 for row in successful if row["needs_review"]),
        "total_characters": sum(int(row["characters"]) for row in successful),
        "total_words": sum(int(row["words"]) for row in successful),
    }

    with REPORT_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})

    lines = [
        "# Statute Ingestion QA",
        "",
        f"Version: `{STATUTE_INGESTION_VERSION}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- **{k}**: {v}" for k, v in metrics.items())
    lines.extend(["", "## Failed manifest rows", ""])

    if failed:
        lines.extend(
            "- " + json.dumps(item, ensure_ascii=False)
            for item in failed
        )
    else:
        lines.append("No manifest-level ingestion failures.")

    review_rows = [row for row in successful if row["needs_review"]]
    if review_rows:
        lines.extend(["", "## Text-quality review queue", ""])
        for row in review_rows:
            lines.append(
                f"- `{row['statute_id']}` — {row['title']} — "
                + ", ".join(row["quality_reasons"])
            )

    QA_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    successful: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row_number, row in enumerate(manifest, start=2):
        issues = validate_row(row, seen_ids)
        statute_id = row.get("statute_id", "")
        if statute_id:
            seen_ids.add(statute_id)

        if issues:
            failed.append({
                "row": row_number,
                "statute_id": statute_id or None,
                "issues": issues,
            })
            continue

        try:
            successful.append(build_record(row))
        except Exception as exc:
            failed.append({
                "row": row_number,
                "statute_id": statute_id or None,
                "issues": [f"{type(exc).__name__}: {exc}"],
            })

    write_jsonl(OUTPUT_PATH, successful)
    write_reports(successful, failed, len(manifest))

    print()
    print("Statute Ingestion")
    print("=" * 70)
    print(f"Ingestion version: {STATUTE_INGESTION_VERSION}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Successful documents: {len(successful)}")
    print(f"Failed documents: {len(failed)}")
    print(
        "Documents needing review: "
        f"{sum(1 for row in successful if row['needs_review'])}"
    )
    print()
    print(f"Output: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"QA report: {QA_PATH}")
    print("=" * 70)

    if failed:
        print()
        print("Failures")
        print("-" * 70)
        for item in failed:
            print(json.dumps(item, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest canonical statute documents from a manifest "
            "into reproducible processed JSONL records."
        )
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    run()


if __name__ == "__main__":
    main()
