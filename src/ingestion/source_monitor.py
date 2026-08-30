import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import MANIFEST_PATH, SOURCE_SCHEMA_CHANGES_PATH


DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",
    r"\d{2}-\d{2}-\d{4}",
    r"\d{2}/\d{2}/\d{4}",
]


@dataclass
class SchemaIssue:
    document_id: str
    source: str
    url: str
    issue_type: str
    severity: str
    message: str


def is_pdf_url(url: str) -> bool:
    clean_url = str(url).lower().split("?")[0].split("#")[0]
    return clean_url.endswith(".pdf")


def has_valid_year(year: str) -> bool:
    year = str(year).strip()

    if not year or year.lower() == "nan":
        return True

    return year.isdigit() and len(year) == 4


def check_manifest_row(row: pd.Series) -> list[SchemaIssue]:
    issues = []

    document_id = str(row.get("document_id", ""))
    source = str(row.get("source", ""))
    url = str(row.get("url", ""))
    title = str(row.get("title", ""))
    year = str(row.get("year", ""))

    if not title or title.lower() == "nan":
        issues.append(
            SchemaIssue(
                document_id=document_id,
                source=source,
                url=url,
                issue_type="missing_title_field",
                severity="high",
                message="Title field is missing or empty.",
            )
        )

    if not url or url.lower() == "nan":
        issues.append(
            SchemaIssue(
                document_id=document_id,
                source=source,
                url=url,
                issue_type="missing_pdf_link",
                severity="critical",
                message="URL/PDF link is missing.",
            )
        )

    if source.lower() == "ecourts" and url and not is_pdf_url(url):
        issues.append(
            SchemaIssue(
                document_id=document_id,
                source=source,
                url=url,
                issue_type="missing_pdf_link",
                severity="critical",
                message="eCourts row does not contain a direct PDF URL.",
            )
        )

    if not has_valid_year(year):
        issues.append(
            SchemaIssue(
                document_id=document_id,
                source=source,
                url=url,
                issue_type="changed_date_format",
                severity="medium",
                message=f"Year/date field format changed or invalid: {year}",
            )
        )

    return issues


def check_html_structure(document_id: str, source: str, url: str) -> list[SchemaIssue]:
    issues = []

    if not url or str(url).lower() == "nan":
        return issues

    if is_pdf_url(url):
        return issues

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        table_count = len(soup.find_all("table"))
        link_count = len(soup.find_all("a"))
        button_count = len(soup.find_all("button"))

        pdf_links = [
            link.get("href", "")
            for link in soup.find_all("a")
            if ".pdf" in str(link.get("href", "")).lower()
        ]

        pdf_buttons = [
            button.get("onclick", "")
            for button in soup.find_all("button")
            if ".pdf" in str(button.get("onclick", "")).lower()
        ]

        if table_count == 0 and source.lower() != "ecourts":
            issues.append(
                SchemaIssue(
                    document_id=document_id,
                    source=source,
                    url=url,
                    issue_type="changed_table_structure",
                    severity="medium",
                    message="No HTML table found. Table-based crawler selector may be broken.",
                )
            )

        if link_count == 0 and button_count == 0:
            issues.append(
                SchemaIssue(
                    document_id=document_id,
                    source=source,
                    url=url,
                    issue_type="changed_table_structure",
                    severity="high",
                    message="No links or buttons found. Source HTML structure may have changed.",
                )
            )

        if not pdf_links and not pdf_buttons:
            issues.append(
                SchemaIssue(
                    document_id=document_id,
                    source=source,
                    url=url,
                    issue_type="missing_pdf_link",
                    severity="high",
                    message="No PDF link/button found in HTML page.",
                )
            )

    except Exception as error:
        issues.append(
            SchemaIssue(
                document_id=document_id,
                source=source,
                url=url,
                issue_type="source_fetch_error",
                severity="medium",
                message=str(error),
            )
        )

    return issues


def build_report(issues: list[SchemaIssue], total_rows: int) -> str:
    critical = sum(1 for item in issues if item.severity == "critical")
    high = sum(1 for item in issues if item.severity == "high")
    medium = sum(1 for item in issues if item.severity == "medium")

    lines = [
        "# Gap 44 Source Schema Change Detection Report",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Summary",
        "",
        f"- Manifest rows checked: {total_rows}",
        f"- Total schema issues found: {len(issues)}",
        f"- Critical issues: {critical}",
        f"- High issues: {high}",
        f"- Medium issues: {medium}",
        "",
        "## Issue Table",
        "",
        "| Document ID | Source | Severity | Issue Type | URL | Message |",
        "|---|---|---|---|---|---|",
    ]

    for item in issues:
        lines.append(
            f"| {item.document_id} | {item.source} | {item.severity} | "
            f"{item.issue_type} | {item.url} | {item.message} |"
        )

    lines.extend(["", "## Decision", ""])

    if critical > 0 or high > 0:
        lines.append("Crawler schema assumptions need review before large-scale crawling.")
    elif medium > 0:
        lines.append("Crawler may continue, but schema warnings should be reviewed.")
    else:
        lines.append("No schema change detected from current manifest/source checks.")

    return "\n".join(lines)


def run_source_schema_monitor() -> bool:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    issues = []

    for _, row in manifest.iterrows():
        document_id = str(row.get("document_id", ""))
        source = str(row.get("source", ""))
        url = str(row.get("url", ""))

        issues.extend(check_manifest_row(row))
        issues.extend(check_html_structure(document_id, source, url))

    report = build_report(issues, len(manifest))

    SOURCE_SCHEMA_CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_SCHEMA_CHANGES_PATH.write_text(report, encoding="utf-8")

    critical_or_high = [
        item for item in issues if item.severity in {"critical", "high"}
    ]

    print("\nGap 44 Source Schema Change Detection Report")
    print("=" * 70)
    print(f"Manifest rows checked: {len(manifest)}")
    print(f"Schema issues found: {len(issues)}")
    print(f"Critical/high issues: {len(critical_or_high)}")
    print(f"Report saved to: {SOURCE_SCHEMA_CHANGES_PATH}")
    print("=" * 70)

    for item in issues[:10]:
        print(
            f"{item.document_id} | "
            f"{item.severity} | "
            f"{item.issue_type} | "
            f"{item.message}"
        )

    return len(critical_or_high) == 0


def main() -> None:
    run_source_schema_monitor()


if __name__ == "__main__":
    main()