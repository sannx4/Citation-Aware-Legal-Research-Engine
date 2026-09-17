import csv
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import fitz


# ============================================================
# Project setup
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (
    RAW_CASES_DIR,
    SEED_URLS_PATH,
)


# ============================================================
# Patterns
# ============================================================

YEAR_PATTERN = re.compile(
    r"\b((?:19|20)\d{2})\b"
)

CASE_YEAR_PATTERN = re.compile(
    r"(?:/|\bof\s+)((?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)


# ============================================================
# Helpers
# ============================================================

def clean(value) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() == "nan":
        return ""

    return value


def extract_case_year(title: str) -> str:
    """
    Extract the case-registration year from the case title/number.

    Examples:
        BA/63/2026 -> 2026
        CRA/259/2017 -> 2017
        Appeal No. 14 of 2024 -> 2024

    Note:
        This is the case year, not the judgment/decision year.
    """

    title = clean(title)

    if not title:
        return ""

    match = CASE_YEAR_PATTERN.search(title)

    if match:
        return match.group(1)

    fallback = YEAR_PATTERN.search(title)

    if fallback:
        return fallback.group(1)

    return ""


def extract_first_pages_text(
    pdf_path: Path,
    max_pages: int = 2,
) -> str:
    """
    Extract text from the first few pages of a judgment PDF.
    Court headings are usually present on page 1.
    """

    if not pdf_path.exists():
        return ""

    try:
        parts = []

        with fitz.open(pdf_path) as document:
            page_limit = min(
                max_pages,
                document.page_count,
            )

            for index in range(page_limit):
                text = document[index].get_text("text")

                if text:
                    parts.append(text)

        return "\n".join(parts)

    except Exception:
        return ""


# ============================================================
# Court extraction
# ============================================================

def normalize_court_from_text(
    text: str,
) -> str:
    """
    Detect the originating court from judgment text.

    Only the beginning of the judgment is inspected because later
    paragraphs may mention or cite other courts. Regex matching is
    used because PDF extraction can introduce unusual whitespace.
    """

    if not text:
        return ""

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip().upper()

    # The originating court should normally appear near the beginning
    # of the first page.
    header = normalized[:2000]

    # ---------------------------------------------------------
    # Supreme Court
    # ---------------------------------------------------------
    # Keep this especially strict. A High Court judgment may cite
    # Supreme Court cases later on the same page.
    supreme_header = normalized[:800]

    if re.search(
        r"\bIN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA\b",
        supreme_header,
        flags=re.IGNORECASE,
    ):
        return "Supreme Court of India"

    # ---------------------------------------------------------
    # High Courts
    # ---------------------------------------------------------

    regex_patterns = [
        # Bombay
        (
            r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+BOMBAY",
            "Bombay High Court",
        ),
        (
            r"HIGH\s+COURT\s+OF\s+BOMBAY",
            "Bombay High Court",
        ),
        (
            r"BOMBAY\s+HIGH\s+COURT",
            "Bombay High Court",
        ),

        # Punjab and Haryana
        (
            r"HIGH\s+COURT\s+OF\s+PUNJAB\s*(?:&|AND)\s*HARYANA",
            "Punjab and Haryana High Court",
        ),
        (
            r"PUNJAB\s*(?:&|AND)\s*HARYANA\s+HIGH\s+COURT",
            "Punjab and Haryana High Court",
        ),
        (
            r"HIGH\s+COURT.*?PUNJAB.*?HARYANA.*?CHANDIGARH",
            "Punjab and Haryana High Court",
        ),

        # Calcutta
        (
            r"HIGH\s+COURT\s+AT\s+CALCUTTA",
            "Calcutta High Court",
        ),
        (
            r"HIGH\s+COURT\s+OF\s+CALCUTTA",
            "Calcutta High Court",
        ),
        (
            r"CALCUTTA\s+HIGH\s+COURT",
            "Calcutta High Court",
        ),

        # Gujarat
        (
            r"HIGH\s+COURT\s+OF\s+GUJARAT",
            "Gujarat High Court",
        ),
        (
            r"GUJARAT\s+HIGH\s+COURT",
            "Gujarat High Court",
        ),

        # Gauhati
        (
            r"GAUHATI\s+HIGH\s+COURT",
            "Gauhati High Court",
        ),
        (
            r"HIGH\s+COURT\s+OF\s+ASSAM",
            "Gauhati High Court",
        ),

        # Meghalaya
        (
            r"HIGH\s+COURT\s+OF\s+MEGHALAYA",
            "Meghalaya High Court",
        ),
        (
            r"MEGHALAYA\s+HIGH\s+COURT",
            "Meghalaya High Court",
        ),

        # Jharkhand
        (
            r"HIGH\s+COURT\s+OF\s+JHARKHAND",
            "Jharkhand High Court",
        ),
        (
            r"JHARKHAND\s+HIGH\s+COURT",
            "Jharkhand High Court",
        ),

        # Telangana
        (
            r"HIGH\s+COURT\s+FOR\s+THE\s+STATE\s+OF\s+TELANGANA",
            "Telangana High Court",
        ),
        (
            r"HIGH\s+COURT\s+OF\s+TELANGANA",
            "Telangana High Court",
        ),
        (
            r"TELANGANA\s+HIGH\s+COURT",
            "Telangana High Court",
        ),

        # Andhra Pradesh
        (
            r"HIGH\s+COURT\s+FOR\s+THE\s+STATE\s+OF\s+ANDHRA\s+PRADESH",
            "Andhra Pradesh High Court",
        ),
        (
            r"HIGH\s+COURT\s+OF\s+ANDHRA\s+PRADESH",
            "Andhra Pradesh High Court",
        ),
        (
            r"HIGH\s+COURT\s+OF\s+ANDHRA\s+PRADESH\s+AT\s+AMARAVATI",
            "Andhra Pradesh High Court",
        ),
        (
            r"ANDHRA\s+PRADESH\s+HIGH\s+COURT",
            "Andhra Pradesh High Court",
        ),

        # Himachal Pradesh
        (
            r"HIGH\s+COURT\s+OF\s+HIMACHAL\s+PRADESH",
            "Himachal Pradesh High Court",
        ),
        (
            r"HIMACHAL\s+PRADESH\s+HIGH\s+COURT",
            "Himachal Pradesh High Court",
        ),

        # Patna
        (
            r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+PATNA",
            "Patna High Court",
        ),
        (
            r"PATNA\s+HIGH\s+COURT",
            "Patna High Court",
        ),

        # Delhi
        (
            r"HIGH\s+COURT\s+OF\s+DELHI",
            "Delhi High Court",
        ),
        (
            r"DELHI\s+HIGH\s+COURT",
            "Delhi High Court",
        ),

        # Madras
        (
            r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+MADRAS",
            "Madras High Court",
        ),
        (
            r"MADRAS\s+HIGH\s+COURT",
            "Madras High Court",
        ),

        # Kerala
        (
            r"HIGH\s+COURT\s+OF\s+KERALA",
            "Kerala High Court",
        ),
        (
            r"KERALA\s+HIGH\s+COURT",
            "Kerala High Court",
        ),

        # Karnataka
        (
            r"HIGH\s+COURT\s+OF\s+KARNATAKA",
            "Karnataka High Court",
        ),
        (
            r"KARNATAKA\s+HIGH\s+COURT",
            "Karnataka High Court",
        ),

        # Allahabad
        (
            r"HIGH\s+COURT\s+OF\s+JUDICATURE\s+AT\s+ALLAHABAD",
            "Allahabad High Court",
        ),
        (
            r"ALLAHABAD\s+HIGH\s+COURT",
            "Allahabad High Court",
        ),

        # Rajasthan
        (
            r"RAJASTHAN\s+HIGH\s+COURT",
            "Rajasthan High Court",
        ),
        (
            r"HIGH\s+COURT\s+OF\s+RAJASTHAN",
            "Rajasthan High Court",
        ),

        # Madhya Pradesh
        (
            r"HIGH\s+COURT\s+OF\s+MADHYA\s+PRADESH",
            "Madhya Pradesh High Court",
        ),
        (
            r"MADHYA\s+PRADESH\s+HIGH\s+COURT",
            "Madhya Pradesh High Court",
        ),

        # Chhattisgarh
        (
            r"HIGH\s+COURT\s+OF\s+CHHATTISGARH",
            "Chhattisgarh High Court",
        ),
        (
            r"CHHATTISGARH\s+HIGH\s+COURT",
            "Chhattisgarh High Court",
        ),

        # Uttarakhand
        (
            r"HIGH\s+COURT\s+OF\s+UTTARAKHAND",
            "Uttarakhand High Court",
        ),
        (
            r"UTTARAKHAND\s+HIGH\s+COURT",
            "Uttarakhand High Court",
        ),

        # Orissa
        (
            r"HIGH\s+COURT\s+OF\s+ORISSA",
            "Orissa High Court",
        ),
        (
            r"ORISSA\s+HIGH\s+COURT",
            "Orissa High Court",
        ),

        # Handles broken extraction such as:
        # "IN THE HIGH ... H COURT OF ORISSA AT CUTTAC"
        (
            r"H\s+COURT\s+OF\s+ORISSA(?:\s+AT\s+CUTTAC(?:K)?)?",
            "Orissa High Court",
        ),

        # Sikkim
        (
            r"HIGH\s+COURT\s+OF\s+SIKKIM",
            "Sikkim High Court",
        ),
        (
            r"SIKKIM\s+HIGH\s+COURT",
            "Sikkim High Court",
        ),

        # Tripura
        (
            r"HIGH\s+COURT\s+OF\s+TRIPURA",
            "Tripura High Court",
        ),
        (
            r"TRIPURA\s+HIGH\s+COURT",
            "Tripura High Court",
        ),

        # Manipur
        (
            r"HIGH\s+COURT\s+OF\s+MANIPUR",
            "Manipur High Court",
        ),
        (
            r"MANIPUR\s+HIGH\s+COURT",
            "Manipur High Court",
        ),

        # Jammu & Kashmir and Ladakh
        (
            r"HIGH\s+COURT\s+OF\s+JAMMU\s*(?:&|AND)\s*KASHMIR"
            r"\s*(?:&|AND)?\s*LADAKH",
            "Jammu & Kashmir and Ladakh High Court",
        ),
        (
            r"HIGH\s+COURT\s+OF\s+JAMMU\s*(?:&|AND)\s*KASHMIR",
            "Jammu & Kashmir and Ladakh High Court",
        ),
    ]

    # IMPORTANT:
    # Search only the header, not the entire judgment.
    for pattern, court_name in regex_patterns:
        if re.search(
            pattern,
            header,
            flags=re.IGNORECASE,
        ):
            return court_name

    return ""

def infer_court_from_context(
    title: str,
    pdf_text: str,
) -> str:
    """
    Conservative fallback used only when no explicit court heading
    is found in the PDF text.

    This is useful for short orders where the first page may omit
    the court heading but the jurisdiction/state is clear.
    """

    context = " ".join(
        [
            clean(title),
            clean(pdf_text),
        ]
    ).upper()

    fallback_rules = [
        (
            "STATE OF WEST BENGAL",
            "Calcutta High Court",
        ),
        (
            "STATE OF HARYANA",
            "Punjab and Haryana High Court",
        ),
        (
            "STATE OF PUNJAB",
            "Punjab and Haryana High Court",
        ),
        (
            "STATE OF MAHARASHTRA",
            "Bombay High Court",
        ),
        (
            "STATE OF GUJARAT",
            "Gujarat High Court",
        ),
        (
            "STATE OF JHARKHAND",
            "Jharkhand High Court",
        ),
        (
            "HARYANA AT CHANDIGARH",
            "Punjab and Haryana High Court",
        ),
        (
            "PUNJAB AND HARYANA AT CHANDIGARH",
            "Punjab and Haryana High Court",
        ),
        (
            "STATE OF TELANGANA",
            "Telangana High Court",
        ),
        (
            "STATE OF ANDHRA PRADESH",
            "Andhra Pradesh High Court",
        ),
        (
            "STATE OF HIMACHAL PRADESH",
            "Himachal Pradesh High Court",
        ),
        (
            "STATE OF BIHAR",
            "Patna High Court",
        ),
        (
            "LOCATION: OHC",
            "Orissa High Court",
        ),
        (
            "STATE OF MEGHALAYA",
            "Meghalaya High Court",
        ),
        (
            "STATE OF ASSAM",
            "Gauhati High Court",
        ),
        (
            "STATE OF GOA",
            "Bombay High Court",
        ),
    ]

    for phrase, court_name in fallback_rules:
        if phrase in context:
            return court_name

    return ""


# ============================================================
# Seed metadata file
# ============================================================

def load_seed_rows() -> list[dict]:
    if not SEED_URLS_PATH.exists():
        raise FileNotFoundError(
            f"Seed metadata not found: {SEED_URLS_PATH}"
        )

    with SEED_URLS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def create_backup() -> Path:
    backup_dir = (
        SEED_URLS_PATH.parent
        / "backups"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        backup_dir
        / (
            "seed_urls_before_metadata_"
            f"{timestamp}.csv"
        )
    )

    shutil.copy2(
        SEED_URLS_PATH,
        backup_path,
    )

    return backup_path


def write_seed_rows(
    rows: list[dict],
) -> None:
    fieldnames = [
        "document_id",
        "source",
        "document_type",
        "title",
        "court",
        "year",
        "url",
        "file_name",
    ]

    with SEED_URLS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }
            )


# ============================================================
# Metadata backfill
# ============================================================

def backfill_metadata(
    rows: list[dict],
) -> tuple[
    list[dict],
    list[dict],
]:
    """
    Backfill:
        - case year
        - specific court

    Existing metadata is preserved whenever extraction cannot
    confidently improve it.
    """

    updated_rows = []
    report_rows = []

    for row in rows:
        document_id = clean(
            row.get(
                "document_id"
            )
        )

        file_name = clean(
            row.get(
                "file_name"
            )
        )

        title = clean(
            row.get(
                "title"
            )
        )

        old_year = clean(
            row.get(
                "year"
            )
        )

        old_court = clean(
            row.get(
                "court"
            )
        )

        pdf_path = (
            RAW_CASES_DIR
            / file_name
        )

        # ----------------------------------------------------
        # Year
        # ----------------------------------------------------

        extracted_year = (
            extract_case_year(
                title
            )
        )

        new_year = (
            extracted_year
            or old_year
        )

        # ----------------------------------------------------
        # Read PDF text BEFORE court inference
        # ----------------------------------------------------

        pdf_text = (
            extract_first_pages_text(
                pdf_path
            )
        )

        # ----------------------------------------------------
        # Court - primary source: judgment heading
        # ----------------------------------------------------

        extracted_court = (
            normalize_court_from_text(
                pdf_text
            )
        )

        # ----------------------------------------------------
        # Court - secondary source: conservative context
        # ----------------------------------------------------

        if not extracted_court:
            extracted_court = (
                infer_court_from_context(
                    title,
                    pdf_text,
                )
            )

        # Prefer extracted specific court.
        # If nothing can be extracted, preserve old value.
        new_court = (
            extracted_court
            or old_court
        )

        # ----------------------------------------------------
        # Updated seed row
        # ----------------------------------------------------

        updated = dict(row)

        updated["year"] = new_year
        updated["court"] = new_court

        updated_rows.append(
            updated
        )

        # ----------------------------------------------------
        # Audit report row
        # ----------------------------------------------------

        report_rows.append(
            {
                "document_id":
                    document_id,

                "file_name":
                    file_name,

                "old_year":
                    old_year,

                "new_year":
                    new_year,

                "old_court":
                    old_court,

                "new_court":
                    new_court,

                "year_changed":
                    old_year != new_year,

                "court_changed":
                    old_court != new_court,

                "pdf_exists":
                    pdf_path.exists(),
            }
        )

    return (
        updated_rows,
        report_rows,
    )


# ============================================================
# Backfill report
# ============================================================

def write_report(
    rows: list[dict],
) -> Path:
    report_path = (
        ROOT_DIR
        / "reports"
        / "metadata_backfill_report.csv"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "document_id",
        "file_name",
        "old_year",
        "new_year",
        "old_court",
        "new_court",
        "year_changed",
        "court_changed",
        "pdf_exists",
    ]

    with report_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return report_path


# ============================================================
# Main
# ============================================================

def main() -> None:
    rows = load_seed_rows()

    # Always preserve the previous seed file.
    backup_path = (
        create_backup()
    )

    (
        updated_rows,
        report_rows,
    ) = backfill_metadata(
        rows
    )

    write_seed_rows(
        updated_rows
    )

    report_path = (
        write_report(
            report_rows
        )
    )

    total = len(
        updated_rows
    )

    years_present = sum(
        1
        for row in updated_rows
        if clean(
            row.get("year")
        )
    )

    courts_present = sum(
        1
        for row in updated_rows
        if clean(
            row.get("court")
        )
    )

    year_changes = sum(
        1
        for row in report_rows
        if row["year_changed"]
    )

    court_changes = sum(
        1
        for row in report_rows
        if row["court_changed"]
    )

    generic_courts = sum(
        1
        for row in updated_rows
        if clean(
            row.get("court")
        )
        in {
            "Indian Court",
            "Indian High Court",
        }
    )

    print(
        "\nCorpus Metadata Backfill Report"
    )

    print("=" * 70)

    print(
        f"Seed rows processed: "
        f"{total}"
    )

    print(
        f"Rows with year metadata: "
        f"{years_present}/{total}"
    )

    print(
        f"Years updated: "
        f"{year_changes}"
    )

    print(
        f"Rows with court metadata: "
        f"{courts_present}/{total}"
    )

    print(
        f"Court values updated: "
        f"{court_changes}"
    )

    print(
        f"Generic court values remaining: "
        f"{generic_courts}"
    )

    print(
        f"Backup: "
        f"{backup_path}"
    )

    print(
        f"Report: "
        f"{report_path}"
    )

    print("=" * 70)

    print(
        "\nUpdated metadata samples"
    )

    print("-" * 70)

    changed_rows = [
        row
        for row in report_rows
        if (
            row["year_changed"]
            or row["court_changed"]
        )
    ]

    if not changed_rows:
        print(
            "No metadata changes were required."
        )

    else:
        for row in changed_rows:
            print(
                f"{row['document_id']} | "
                f"year: "
                f"{row['old_year'] or '-'}"
                f" -> "
                f"{row['new_year'] or '-'}"
                f" | court: "
                f"{row['old_court'] or '-'}"
                f" -> "
                f"{row['new_court'] or '-'}"
            )


if __name__ == "__main__":
    main()
