import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import fitz
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


# ============================================================
# Project setup
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (
    RAW_CASES_DIR,
    SEED_URLS_PATH,
)


# ============================================================
# Constants
# ============================================================

ECOURTS_URL = (
    "https://judgments.ecourts.gov.in/pdfsearch/"
)

PDF_PATH_PATTERN = re.compile(
    r"""open_pdf\([^)]*['"]([^'"]+\.pdf[^'"]*)['"]""",
    flags=re.IGNORECASE,
)

ECOURTS_ID_PATTERN = re.compile(
    r"^ECOURTS_(\d+)$",
    flags=re.IGNORECASE,
)

YEAR_PATTERN = re.compile(
    r"\b((?:19|20)\d{2})\b"
)


# ============================================================
# PDF validation
# ============================================================

def is_valid_existing_pdf(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        if path.stat().st_size <= 0:
            return False

        with path.open("rb") as file:
            first_bytes = file.read(5)

        if not first_bytes.startswith(b"%PDF"):
            return False

        with fitz.open(path) as document:
            return document.page_count > 0

    except Exception:
        return False


# ============================================================
# Helpers
# ============================================================

def clean_pdf_path(raw_path: str) -> str:
    return raw_path.split("#")[0].strip()


def normalize_text(value: str) -> str:
    return " ".join(
        str(value)
        .lower()
        .strip()
        .split()
    )


def extract_year_from_title(
    title: str,
) -> str:
    """
    Examples:

    BA/63/2026
        -> 2026

    CRA/259/2017
        -> 2017

    CR.A/115/2020
        -> 2020
    """

    matches = YEAR_PATTERN.findall(
        title or ""
    )

    if not matches:
        return ""

    # Usually the first year appearing in the
    # case number is the relevant case year.
    return matches[0]


def extract_pdf_path_from_onclick(
    onclick: str | None,
) -> str | None:

    if not onclick:
        return None

    match = PDF_PATH_PATTERN.search(
        onclick
    )

    if not match:
        return None

    return clean_pdf_path(
        match.group(1)
    )


# ============================================================
# Extract result buttons
# ============================================================

def extract_pdf_buttons(
    page,
) -> list[dict]:

    items = []

    buttons = page.locator(
        "button"
    ).all()

    for index, button in enumerate(
        buttons
    ):

        onclick = button.get_attribute(
            "onclick"
        )

        pdf_path = (
            extract_pdf_path_from_onclick(
                onclick
            )
        )

        if not pdf_path:
            continue

        try:
            title = " ".join(
                button.inner_text(
                    timeout=1000
                ).split()
            )

        except Exception:
            title = (
                "eCourts Judgment"
            )

        year = extract_year_from_title(
            title
        )

        items.append(
            {
                "button_index": index,
                "title": title,
                "year": year,
                "pdf_path": pdf_path,
                "url": urljoin(
                    ECOURTS_URL,
                    pdf_path,
                ),
            }
        )

    return items


# ============================================================
# Existing seed metadata
# ============================================================

def load_existing_seed_rows() -> list[dict]:
    if not SEED_URLS_PATH.exists():
        return []

    rows = []

    with SEED_URLS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# Persistent document IDs
# ============================================================

def get_highest_existing_id() -> int:
    """
    Determine the highest ECOURTS ID from both:

    1. seed_urls.csv
    2. actual PDF filenames

    This prevents accidental overwriting even if one source
    is temporarily incomplete.
    """

    highest = 0

    # --------------------------------------------------------
    # Check seed_urls.csv
    # --------------------------------------------------------

    for row in load_existing_seed_rows():

        document_id = str(
            row.get(
                "document_id",
                "",
            )
        ).strip()

        match = ECOURTS_ID_PATTERN.match(
            document_id
        )

        if match:
            highest = max(
                highest,
                int(
                    match.group(1)
                ),
            )

    # --------------------------------------------------------
    # Check actual PDF files
    # --------------------------------------------------------

    if RAW_CASES_DIR.exists():

        for path in RAW_CASES_DIR.glob(
            "ECOURTS_*.pdf"
        ):

            document_id = (
                path.stem
            )

            match = (
                ECOURTS_ID_PATTERN.match(
                    document_id
                )
            )

            if match:
                highest = max(
                    highest,
                    int(
                        match.group(1)
                    ),
                )

    return highest


def get_next_document_index() -> int:
    return (
        get_highest_existing_id()
        + 1
    )


# ============================================================
# Duplicate detection
# ============================================================

def get_existing_identity_sets():
    rows = load_existing_seed_rows()

    existing_titles = set()
    existing_urls = set()

    for row in rows:

        title = normalize_text(
            row.get(
                "title",
                "",
            )
        )

        url = str(
            row.get(
                "url",
                "",
            )
        ).strip()

        if title:
            existing_titles.add(
                title
            )

        if url:
            existing_urls.add(
                url
            )

    return (
        existing_titles,
        existing_urls,
    )


def prepare_batch_items(
    items: list[dict],
) -> list[dict]:

    (
        existing_titles,
        existing_urls,
    ) = get_existing_identity_sets()

    next_index = (
        get_next_document_index()
    )

    prepared = []

    for item in items:

        title_key = normalize_text(
            item["title"]
        )

        url = item["url"]

        # ----------------------------------------------------
        # Skip previously registered judgments
        # ----------------------------------------------------

        if (
            title_key
            and title_key
            in existing_titles
        ):
            print(
                "SKIPPING EXISTING TITLE | "
                f"{item['title']}"
            )

            continue

        if (
            url
            and url
            in existing_urls
        ):
            print(
                "SKIPPING EXISTING URL | "
                f"{item['title']}"
            )

            continue

        # ----------------------------------------------------
        # Assign persistent ID
        # ----------------------------------------------------

        document_id = (
            f"ECOURTS_{next_index:06d}"
        )

        prepared_item = {
            **item,
            "document_id":
                document_id,
            "file_name":
                f"{document_id}.pdf",
        }

        prepared.append(
            prepared_item
        )

        if title_key:
            existing_titles.add(
                title_key
            )

        if url:
            existing_urls.add(
                url
            )

        next_index += 1

    return prepared


# ============================================================
# Append metadata
# ============================================================

def append_seed_row(
    item: dict,
) -> None:

    SEED_URLS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    file_exists = (
        SEED_URLS_PATH.exists()
        and
        SEED_URLS_PATH.stat().st_size
        > 0
    )

    with SEED_URLS_PATH.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "document_id":
                    item["document_id"],

                "source":
                    "ecourts",

                "document_type":
                    "judgment",

                "title":
                    item["title"],

                # We know these are High Court
                # judgment-search results.
                # Specific HC extraction can be
                # implemented separately.
                "court":
                    "Indian High Court",

                "year":
                    item.get(
                        "year",
                        "",
                    ),

                "url":
                    item["url"],

                "file_name":
                    item["file_name"],
            }
        )


# ============================================================
# Browser-side PDF download
# ============================================================

def fetch_bytes_inside_browser(
    page,
    url: str,
) -> bytes:

    byte_list = page.evaluate(
        """
        async (url) => {

            const response = await fetch(
                url,
                {
                    method: "GET",
                    credentials: "include",
                    cache: "no-store"
                }
            );

            const buffer =
                await response.arrayBuffer();

            return Array.from(
                new Uint8Array(buffer)
            );
        }
        """,
        url,
    )

    return bytes(
        byte_list
    )


def looks_like_pdf(
    content: bytes,
) -> bool:

    return content.startswith(
        b"%PDF"
    )


def close_pdf_viewer(
    page,
) -> None:

    try:
        page.keyboard.press(
            "Escape"
        )

        time.sleep(1)

    except Exception:
        pass


# ============================================================
# Download one judgment
# ============================================================

def download_pdf_item(
    page,
    item: dict,
    output_path: Path,
) -> bool:

    document_id = (
        item["document_id"]
    )

    # --------------------------------------------------------
    # Existing valid file
    # --------------------------------------------------------

    if is_valid_existing_pdf(
        output_path
    ):

        print(
            f"SKIPPED | "
            f"{document_id} | "
            "already valid PDF"
        )

        return True

    button = (
        page.locator("button")
        .nth(
            item["button_index"]
        )
    )

    print(
        f"\nCLICKING | "
        f"{document_id} | "
        f"{item['title']}"
    )

    try:

        with page.expect_response(
            lambda response:
                "/tmp/" in response.url
                and
                response.url
                .lower()
                .endswith(".pdf"),
            timeout=15000,
        ) as response_info:

            button.click()

        response = (
            response_info.value
        )

        tmp_pdf_url = (
            response.url
        )

        print(
            "Captured tmp PDF URL: "
            f"{tmp_pdf_url}"
        )

        content = (
            fetch_bytes_inside_browser(
                page,
                tmp_pdf_url,
            )
        )

        # ----------------------------------------------------
        # Validate before permanent save
        # ----------------------------------------------------

        if not looks_like_pdf(
            content
        ):

            debug_path = (
                output_path.with_suffix(
                    ".debug.html"
                )
            )

            debug_path.write_bytes(
                content
            )

            print(
                f"FAILED | "
                f"{document_id} | "
                "not real PDF | "
                f"debug={debug_path}"
            )

            return False

        # ----------------------------------------------------
        # Atomic-ish save through temporary file
        # ----------------------------------------------------

        temp_path = (
            output_path.with_suffix(
                ".pdf.part"
            )
        )

        temp_path.write_bytes(
            content
        )

        # Extra validation before rename.
        if not looks_like_pdf(
            temp_path.read_bytes()[:5]
        ):

            temp_path.unlink(
                missing_ok=True
            )

            print(
                f"FAILED | "
                f"{document_id} | "
                "temporary file validation failed"
            )

            return False

        temp_path.replace(
            output_path
        )

        print(
            f"SUCCESS | "
            f"{document_id} | "
            f"size={len(content)} | "
            f"{output_path}"
        )

        return True

    except PlaywrightTimeoutError:

        print(
            f"FAILED | "
            f"{document_id} | "
            "No /tmp/*.pdf response captured"
        )

        return False

    except Exception as error:

        print(
            f"FAILED | "
            f"{document_id} | "
            f"{error}"
        )

        return False

    finally:

        close_pdf_viewer(
            page
        )


# ============================================================
# Download batch
# ============================================================

def download_all(
    page,
    items: list[dict],
) -> None:

    RAW_CASES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_count = 0
    failure_count = 0

    for item in items:

        document_id = (
            item["document_id"]
        )

        output_path = (
            RAW_CASES_DIR
            /
            item["file_name"]
        )

        success = download_pdf_item(
            page,
            item,
            output_path,
        )

        if success:

            success_count += 1

            # -----------------------------------------------
            # Register metadata ONLY after successful PDF
            # -----------------------------------------------

            append_seed_row(
                item
            )

        else:

            failure_count += 1

        time.sleep(1)

    print(
        "\nAutomated PDF Save Summary"
    )

    print("=" * 70)

    print(
        f"New items attempted: "
        f"{len(items)}"
    )

    print(
        f"Successfully saved: "
        f"{success_count}"
    )

    print(
        f"Failed: "
        f"{failure_count}"
    )

    print(
        "Next available ID: "
        f"ECOURTS_"
        f"{get_next_document_index():06d}"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "\nHuman-Assisted eCourts "
        "Automated PDF Saver"
    )

    print("=" * 70)

    print(
        "Existing highest ECOURTS ID: "
        f"{get_highest_existing_id()}"
    )

    print(
        "Next document ID: "
        f"ECOURTS_"
        f"{get_next_document_index():06d}"
    )

    print("-" * 70)

    print("1. Browser will open.")
    print("2. Enter keyword manually.")
    print("3. Type captcha manually.")
    print("4. Click Search manually.")
    print(
        "5. Wait until judgment "
        "results appear."
    )
    print(
        "6. Come back here and "
        "press ENTER."
    )

    print("=" * 70)

    with sync_playwright() as playwright:

        browser = (
            playwright.chromium.launch(
                headless=False,
                slow_mo=300,
                args=[
                    "--disable-extensions",
                    "--disable-pdf-extension",
                    "--disable-plugins",
                ],
            )
        )

        page = browser.new_page()

        page.goto(
            ECOURTS_URL,
            wait_until=(
                "domcontentloaded"
            ),
        )

        input(
            "\nAfter judgment results "
            "appear, press ENTER here..."
        )

        items = extract_pdf_buttons(
            page
        )

        if not items:

            print(
                "No PDF buttons found."
            )

            browser.close()

            return

        print(
            f"\nPDF buttons found: "
            f"{len(items)}"
        )

        # ----------------------------------------------------
        # Assign new IDs and remove results already collected
        # ----------------------------------------------------

        batch_items = (
            prepare_batch_items(
                items
            )
        )

        if not batch_items:

            print(
                "\nNo new judgments "
                "found in this result set."
            )

            browser.close()

            return

        print(
            f"New judgments to download: "
            f"{len(batch_items)}"
        )

        print(
            "Assigned ID range: "
            f"{batch_items[0]['document_id']}"
            " -> "
            f"{batch_items[-1]['document_id']}"
        )

        download_all(
            page,
            batch_items,
        )

        print(
            f"\nSeed metadata file: "
            f"{SEED_URLS_PATH}"
        )

        browser.close()


if __name__ == "__main__":
    main()