import csv
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from src.config import SEED_URLS_PATH


ECOURTS_URL = "https://judgments.ecourts.gov.in/pdfsearch/"

FIELDNAMES = [
    "document_id",
    "source",
    "document_type",
    "title",
    "court",
    "year",
    "url",
    "file_name",
]


PDF_PATH_PATTERN = re.compile(
    r"open_pdf\([^)]*['\"]([^'\"]+\.pdf[^'\"]*)['\"]",
    flags=re.IGNORECASE,
)


def ensure_seed_file_exists() -> None:
    SEED_URLS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not SEED_URLS_PATH.exists():
        with SEED_URLS_PATH.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()


def existing_urls() -> set[str]:
    ensure_seed_file_exists()

    urls = set()

    with SEED_URLS_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("url"):
                urls.add(row["url"])

    return urls


def append_rows(rows: list[dict]) -> None:
    ensure_seed_file_exists()

    with SEED_URLS_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        for row in rows:
            writer.writerow(row)


def clean_pdf_path(raw_path: str) -> str:
    return raw_path.split("#")[0].strip()


def extract_pdf_path_from_onclick(onclick: str | None) -> str | None:
    if not onclick:
        return None

    match = PDF_PATH_PATTERN.search(onclick)

    if not match:
        return None

    return clean_pdf_path(match.group(1))


def extract_pdf_links_from_buttons(page) -> list[dict]:
    results = []

    buttons = page.locator("button").all()

    for button in buttons:
        onclick = button.get_attribute("onclick")
        pdf_path = extract_pdf_path_from_onclick(onclick)

        if not pdf_path:
            continue

        try:
            title = " ".join(button.inner_text(timeout=1000).split())
        except Exception:
            title = "eCourts Judgment"

        full_url = urljoin(ECOURTS_URL, pdf_path)

        results.append(
            {
                "title": title or "eCourts Judgment",
                "url": full_url,
            }
        )

    unique = {}
    for item in results:
        unique[item["url"]] = item

    return list(unique.values())


def build_rows(items: list[dict], start_index: int = 1) -> list[dict]:
    rows = []

    for index, item in enumerate(items, start=start_index):
        document_id = f"ECOURTS_{index:06d}"

        rows.append(
            {
                "document_id": document_id,
                "source": "ecourts",
                "document_type": "judgment",
                "title": item["title"],
                "court": "Indian Court",
                "year": "",
                "url": item["url"],
                "file_name": f"{document_id}.pdf",
            }
        )

    return rows


def print_page_debug(page) -> None:
    print("\nPage Debug")
    print("=" * 70)
    print(f"Page title: {page.title()}")
    print(f"Current URL: {page.url}")
    print(f"Total links on page: {page.locator('a').count()}")
    print(f"Total buttons on page: {page.locator('button').count()}")
    print("=" * 70)


def main() -> None:
    print("\nHuman-Assisted eCourts Crawler")
    print("=" * 70)
    print("A browser will open.")
    print("1. Enter keyword/search details.")
    print("2. Type captcha manually.")
    print("3. Click Search manually.")
    print("4. Wait until judgment results appear.")
    print("5. Come back here and press ENTER.")
    print("=" * 70)

    already_seen = existing_urls()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False,
            slow_mo=300,
        )

        page = browser.new_page()
        page.goto(ECOURTS_URL, wait_until="domcontentloaded")

        input("\nAfter judgment search results appear in browser, press ENTER here...")

        print_page_debug(page)

        items = extract_pdf_links_from_buttons(page)

        new_items = [
            item for item in items
            if item["url"] not in already_seen
        ]

        if not new_items:
            print("\nNo new judgment PDF paths found from buttons.")
            print("Possible reason: result buttons were not loaded yet.")
            browser.close()
            return

        rows = build_rows(new_items, start_index=len(already_seen) + 1)
        append_rows(rows)

        print("\nJudgment PDF Paths Extracted")
        print("=" * 70)
        print(f"Total PDF paths found: {len(items)}")
        print(f"New rows added: {len(rows)}")
        print(f"Saved to: {SEED_URLS_PATH}")
        print("=" * 70)

        for row in rows[:10]:
            print(f"{row['document_id']} | {row['title']} | {row['url']}")

        browser.close()


if __name__ == "__main__":
    main()
