import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import fitz
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from src.config import RAW_CASES_DIR, SEED_URLS_PATH


ECOURTS_URL = "https://judgments.ecourts.gov.in/pdfsearch/"

PDF_PATH_PATTERN = re.compile(
    r"open_pdf\([^)]*['\"]([^'\"]+\.pdf[^'\"]*)['\"]",
    flags=re.IGNORECASE,
)


def is_valid_existing_pdf(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        if path.stat().st_size == 0:
            return False

        first_bytes = path.read_bytes()[:5]

        if not first_bytes.startswith(b"%PDF"):
            return False

        with fitz.open(path) as doc:
            return doc.page_count > 0

    except Exception:
        return False


def clean_pdf_path(raw_path: str) -> str:
    return raw_path.split("#")[0].strip()


def extract_pdf_path_from_onclick(onclick: str | None) -> str | None:
    if not onclick:
        return None

    match = PDF_PATH_PATTERN.search(onclick)

    if not match:
        return None

    return clean_pdf_path(match.group(1))


def extract_pdf_buttons(page) -> list[dict]:
    items = []

    for index, button in enumerate(page.locator("button").all()):
        onclick = button.get_attribute("onclick")
        pdf_path = extract_pdf_path_from_onclick(onclick)

        if not pdf_path:
            continue

        try:
            title = " ".join(button.inner_text(timeout=1000).split())
        except Exception:
            title = "eCourts Judgment"

        items.append(
            {
                "button_index": index,
                "title": title,
                "pdf_path": pdf_path,
                "url": urljoin(ECOURTS_URL, pdf_path),
            }
        )

    return items


def write_seed_urls(items: list[dict]) -> None:
    SEED_URLS_PATH.parent.mkdir(parents=True, exist_ok=True)

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

    with SEED_URLS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, item in enumerate(items, start=1):
            document_id = f"ECOURTS_{index:06d}"

            writer.writerow(
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


def fetch_bytes_inside_browser(page, url: str) -> bytes:
    byte_list = page.evaluate(
        """
        async (url) => {
            const response = await fetch(url, {
                method: "GET",
                credentials: "include",
                cache: "no-store"
            });

            const buffer = await response.arrayBuffer();
            return Array.from(new Uint8Array(buffer));
        }
        """,
        url,
    )

    return bytes(byte_list)


def looks_like_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF")


def close_pdf_viewer(page) -> None:
    try:
        page.keyboard.press("Escape")
        time.sleep(1)
    except Exception:
        pass


def download_pdf_item(page, item: dict, document_id: str, output_path: Path) -> bool:
    if is_valid_existing_pdf(output_path):
        print(f"SKIPPED | {document_id} | already valid PDF")
        return True

    button = page.locator("button").nth(item["button_index"])

    print(f"\nCLICKING | {document_id} | {item['title']}")

    try:
        with page.expect_response(
            lambda response: "/tmp/" in response.url and response.url.lower().endswith(".pdf"),
            timeout=15000,
        ) as response_info:
            button.click()

        response = response_info.value
        tmp_pdf_url = response.url

        print(f"Captured tmp PDF URL: {tmp_pdf_url}")

        content = fetch_bytes_inside_browser(page, tmp_pdf_url)

        if looks_like_pdf(content):
            output_path.write_bytes(content)
            print(f"SUCCESS | {document_id} | size={len(content)} | {output_path}")
            return True

        debug_path = output_path.with_suffix(".debug.html")
        debug_path.write_bytes(content)
        print(f"FAILED | {document_id} | not real PDF | debug={debug_path}")
        return False

    except PlaywrightTimeoutError:
        print(f"FAILED | {document_id} | No /tmp/*.pdf response captured")
        return False

    except Exception as error:
        print(f"FAILED | {document_id} | {error}")
        return False

    finally:
        close_pdf_viewer(page)


def download_all(page, items: list[dict]) -> None:
    RAW_CASES_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for index, item in enumerate(items, start=1):
        document_id = f"ECOURTS_{index:06d}"
        output_path = RAW_CASES_DIR / f"{document_id}.pdf"

        success = download_pdf_item(page, item, document_id, output_path)

        if success:
            success_count += 1

        time.sleep(1)

    print("\nAutomated PDF Save Summary")
    print("=" * 70)
    print(f"Total items: {len(items)}")
    print(f"Valid or saved PDFs: {success_count}")
    print(f"Failed: {len(items) - success_count}")
    print("=" * 70)


def main() -> None:
    print("\nHuman-Assisted eCourts Automated PDF Saver")
    print("=" * 70)
    print("1. Browser will open.")
    print("2. Enter keyword manually.")
    print("3. Type captcha manually.")
    print("4. Click Search manually.")
    print("5. Wait until judgment results appear.")
    print("6. Come back here and press ENTER.")
    print("=" * 70)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False,
            slow_mo=300,
            args=[
                "--disable-extensions",
                "--disable-pdf-extension",
                "--disable-plugins",
            ],
        )

        page = browser.new_page()
        page.goto(ECOURTS_URL, wait_until="domcontentloaded")

        input("\nAfter judgment results appear, press ENTER here...")

        items = extract_pdf_buttons(page)

        if not items:
            print("No PDF buttons found.")
            browser.close()
            return

        print(f"\nPDF buttons found: {len(items)}")

        write_seed_urls(items)
        print(f"Seed URLs saved to: {SEED_URLS_PATH}")

        download_all(page, items)

        browser.close()


if __name__ == "__main__":
    main()
