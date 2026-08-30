import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

ECOURTS_URL = "https://judgments.ecourts.gov.in/pdfsearch/"


def main() -> None:
    print("\nInspect eCourts Results Page")
    print("=" * 70)
    print("1. Browser will open.")
    print("2. Search keyword manually.")
    print("3. Type captcha manually.")
    print("4. Click Search.")
    print("5. Wait for results.")
    print("6. Come back and press ENTER.")
    print("=" * 70)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False, slow_mo=300)
        page = browser.new_page()
        page.goto(ECOURTS_URL, wait_until="domcontentloaded")

        input("\nAfter results appear, press ENTER here...")

        print("\nPAGE INFO")
        print("=" * 70)
        print(f"Title: {page.title()}")
        print(f"URL: {page.url}")
        print(f"Links: {page.locator('a').count()}")
        print(f"Buttons: {page.locator('button').count()}")
        print(f"Inputs: {page.locator('input').count()}")
        print(f"Forms: {page.locator('form').count()}")

        print("\nBUTTONS")
        print("=" * 70)
        buttons = page.locator("button").all()

        for i, button in enumerate(buttons, start=1):
            try:
                text = button.inner_text(timeout=1000).strip()
            except Exception:
                text = ""

            onclick = button.get_attribute("onclick")
            button_type = button.get_attribute("type")
            button_id = button.get_attribute("id")
            button_class = button.get_attribute("class")

            print(f"\nButton {i}")
            print(f"text    : {text}")
            print(f"id      : {button_id}")
            print(f"class   : {button_class}")
            print(f"type    : {button_type}")
            print(f"onclick : {onclick}")

        print("\nLINKS")
        print("=" * 70)
        links = page.locator("a").all()

        for i, link in enumerate(links, start=1):
            try:
                text = link.inner_text(timeout=1000).strip()
            except Exception:
                text = ""

            href = link.get_attribute("href")
            onclick = link.get_attribute("onclick")
            link_id = link.get_attribute("id")
            link_class = link.get_attribute("class")

            print(f"\nLink {i}")
            print(f"text    : {text[:100]}")
            print(f"id      : {link_id}")
            print(f"class   : {link_class}")
            print(f"href    : {href}")
            print(f"onclick : {onclick}")

        print("\nINPUTS")
        print("=" * 70)
        inputs = page.locator("input").all()

        for i, item in enumerate(inputs, start=1):
            print(f"\nInput {i}")
            print(f"name  : {item.get_attribute('name')}")
            print(f"id    : {item.get_attribute('id')}")
            print(f"type  : {item.get_attribute('type')}")
            print(f"value : {item.get_attribute('value')}")

        browser.close()


if __name__ == "__main__":
    main()