import sys
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import MANIFEST_PATH, CRAWL_COMPLIANCE_REPORT_PATH


USER_AGENT = "LegalResearchEngineBot"


@dataclass
class RobotsCheckResult:
    domain: str
    robots_url: str
    url: str
    allowed: bool
    crawl_delay: str
    status: str
    source_restriction: str
    error: str


def get_domain(url: str) -> str:
    return urlparse(str(url)).netloc.lower()


def get_robots_url(url: str) -> str:
    parsed = urlparse(str(url))
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def get_source_restriction(domain: str) -> str:
    if "ecourts.gov.in" in domain:
        return "Manual captcha/search must be respected. Do not bypass captcha. Download only after lawful user-assisted access."

    if "sci.gov.in" in domain:
        return "Use polite crawling. Prefer official public PDFs and avoid aggressive scraping."

    if "indiacode.nic.in" in domain:
        return "Use polite crawling. Prefer official public statute pages."

    return "Unknown source policy. Manual review recommended."


def check_single_url(url: str) -> RobotsCheckResult:
    domain = get_domain(url)
    robots_url = get_robots_url(url)

    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = requests.get(robots_url, timeout=15)

        if response.status_code >= 400:
            return RobotsCheckResult(
                domain=domain,
                robots_url=robots_url,
                url=url,
                allowed=False,
                crawl_delay="unknown",
                status=f"robots_fetch_failed_{response.status_code}",
                source_restriction=get_source_restriction(domain),
                error=f"Could not fetch robots.txt: HTTP {response.status_code}",
            )

        parser.parse(response.text.splitlines())

        allowed = parser.can_fetch(USER_AGENT, url)
        crawl_delay = parser.crawl_delay(USER_AGENT)

        if crawl_delay is None:
            crawl_delay = parser.crawl_delay("*")

        return RobotsCheckResult(
            domain=domain,
            robots_url=robots_url,
            url=url,
            allowed=allowed,
            crawl_delay=str(crawl_delay) if crawl_delay else "not_specified",
            status="allowed" if allowed else "disallowed",
            source_restriction=get_source_restriction(domain),
            error="",
        )

    except Exception as error:
        return RobotsCheckResult(
            domain=domain,
            robots_url=robots_url,
            url=url,
            allowed=False,
            crawl_delay="unknown",
            status="robots_check_error",
            source_restriction=get_source_restriction(domain),
            error=str(error),
        )


def build_compliance_report(results: list[RobotsCheckResult]) -> str:
    total = len(results)
    allowed_count = sum(1 for item in results if item.allowed)
    disallowed_count = total - allowed_count

    lines = [
        "# Gap 42 Robots / Terms Compliance Report",
        "",
        "## Summary",
        "",
        f"- Total URLs checked: {total}",
        f"- Allowed by robots.txt: {allowed_count}",
        f"- Disallowed / failed / manual review needed: {disallowed_count}",
        "",
        "## Compliance Table",
        "",
        "| Domain | Status | Crawl Delay | URL | Source Restriction | Error |",
        "|---|---|---|---|---|---|",
    ]

    for item in results:
        lines.append(
            f"| {item.domain} | {item.status} | {item.crawl_delay} | "
            f"{item.url} | {item.source_restriction} | {item.error} |"
        )

    lines.extend(
        [
            "",
            "## Policy Notes",
            "",
            "- Do not bypass captcha.",
            "- Do not overload official court websites.",
            "- Respect robots.txt decisions.",
            "- Use rate limiting before every request.",
            "- Prefer official public PDFs and public statute pages.",
            "",
            "## Decision",
            "",
        ]
    )

    if disallowed_count == 0:
        lines.append("Crawler may proceed with rate limiting.")
    else:
        lines.append(
            "Crawler should not automatically download disallowed or unchecked URLs. "
            "Manual review is required."
        )

    return "\n".join(lines)


def run_compliance_check() -> bool:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)

    urls = [
        str(url)
        for url in manifest["url"].dropna().unique().tolist()
        if str(url).strip()
    ]

    results = [check_single_url(url) for url in urls]

    report = build_compliance_report(results)

    CRAWL_COMPLIANCE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CRAWL_COMPLIANCE_REPORT_PATH.write_text(report, encoding="utf-8")

    total = len(results)
    allowed_count = sum(1 for item in results if item.allowed)
    blocked_count = total - allowed_count

    print("\nGap 42 Robots / Terms Compliance Report")
    print("=" * 70)
    print(f"URLs checked: {total}")
    print(f"Allowed: {allowed_count}")
    print(f"Blocked/manual review: {blocked_count}")
    print(f"Report saved to: {CRAWL_COMPLIANCE_REPORT_PATH}")
    print("=" * 70)

    for item in results[:10]:
        print(f"{item.domain} | {item.status} | {item.url}")

    return blocked_count == 0


def main() -> None:
    run_compliance_check()


if __name__ == "__main__":
    main()
