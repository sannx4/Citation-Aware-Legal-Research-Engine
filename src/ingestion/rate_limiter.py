import csv
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import CRAWL_RATE_REPORT_PATH
from src.ingestion.crawler_policy import get_policy_for_url


class RateLimiter:
    def __init__(self) -> None:
        self.last_request_time = defaultdict(float)
        self.request_history = defaultdict(deque)

    def log_event(
        self,
        document_id: str,
        url: str,
        action: str,
        waited_seconds: float = 0.0,
    ) -> None:
        policy = get_policy_for_url(url)

        self.write_report(
            {
                "document_id": document_id,
                "url": url,
                "domain": policy["domain"],
                "delay_seconds": policy["delay_seconds"],
                "max_requests_per_minute": policy["max_requests_per_minute"],
                "waited_seconds": round(waited_seconds, 2),
                "action": action,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def wait(self, document_id: str, url: str) -> None:
        policy = get_policy_for_url(url)

        domain = policy["domain"]
        delay_seconds = policy["delay_seconds"]
        max_requests_per_minute = policy["max_requests_per_minute"]

        now = time.time()

        elapsed = now - self.last_request_time[domain]
        delay_wait = max(0, delay_seconds - elapsed)

        history = self.request_history[domain]

        while history and now - history[0] > 60:
            history.popleft()

        rpm_wait = 0.0

        if len(history) >= max_requests_per_minute:
            oldest_request = history[0]
            rpm_wait = max(0, 60 - (now - oldest_request))

        total_wait = max(delay_wait, rpm_wait)

        if total_wait > 0:
            print(
                f"RATE LIMIT | {domain} | "
                f"waiting {round(total_wait, 2)} sec"
            )
            time.sleep(total_wait)

        request_time = time.time()

        self.last_request_time[domain] = request_time
        self.request_history[domain].append(request_time)

        self.log_event(
            document_id=document_id,
            url=url,
            action="request_allowed",
            waited_seconds=total_wait,
        )

    def write_report(self, row: dict) -> None:
        CRAWL_RATE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "document_id",
            "url",
            "domain",
            "delay_seconds",
            "max_requests_per_minute",
            "waited_seconds",
            "action",
            "timestamp",
        ]

        file_exists = CRAWL_RATE_REPORT_PATH.exists()

        with CRAWL_RATE_REPORT_PATH.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
