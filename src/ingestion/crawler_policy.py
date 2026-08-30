from urllib.parse import urlparse


DEFAULT_POLICY = {
    "default_delay_seconds": 2,
    "default_max_requests_per_minute": 20,
    "domain_policies": {
        "judgments.ecourts.gov.in": {
            "delay_seconds": 4,
            "max_requests_per_minute": 12,
        },
        "main.sci.gov.in": {
            "delay_seconds": 5,
            "max_requests_per_minute": 10,
        },
        "indiacode.nic.in": {
            "delay_seconds": 5,
            "max_requests_per_minute": 10,
        },
    },
}


def get_domain(url: str) -> str:
    parsed = urlparse(str(url))
    return parsed.netloc.lower()


def get_policy_for_url(url: str) -> dict:
    domain = get_domain(url)

    domain_policy = DEFAULT_POLICY["domain_policies"].get(domain, {})

    return {
        "domain": domain,
        "delay_seconds": domain_policy.get(
            "delay_seconds",
            DEFAULT_POLICY["default_delay_seconds"],
        ),
        "max_requests_per_minute": domain_policy.get(
            "max_requests_per_minute",
            DEFAULT_POLICY["default_max_requests_per_minute"],
        ),
    }