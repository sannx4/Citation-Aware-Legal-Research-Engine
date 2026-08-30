import re


def extract_bench_size(text: str) -> int:
    text = str(text).lower()

    patterns = [
        r"(\d+)\s*judge bench",
        r"bench of\s*(\d+)",
        r"(\d+)\s*judges",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    if "constitution bench" in text:
        return 5

    if "division bench" in text:
        return 2

    if "single judge" in text:
        return 1

    return 2


def bench_strength_weight(bench_size: int) -> float:
    if bench_size >= 7:
        return 1.00

    if bench_size >= 5:
        return 0.90

    if bench_size >= 3:
        return 0.70

    if bench_size == 2:
        return 0.55

    if bench_size == 1:
        return 0.35

    return 0.40

