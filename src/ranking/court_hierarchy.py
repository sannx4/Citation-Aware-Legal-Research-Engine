def court_weight(court: str) -> float:
    court = str(court).lower()

    if "supreme court" in court:
        return 1.00

    if "high court" in court:
        return 0.75

    if "district" in court or "sessions" in court:
        return 0.45

    if "tribunal" in court:
        return 0.40

    return 0.30