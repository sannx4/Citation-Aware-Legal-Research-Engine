import re


TREATMENT_PATTERNS = {
    "OVERRULED": [
        r"\boverruled\b",
        r"\bstands\s+overruled\b",
        r"\bno\s+longer\s+good\s+law\b",
        r"\bnot\s+good\s+law\b",
    ],
    "DISTINGUISHED": [
        r"\bdistinguished\b",
        r"\bdistinguishable\b",
        r"\bnot\s+applicable\s+to\s+the\s+facts\b",
    ],
    "DOUBTED": [
        r"\bdoubted\b",
        r"\bdoubt\s+the\s+correctness\b",
        r"\bcorrectness\s+of\s+.*?\s+is\s+doubted\b",
    ],
    "CRITICIZED": [
        r"\bcriticized\b",
        r"\bdisapproved\b",
        r"\bnot\s+approved\b",
    ],
    "FOLLOWED": [
        r"\bfollowed\b",
        r"\bfollowing\s+the\s+decision\b",
        r"\bfollowing\s+the\s+judgment\b",
        r"\bwe\s+follow\b",
    ],
    "RELIED_ON": [
        r"\brelied\s+on\b",
        r"\brelied\s+upon\b",
        r"\brelying\s+on\b",
        r"\bplaced\s+reliance\s+on\b",
    ],
    "APPROVED": [
        r"\bapproved\b",
        r"\baffirmed\b",
        r"\bupheld\b",
    ],
    "REFERRED": [
        r"\breferred\s+to\b",
        r"\bmentioned\b",
        r"\bcited\b",
        r"\bconsidered\b",
    ],
}


TREATMENT_PRIORITY = [
    "OVERRULED",
    "DISTINGUISHED",
    "DOUBTED",
    "CRITICIZED",
    "FOLLOWED",
    "RELIED_ON",
    "APPROVED",
    "REFERRED",
]


NEGATIVE_TREATMENTS = {
    "OVERRULED",
    "DISTINGUISHED",
    "DOUBTED",
    "CRITICIZED",
}


POSITIVE_TREATMENTS = {
    "FOLLOWED",
    "RELIED_ON",
    "APPROVED",
}


NEUTRAL_TREATMENTS = {
    "REFERRED",
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def classify_treatment(context: str) -> dict:
    text = normalize_text(context)
    lower_text = text.lower()

    matches = []

    for treatment, patterns in TREATMENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower_text, flags=re.IGNORECASE):
                matches.append(
                    {
                        "treatment": treatment,
                        "matched_pattern": pattern,
                    }
                )

    if not matches:
        return {
            "treatment": "REFERRED",
            "polarity": "NEUTRAL",
            "confidence": 0.50,
            "matched_pattern": "",
        }

    matches.sort(
        key=lambda row: TREATMENT_PRIORITY.index(row["treatment"])
        if row["treatment"] in TREATMENT_PRIORITY
        else 999
    )

    best = matches[0]
    treatment = best["treatment"]

    if treatment in NEGATIVE_TREATMENTS:
        polarity = "NEGATIVE"
        confidence = 0.90
    elif treatment in POSITIVE_TREATMENTS:
        polarity = "POSITIVE"
        confidence = 0.85
    else:
        polarity = "NEUTRAL"
        confidence = 0.70

    return {
        "treatment": treatment,
        "polarity": polarity,
        "confidence": confidence,
        "matched_pattern": best["matched_pattern"],
    }
