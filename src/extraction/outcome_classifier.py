import re


OUTCOME_PATTERNS = {
    "ALLOWED": [
        r"\b(?:petition|appeal|application|writ petition)\s+(?:is\s+)?allowed\b",
        r"\bwe\s+allow\s+(?:the\s+)?(?:petition|appeal|application)\b",
    ],
    "DISMISSED": [
        r"\b(?:petition|appeal|application|writ petition)\s+(?:is\s+)?dismissed\b",
        r"\bwe\s+dismiss\s+(?:the\s+)?(?:petition|appeal|application)\b",
        r"\bdismissed\s+accordingly\b",
    ],
    "DISPOSED": [
        r"\b(?:petition|appeal|application|matter)\s+(?:is\s+)?disposed\s+of\b",
        r"\bdisposed\s+of\s+accordingly\b",
    ],
    "HELD": [
        r"\bit\s+is\s+held\s+that\b",
        r"\bwe\s+hold\s+that\b",
        r"\bthe\s+court\s+held\s+that\b",
    ],
    "OVERRULED": [
        r"\bis\s+overruled\b",
        r"\bstands\s+overruled\b",
        r"\bwe\s+overrule\b",
    ],
    "UPHELD": [
        r"\bis\s+upheld\b",
        r"\bwe\s+uphold\b",
        r"\bconstitutional\s+validity\s+(?:is\s+)?upheld\b",
    ],
    "STRUCK_DOWN": [
        r"\bis\s+struck\s+down\b",
        r"\bstands\s+struck\s+down\b",
        r"\bwe\s+strike\s+down\b",
    ],
    "CONSTITUTIONAL_VALIDITY_UPHELD": [
        r"\bconstitutional\s+validity\s+(?:of\s+.*?\s+)?(?:is\s+)?upheld\b",
        r"\bvalidity\s+of\s+.*?\s+(?:is\s+)?upheld\b",
    ],
}


def classify_outcome_sentence(sentence: str) -> tuple[str, float]:
    text = " ".join(sentence.split())

    for outcome, patterns in OUTCOME_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return outcome, 0.90

    return "UNKNOWN", 0.0


def classify_case_outcome(text: str) -> dict:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    matches = []

    for sentence in sentences:
        outcome, confidence = classify_outcome_sentence(sentence)

        if outcome != "UNKNOWN":
            matches.append(
                {
                    "outcome": outcome,
                    "confidence": confidence,
                    "evidence_sentence": sentence.strip(),
                }
            )

    if not matches:
        return {
            "outcome": "UNKNOWN",
            "confidence": 0.0,
            "evidence_sentence": "",
        }

    priority = [
        "CONSTITUTIONAL_VALIDITY_UPHELD",
        "STRUCK_DOWN",
        "OVERRULED",
        "UPHELD",
        "ALLOWED",
        "DISMISSED",
        "DISPOSED",
        "HELD",
    ]

    matches.sort(
        key=lambda row: priority.index(row["outcome"])
        if row["outcome"] in priority
        else 999
    )

    return matches[0]

