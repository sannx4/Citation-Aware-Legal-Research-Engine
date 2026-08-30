import re


CASE_NAME_PATTERN = re.compile(
    r"\b("
    r"[A-Z][A-Za-z.\s&'()-]{1,100}?"
    r")\s+"
    r"(v\.|vs\.?|versus)\s+"
    r"("
    r"[A-Z][A-Za-z.\s&'()-]{1,120}?"
    r")"
    r"(?=[,.;:\n)]|\s+reported|\s+held|\s+observed|\s+case|\s+under|\s+and|$)",
    flags=re.IGNORECASE,
)

SCC_PATTERN = re.compile(
    r"\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)",
    flags=re.IGNORECASE,
)

AIR_PATTERN = re.compile(
    r"\bAIR\s+(\d{4})\s+([A-Z]{2,5})\s+(\d+)\b",
    flags=re.IGNORECASE,
)

NEUTRAL_CITATION_PATTERN = re.compile(
    r"\b(\d{4})\s+(INSC|INHC|SCC\s+OnLine\s+[A-Z]{2,10})\s+(\d+)\b",
    flags=re.IGNORECASE,
)

FALSE_POSITIVE_TERMS = {
    "article",
    "section",
    "facts",
    "issues",
    "reasoning",
    "decision",
    "petitioner",
    "respondent",
    "appellant",
    "court",
}


def clean_text(value: str) -> str:
    return " ".join(str(value).strip(" ,.;:\n\t").split())


def is_false_case_name(party_1: str, party_2: str) -> bool:
    p1 = party_1.lower().strip()
    p2 = party_2.lower().strip()

    if len(p1) < 2 or len(p2) < 2:
        return True

    if p1 in FALSE_POSITIVE_TERMS or p2 in FALSE_POSITIVE_TERMS:
        return True

    if "article" in p1 or "article" in p2:
        return True

    if "section" in p1 or "section" in p2:
        return True

    return False
