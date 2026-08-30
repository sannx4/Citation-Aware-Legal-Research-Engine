import re


EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
PINCODE_PATTERN = re.compile(r"\b\d{6}\b")

MINOR_PATTERN = re.compile(
    r"\b(?:minor|child|victim girl|victim boy)\s+([A-Z][A-Za-z]+)\b",
    flags=re.IGNORECASE,
)


def mask_pii(text: str) -> tuple[str, dict]:
    counts = {
        "emails": 0,
        "phones": 0,
        "aadhaar": 0,
        "pan": 0,
        "pincode": 0,
        "minor_names": 0,
    }

    text, counts["emails"] = EMAIL_PATTERN.subn("[EMAIL_MASKED]", text)
    text, counts["phones"] = PHONE_PATTERN.subn("[PHONE_MASKED]", text)
    text, counts["aadhaar"] = AADHAAR_PATTERN.subn("[AADHAAR_MASKED]", text)
    text, counts["pan"] = PAN_PATTERN.subn("[PAN_MASKED]", text)
    text, counts["pincode"] = PINCODE_PATTERN.subn("[PINCODE_MASKED]", text)
    text, counts["minor_names"] = MINOR_PATTERN.subn("minor [MINOR_NAME_MASKED]", text)

    return text, counts
