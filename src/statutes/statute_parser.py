from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "statute_documents.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "statute_provisions.jsonl"

REPORT_PATH = PROJECT_ROOT / "reports" / "statute_parser_report.csv"
QA_PATH = PROJECT_ROOT / "reports" / "statute_parser_qa.md"



# ---------------------------------------------------------------------
# Parser configuration
# ---------------------------------------------------------------------

PARSER_VERSION = "1.0.7"

STATUTE_RULES: dict[str, dict[str, Any]] = {
    "CONST_INDIA_1950": {
        "top_level_type": "ARTICLE",
        "max_numeric_base": 395,
    },
    "IPC_1860": {
        "top_level_type": "SECTION",
        "max_numeric_base": 511,
    },
    "EVIDENCE_ACT_1872": {
        "top_level_type": "SECTION",
        "max_numeric_base": 167,
    },
    "CRPC_1973": {
        "top_level_type": "SECTION",
        "max_numeric_base": 484,
    },
    "IT_ACT_2000": {
        "top_level_type": "SECTION",
        "max_numeric_base": 94,
    },
    "BNS_2023": {
        "top_level_type": "SECTION",
        "max_numeric_base": 358,
    },
    "BNSS_2023": {
        "top_level_type": "SECTION",
        "max_numeric_base": 533,
    },
    "BSA_2023": {
        "top_level_type": "SECTION",
        "max_numeric_base": 170,
    },
}


# ---------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------

PART_RE = re.compile(
    r"^\s*PART\s+([IVXLCDM]+|[A-Z0-9-]+)\b(?:[\s:.\-–—]+(.*))?$",
    re.IGNORECASE,
)

CHAPTER_RE = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+|[A-Z0-9-]+)\b(?:[\s:.\-–—]+(.*))?$",
    re.IGNORECASE,
)

SCHEDULE_RE = re.compile(
    r"^\s*(THE\s+)?"
    r"(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|"
    r"TENTH|ELEVENTH|TWELFTH|[A-Z0-9-]+)\s+SCHEDULE\b(.*)$",
    re.IGNORECASE,
)

STANDARD_TOP_RE = re.compile(
    r"^\s*\[?\s*(\d{1,4}[A-Z]{0,3})\.\s*(.*?)\s*\]?\s*$",
    re.IGNORECASE,
)

EDITORIAL_PREFIX_TOP_RE = re.compile(
    r"^\s*\d+\*+\[\s*(\d{1,4}[A-Z]{0,3})\.\s*(.*?)\s*$",
    re.IGNORECASE,
)

FOOTNOTE_PREFIX_TOP_RE = re.compile(
    r"^\s*\d+\[\s*(\d{1,4}[A-Z]{0,3})\.\s*(.*?)\s*$",
    re.IGNORECASE,
)

SUBSECTION_RE = re.compile(
    r"^\s*\((\d+[A-Z]?)\)\s+(.+?)\s*$",
    re.IGNORECASE,
)

CLAUSE_RE = re.compile(
    r"^\s*\(([a-z]{1,3})\)\s+(.+?)\s*$",
    re.IGNORECASE,
)

SUBCLAUSE_RE = re.compile(
    r"^\s*\(([ivxlcdm]+)\)\s+(.+?)\s*$",
    re.IGNORECASE,
)

EXPLANATION_RE = re.compile(
    r"^\s*Explanation(?:\s+([IVXLCDM]+|\d+))?\s*[.—:\-–]*\s*(.*)$",
    re.IGNORECASE,
)

PROVISO_RE = re.compile(
    r"^\s*Provided\s+(?:that\s+)?(.*)$",
    re.IGNORECASE,
)

ILLUSTRATION_RE = re.compile(
    r"^\s*Illustrations?\s*[.—:\-–]*\s*(.*)$",
    re.IGNORECASE,
)

FOOTERISH_RE = re.compile(
    r"^\s*(?:page\s+\d+|\d+\s+of\s+\d+)\s*$",
    re.IGNORECASE,
)

EDITORIAL_REST_RE = re.compile(
    r"^(?:"
    r"subs?\.?\s+by\b|"
    r"ins\.?\s+by\b|"
    r"inserted\s+by\b|"
    r"added\s+by\b|"
    r"omitted\s+by\b|"
    r"rep\.?\s+by\b|"
    r"repealed\s+by\b|"
    r"the\s+words?\b|"
    r"the\s+brackets?\b|"
    r"clause\s+\([^)]+\)\b|"
    r"sub-?section\s+\([^)]+\)\b"
    r")",
    re.IGNORECASE,
)

BODY_LEAD_RE = re.compile(
    r"^(?:"
    r"\(\d+[A-Z]?\)|"
    r"whoever\b|where\b|when\b|if\b|any\b|every\b|no\b|"
    r"nothing\b|notwithstanding\b|subject\b|save\b|"
    r"the\s+(?:central|state|appropriate)\s+government\b|"
    r"a\s+person\b|an\s+officer\b"
    r")",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    number: str
    title: str
    text: str
    part: str | None
    chapter: str | None
    schedule: str | None
    start_line: int
    end_line: int
    heading_style: str
    number_repaired: bool = False


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def normalize_inline(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_lines(text: str) -> list[str]:
    lines: list[str] = []

    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = normalize_inline(raw)

        if not line:
            lines.append("")
            continue

        if FOOTERISH_RE.fullmatch(line):
            continue

        lines.append(line)

    return lines


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Statute document JSONL not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}"
                ) from exc

    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def is_constitution(statute: dict[str, Any]) -> bool:
    statute_id = str(statute.get("statute_id", "")).upper()
    title = str(statute.get("title", "")).lower()

    return (
        statute_id.startswith("CONST")
        or "constitution of india" in title
    )


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned.upper()


def make_provision_id(
    statute_id: str,
    provision_type: str,
    number: str,
    suffix: str | None = None,
) -> str:
    base = f"{safe_slug(statute_id)}_{safe_slug(provision_type)}_{safe_slug(number)}"

    if suffix:
        base += f"_{safe_slug(suffix)}"

    return base


def strip_heading_punctuation(text: str) -> str:
    text = normalize_inline(text)
    text = re.sub(r"\s*[.—–:-]+\s*$", "", text)
    return text.strip()


def numeric_base(value: str) -> int | None:
    match = re.match(r"^(\d+)", str(value or "").strip().upper())

    if not match:
        return None

    return int(match.group(1))


def provision_sort_key(value: str) -> tuple[int, str]:
    raw = str(value or "").strip().upper()
    match = re.match(r"^(\d+)([A-Z]*)$", raw)

    if not match:
        return (10**9, raw)

    return (
        int(match.group(1)),
        match.group(2),
    )


def statute_rule(statute_id: str) -> dict[str, Any]:
    return STATUTE_RULES.get(
        statute_id,
        {
            "top_level_type": "SECTION",
            "max_numeric_base": None,
        },
    )


def repair_number_if_safe(
    statute_id: str,
    number: str,
) -> tuple[str, bool]:
    rule = statute_rule(statute_id)
    max_numeric = rule.get("max_numeric_base")
    base = numeric_base(number)

    if base is None or max_numeric is None:
        return number, False

    if base <= max_numeric:
        return number, False

    # Common official-PDF OCR/text-layer corruption:
    # a footnote marker "1" becomes attached to the provision number,
    # e.g. "1161." instead of "161." in the Evidence Act.
    digit_match = re.match(r"^(\d+)([A-Z]*)$", number.upper())

    if not digit_match:
        return number, False

    digits = digit_match.group(1)
    suffix = digit_match.group(2)

    if len(digits) >= 4 and digits.startswith("1"):
        repaired_digits = digits[1:]
        repaired_base = int(repaired_digits)

        # Requiring >=100 avoids turning a year such as 2023 into section 23.
        if 100 <= repaired_base <= max_numeric:
            return f"{repaired_digits}{suffix}", True

    return number, False


def heading_is_editorial_note(rest: str) -> bool:
    return bool(
        rest
        and EDITORIAL_REST_RE.match(normalize_inline(rest))
    )


def parse_top_heading(
    statute_id: str,
    line: str,
) -> tuple[str, str, str, bool] | None:
    """
    Return:
        (number, rest, heading_style, number_repaired)

    Handles:
      104.Whoever ...
      117. (1) Where ...
      66A. ...
      [2A. ...]
      1*[505. Statements ...]  (IPC official PDF text layer)
      1161. ... -> 161.        (safe OCR repair)
    """
    match = EDITORIAL_PREFIX_TOP_RE.match(line)

    if match:
        number = match.group(1).upper()
        rest = normalize_inline(match.group(2))
        style = "EDITORIAL_PREFIX_BRACKET"
    else:
        match = FOOTNOTE_PREFIX_TOP_RE.match(line)

        if match:
            number = match.group(1).upper()
            rest = normalize_inline(match.group(2))
            style = "FOOTNOTE_PREFIX_BRACKET"
        else:
            match = STANDARD_TOP_RE.match(line)

            if not match:
                return None

            number = match.group(1).upper()
            rest = normalize_inline(match.group(2))
            style = "STANDARD"

    repaired_number, repaired = repair_number_if_safe(
        statute_id,
        number,
    )

    rule = statute_rule(statute_id)
    max_numeric = rule.get("max_numeric_base")
    base = numeric_base(repaired_number)

    if (
        max_numeric is not None
        and base is not None
        and base > max_numeric
    ):
        return None

    # Reject amendment footnotes like:
    #   1. Subs. by ...
    #   2. Ins. by ...
    # while preserving provisions such as "20. [Omitted.]".
    if style == "STANDARD" and heading_is_editorial_note(rest):
        # "383. Omitted by ..." and "161. Repealed by ..." can be the
        # canonical provision stub itself. Keep those so they close the prior
        # active candidate and can carry legislative state.
        if not re.match(
            r"^(?:omitted\s+by\b|repealed\s+by\b|rep\.?\s+by\b)",
            normalize_inline(rest),
            re.IGNORECASE,
        ):
            return None

    # Reject decimal/table artifacts parsed as top-level provisions:
    #   370.64  -> number=370, rest=64
    # These commonly occur in schedules and statistical/boundary tables.
    if rest and re.fullmatch(r"\d+(?:\.\d+)?", rest):
        return None

    return (
        repaired_number,
        rest,
        style,
        repaired,
    )


def derive_title(
    candidate: Candidate,
) -> str | None:
    rest = strip_heading_punctuation(candidate.title)

    if not rest:
        return None

    # "21. Protection ...—text" -> title before em dash.
    split = re.split(r"\s*[.—–]\s*", rest, maxsplit=1)

    if len(split) == 2:
        first = strip_heading_punctuation(split[0])
        if first:
            return first

    # Some modern Acts have no marginal heading in the body:
    # "104.Whoever..." or "117. (1) Where..."
    if BODY_LEAD_RE.match(rest):
        return None

    # Avoid storing an entire body sentence as a "title".
    if len(rest.split()) > 22:
        return None

    return rest



# ---------------------------------------------------------------------
# Provision state
# ---------------------------------------------------------------------

def infer_provision_status(
    text: str,
    title: str | None = None,
    heading_style: str | None = None,
) -> str:
    """
    Descriptive state of the provision itself.

    Important distinction:
      - "165. Repealed by ..."       -> REPEALED
      - "358. Repeal and savings"    -> ACTIVE
      - an active provision whose body says another law "is hereby repealed"
        -> ACTIVE

    v1.0.6 inspected too much body text and therefore over-classified active
    repeal-and-savings provisions. v1.0.7 only treats explicit provision-level
    markers as inactive.
    """
    raw_text = text or ""
    normalized_title = normalize_inline(
        title or ""
    )

    # Recovery candidates are created only from explicit inactive evidence.
    if heading_style == "INACTIVE_RANGE_RECOVERY":
        if re.search(
            r"\b(?:rep\.?\s*by|repealed(?:\s+by)?)\b",
            raw_text,
            re.IGNORECASE,
        ):
            return "REPEALED"

        if re.search(
            r"\bomitted(?:\s+by)?\b",
            raw_text,
            re.IGNORECASE,
        ):
            return "OMITTED"

    if heading_style == "INACTIVE_DIRECT_RECOVERY":
        if re.search(
            r"\bomitted(?:\s+by)?\b",
            raw_text,
            re.IGNORECASE,
        ):
            return "OMITTED"

        if re.search(
            r"\b(?:rep\.?\s*by|repealed(?:\s+by)?)\b",
            raw_text,
            re.IGNORECASE,
        ):
            return "REPEALED"

    # Explicit title-level markers.
    if re.match(
        r"^\[?\s*omitted(?:\b|\])",
        normalized_title,
        re.IGNORECASE,
    ):
        return "OMITTED"

    if re.match(
        r"^\[?\s*(?:repealed(?:\s+by)?|rep\.?\s*by)\b",
        normalized_title,
        re.IGNORECASE,
    ):
        return "REPEALED"

    # Inspect only the first physical line of ordinary candidates. This catches:
    #   [238. Omitted.]
    #   161. Repealed by ...
    # but does not allow later body language to change the provision's status.
    first_line = ""

    for line in raw_text.splitlines():
        if line.strip():
            first_line = normalize_inline(
                line
            )
            break

    heading_body_match = re.match(
        r"^\s*(?:\d+\*+\[?\s*)?"
        r"\[?\s*\d{1,4}[A-Z]{0,3}\.\s*"
        r"(.*?)\s*\]?\s*$",
        first_line,
        re.IGNORECASE,
    )

    heading_body = (
        normalize_inline(
            heading_body_match.group(1)
        )
        if heading_body_match
        else ""
    )

    if re.match(
        r"^\[?\s*omitted(?:\b|\])",
        heading_body,
        re.IGNORECASE,
    ):
        return "OMITTED"

    if re.match(
        r"^\[?\s*(?:repealed(?:\s+by)?|rep\.?\s*by)\b",
        heading_body,
        re.IGNORECASE,
    ):
        return "REPEALED"

    return "ACTIVE"


def looks_like_legal_body_candidate(
    candidate: Candidate,
) -> bool:
    text = normalize_inline(candidate.text)
    words = len(text.split())

    if words < 6:
        return False

    return bool(
        re.search(
            r"\b(?:shall|may|means|whoever|where|when|"
            r"notwithstanding|provided|President|Parliament|Court|"
            r"Government|person|officer)\b",
            text[:500],
            re.IGNORECASE,
        )
        or re.search(
            r"\(\d+[A-Z]?\)",
            text[:500],
        )
    )


def deduplicate_with_schedule_awareness(
    candidates: list[Candidate],
    total_lines: int,
) -> tuple[list[Candidate], int, int]:
    """
    Prefer non-schedule occurrences for each provision number.

    If a provision only appears while the PDF scanner thinks it is inside a
    schedule, retain it when it still looks like genuine legislative body text
    or is explicitly OMITTED / REPEALED. This recovers PDF state-leak cases
    without allowing short table values such as `370.64`.
    """
    grouped: dict[str, list[Candidate]] = {}

    for candidate in candidates:
        grouped.setdefault(
            candidate.number.upper(),
            [],
        ).append(candidate)

    selected: list[Candidate] = []
    duplicate_count = 0
    schedule_only_rejected = 0

    for group in grouped.values():
        duplicate_count += max(
            0,
            len(group) - 1,
        )

        non_schedule = [
            candidate
            for candidate in group
            if candidate.schedule is None
        ]

        if non_schedule:
            pool = non_schedule
        else:
            eligible = [
                candidate
                for candidate in group
                if (
                    infer_provision_status(
                        candidate.text,
                        candidate.title,
                        candidate.heading_style,
                    )
                    in {"OMITTED", "REPEALED"}
                    or looks_like_legal_body_candidate(
                        candidate
                    )
                )
            ]

            if not eligible:
                schedule_only_rejected += 1
                continue

            pool = eligible

        best = max(
            pool,
            key=lambda candidate: candidate_score(
                candidate,
                total_lines,
            ),
        )

        selected.append(best)

    return (
        sorted(
            selected,
            key=lambda candidate: provision_sort_key(candidate.number),
        ),
        duplicate_count,
        schedule_only_rejected,
    )




def expand_inactive_range(
    start_token: str,
    end_token: str,
) -> list[str]:
    """
    Expand ranges such as:
        161 to 165A
    into:
        161, 162, 163, 164, 165, 165A

    This is intentionally conservative: only a numeric run plus an optional
    alphabetic suffix on the endpoint is supported.
    """
    start_match = re.fullmatch(
        r"(\d+)([A-Z]*)",
        start_token.upper(),
    )
    end_match = re.fullmatch(
        r"(\d+)([A-Z]*)",
        end_token.upper(),
    )

    if not start_match or not end_match:
        return []

    start_number = int(start_match.group(1))
    end_number = int(end_match.group(1))
    end_suffix = end_match.group(2)

    if start_number > end_number:
        return []

    values = [
        str(number)
        for number in range(
            start_number,
            end_number + 1,
        )
    ]

    if end_suffix:
        values.append(
            f"{end_number}{end_suffix}"
        )

    return values


def recover_explicit_inactive_candidates(
    statute: dict[str, Any],
    selected_candidates: list[Candidate],
) -> list[Candidate]:
    """
    Recover explicit OMITTED / REPEALED provisions from official-PDF text
    when they are encoded in forms that are not ordinary provision headings.

    Real corpus examples:
      Constitution:
        385. [Provision as to provisional Legislatures ...]
        ... Omitted by ...

      IPC:
        161 to 165A.Rep. by the Prevention of Corruption Act, 1988 ...

    Only provisions not already selected are recovered.
    """
    statute_id = statute["statute_id"]
    rule = statute_rule(statute_id)
    max_numeric = rule.get("max_numeric_base")

    existing_numbers = {
        candidate.number.upper()
        for candidate in selected_candidates
    }

    lines = normalize_lines(
        str(statute.get("text", ""))
    )

    recovered: dict[str, Candidate] = {}

    inactive_re = re.compile(
        r"\b(?:omitted|repealed|rep\.?\s*by)\b",
        re.IGNORECASE,
    )

    direct_heading_re = re.compile(
        r"^\s*\[?\s*(\d{1,4}[A-Z]{0,3})\.\s*(.*?)\s*$",
        re.IGNORECASE,
    )

    range_re = re.compile(
        r"^\s*(?:\d+\*+\[?\s*)?"
        r"(\d{1,4}[A-Z]{0,3})\s+to\s+"
        r"(\d{1,4}[A-Z]{0,3})\.\s*(.*)$",
        re.IGNORECASE,
    )

    # --------------------------------------------------------------
    # Direct inactive stubs, including split PDF headings.
    # --------------------------------------------------------------
    for index, line in enumerate(lines, start=1):
        match = direct_heading_re.match(line)

        if not match:
            continue

        number = match.group(1).upper()

        base = numeric_base(number)

        if (
            max_numeric is not None
            and base is not None
            and base > max_numeric
        ):
            continue

        nearby_lines = [line]

        # Four continuation lines are enough for the old Constitution PDF
        # while remaining local enough to avoid unrelated later material.
        for offset in range(1, 5):
            raw_index = index - 1 + offset

            if raw_index >= len(lines):
                break

            continuation = lines[raw_index]

            # Stop when a new ordinary provision clearly begins.
            next_heading = direct_heading_re.match(
                continuation
            )

            if (
                next_heading
                and next_heading.group(1).upper() != number
            ):
                break

            nearby_lines.append(
                continuation
            )

        snippet = "\n".join(
            nearby_lines
        ).strip()

        # Direct recovery must prove that THIS provision is inactive.
        # Do not treat an active "Repeal and savings" section as REPEALED
        # merely because its body repeals another enactment.
        normalized_snippet = normalize_inline(
            snippet
        )

        heading_payload = re.sub(
            r"^\s*\[?\s*\d{1,4}[A-Z]{0,3}\.\s*",
            "",
            normalized_snippet,
            count=1,
            flags=re.IGNORECASE,
        )

        explicit_direct_inactive = bool(
            re.match(
                r"^\[?\s*(?:"
                r"omitted(?:\s+by)?\b|"
                r"repealed\s+by\b|"
                r"rep\.?\s+by\b"
                r")",
                heading_payload,
                re.IGNORECASE,
            )
            or re.search(
                r"\]\s*[.—–-]*\s*(?:"
                r"omitted\s+by\b|"
                r"repealed\s+by\b|"
                r"rep\.?\s+by\b"
                r")",
                heading_payload[:500],
                re.IGNORECASE,
            )
        )

        if not explicit_direct_inactive:
            continue

        candidate = Candidate(
            number=number,
            title=normalize_inline(
                match.group(2)
            ),
            text=snippet,
            part=None,
            chapter=None,
            schedule=None,
            start_line=index,
            end_line=(
                index
                + len(nearby_lines)
                - 1
            ),
            heading_style="INACTIVE_DIRECT_RECOVERY",
            number_repaired=False,
        )

        recovered[number] = candidate

    # --------------------------------------------------------------
    # Explicit repealed/omitted ranges such as "161 to 165A.Rep. by..."
    # --------------------------------------------------------------
    for index, line in enumerate(lines, start=1):
        match = range_re.match(line)

        if not match:
            continue

        start_token = match.group(1).upper()
        end_token = match.group(2).upper()

        nearby_lines = [line]

        for offset in (1, 2):
            raw_index = index - 1 + offset

            if raw_index < len(lines):
                nearby_lines.append(
                    lines[raw_index]
                )

        snippet = "\n".join(
            nearby_lines
        ).strip()

        if not inactive_re.search(snippet):
            continue

        expanded_numbers = expand_inactive_range(
            start_token,
            end_token,
        )

        for number in expanded_numbers:
            if (
                number in existing_numbers
                or number in recovered
            ):
                continue

            base = numeric_base(number)

            if (
                max_numeric is not None
                and base is not None
                and base > max_numeric
            ):
                continue

            status_word = (
                "Repealed"
                if re.search(
                    r"\b(?:repealed|rep\.?\s*by)\b",
                    snippet,
                    re.IGNORECASE,
                )
                else "Omitted"
            )

            recovered[number] = Candidate(
                number=number,
                title=status_word,
                text=snippet,
                part=None,
                chapter=None,
                schedule=None,
                start_line=index,
                end_line=(
                    index
                    + len(nearby_lines)
                    - 1
                ),
                heading_style="INACTIVE_RANGE_RECOVERY",
                number_repaired=False,
            )

    return sorted(
        recovered.values(),
        key=lambda candidate: provision_sort_key(
            candidate.number
        ),
    )



# ---------------------------------------------------------------------
# Top-level provision extraction
# ---------------------------------------------------------------------

def scan_top_level_candidates(
    statute: dict[str, Any],
) -> tuple[
    list[Candidate],
    list[dict[str, Any]],
    dict[str, int],
]:
    """
    Scan the entire extracted text.

    Important:
    Do NOT stop when a SCHEDULE heading is encountered. Official PDFs often
    contain a table of contents before the actual body, and that contents page
    may itself list "THE FIRST SCHEDULE". v1.0.1 could therefore close the
    statute before reaching the enacted text.

    Body-window selection happens later, after all candidates are available.
    """
    statute_id = statute["statute_id"]
    lines = normalize_lines(str(statute.get("text", "")))

    candidates: list[Candidate] = []
    structures: list[dict[str, Any]] = []

    current_part: str | None = None
    current_chapter: str | None = None
    current_schedule: str | None = None

    active: dict[str, Any] | None = None

    diagnostics = {
        "out_of_range_rejected": 0,
        "ocr_number_repairs": 0,
        "editorial_notes_rejected": 0,
    }

    rule = statute_rule(statute_id)
    max_numeric = rule.get("max_numeric_base")

    def close_active(end_line: int) -> None:
        nonlocal active

        if active is None:
            return

        text = "\n".join(active["body_lines"]).strip()

        candidates.append(
            Candidate(
                number=active["number"],
                title=active["title"],
                text=text,
                part=active["part"],
                chapter=active["chapter"],
                schedule=active["schedule"],
                start_line=active["start_line"],
                end_line=end_line,
                heading_style=active["heading_style"],
                number_repaired=active["number_repaired"],
            )
        )

        active = None

    for index, line in enumerate(lines, start=1):
        part_match = PART_RE.match(line)

        if part_match:
            close_active(index - 1)

            current_part = part_match.group(1).upper()
            current_chapter = None
            current_schedule = None

            structures.append(
                {
                    "type": "PART",
                    "number": current_part,
                    "title": strip_heading_punctuation(
                        part_match.group(2) or ""
                    ),
                    "line": index,
                }
            )
            continue

        chapter_match = CHAPTER_RE.match(line)

        if chapter_match:
            close_active(index - 1)

            current_chapter = chapter_match.group(1).upper()
            current_schedule = None

            structures.append(
                {
                    "type": "CHAPTER",
                    "number": current_chapter,
                    "title": strip_heading_punctuation(
                        chapter_match.group(2) or ""
                    ),
                    "line": index,
                }
            )
            continue

        schedule_match = SCHEDULE_RE.match(line)

        if schedule_match:
            close_active(index - 1)

            current_schedule = schedule_match.group(2).upper()
            current_part = None
            current_chapter = None

            structures.append(
                {
                    "type": "SCHEDULE",
                    "number": current_schedule,
                    "title": strip_heading_punctuation(
                        schedule_match.group(3) or ""
                    ),
                    "line": index,
                }
            )
            continue

        before = (
            STANDARD_TOP_RE.match(line)
            or EDITORIAL_PREFIX_TOP_RE.match(line)
            or FOOTNOTE_PREFIX_TOP_RE.match(line)
        )

        parsed = parse_top_heading(
            statute_id,
            line,
        )

        if parsed is None:
            if before:
                rough_num_match = re.search(
                    r"(\d{1,4}[A-Z]{0,3})\.",
                    line,
                )

                if rough_num_match:
                    rough_num = rough_num_match.group(1).upper()
                    rough_base = numeric_base(rough_num)

                    if (
                        max_numeric is not None
                        and rough_base is not None
                        and rough_base > max_numeric
                    ):
                        diagnostics[
                            "out_of_range_rejected"
                        ] += 1

                    elif heading_is_editorial_note(
                        re.sub(
                            r"^\s*\[?\s*\d{1,4}[A-Z]{0,3}\.\s*",
                            "",
                            line,
                            flags=re.IGNORECASE,
                        )
                    ):
                        diagnostics[
                            "editorial_notes_rejected"
                        ] += 1

            if active is not None:
                active["body_lines"].append(line)

            continue

        number, rest, style, repaired = parsed

        close_active(index - 1)

        if repaired:
            diagnostics["ocr_number_repairs"] += 1

        active = {
            "number": number,
            "title": rest,
            "body_lines": [line],
            "part": current_part,
            "chapter": current_chapter,
            "schedule": current_schedule,
            "start_line": index,
            "heading_style": style,
            "number_repaired": repaired,
        }

    close_active(len(lines))

    return candidates, structures, diagnostics


def candidate_score(
    candidate: Candidate,
    total_lines: int,
) -> tuple[int, int, int, int]:
    """
    Prefer a real body occurrence over:
      - arrangement/table-of-contents entries,
      - marginal listings,
      - amendment notes.

    The body occurrence is normally substantially longer.
    """
    text = normalize_inline(candidate.text)
    words = len(text.split())

    legal_body_bonus = 0

    if re.search(
        r"\b(?:shall|may|means|whoever|where|when|provided|notwithstanding)\b",
        text[:500],
        re.IGNORECASE,
    ):
        legal_body_bonus += 40

    if re.search(r"\(\d+[A-Z]?\)", text[:500]):
        legal_body_bonus += 20

    if candidate.heading_style in {
        "EDITORIAL_PREFIX_BRACKET",
        "FOOTNOTE_PREFIX_BRACKET",
    }:
        legal_body_bonus += 10

    # Small preference for later occurrences because official PDFs often put
    # the arrangement of sections first and the enacted text afterwards.
    later_bonus = int(
        20 * candidate.start_line / max(total_lines, 1)
    )

    return (
        words + legal_body_bonus + later_bonus,
        len(text),
        int(candidate.number_repaired),
        candidate.start_line,
    )


def deduplicate_candidates(
    candidates: list[Candidate],
    total_lines: int,
) -> tuple[list[Candidate], int]:
    best: dict[str, Candidate] = {}
    duplicate_count = 0

    for candidate in candidates:
        key = candidate.number.upper()

        if key not in best:
            best[key] = candidate
            continue

        duplicate_count += 1

        if (
            candidate_score(candidate, total_lines)
            > candidate_score(best[key], total_lines)
        ):
            best[key] = candidate

    return (
        sorted(
            best.values(),
            key=lambda item: provision_sort_key(item.number),
        ),
        duplicate_count,
    )


def derive_body_bounds_from_selected(
    statute_id: str,
    selected_candidates: list[Candidate],
) -> dict[str, int | None]:
    """
    Infer the enacted-body bounds only AFTER per-provision deduplication.

    v1.0.2 tried to choose Section/Article 1 and the terminal provision before
    deduplication. Old official PDFs contain TOCs, schedules, amendment notes,
    and repeated numbering, so a wrong anchor could delete most of the Act.

    v1.0.3 keeps all candidates for deduplication first. The selected body-like
    occurrence of each provision is then used only to infer reporting/structure
    bounds; it never deletes top-level provisions.
    """
    rule = statute_rule(statute_id)
    max_numeric = rule.get("max_numeric_base")

    by_number = {
        candidate.number.upper(): candidate
        for candidate in selected_candidates
    }

    start_candidate = by_number.get("1")

    terminal_candidate = (
        by_number.get(str(max_numeric))
        if max_numeric is not None
        else None
    )

    body_start = (
        start_candidate.start_line
        if start_candidate is not None
        else None
    )

    body_end = (
        terminal_candidate.end_line
        if terminal_candidate is not None
        else None
    )

    if (
        body_start is not None
        and body_end is not None
        and body_start >= body_end
    ):
        # Do not report a misleading reversed window.
        body_start = None
        body_end = None

    return {
        "body_start_line": body_start,
        "body_end_line": body_end,
    }


def filter_structure_nodes_to_selected_body(
    structures: list[dict[str, Any]],
    body_start: int | None,
    body_end: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Structure headings (PART/CHAPTER/SCHEDULE) can be duplicated in a TOC.
    Filter them only when a trustworthy post-dedup body interval exists.
    """
    if body_start is None or body_end is None:
        return structures, 0

    filtered = [
        structure
        for structure in structures
        if body_start <= structure["line"] <= body_end
    ]

    return (
        filtered,
        len(structures) - len(filtered),
    )



# ---------------------------------------------------------------------
# Child parsing
# ---------------------------------------------------------------------

def normalize_child_line(
    parent_number: str,
    line: str,
    relative_line: int,
) -> str:
    """
    Normalize official-PDF child markers.

    Examples:
      117. (1) Where ...          -> (1) Where ...
      1. Name ...—(1) India ...  -> (1) India ...
      1[(2) The States ...       -> (2) The States ...
    """
    value = normalize_inline(line)

    # Footnote marker wrapped around a subsection/clause marker.
    footnote_wrapped = re.match(
        r"^\s*\d+\[\s*(\((?:\d+[A-Z]?|[a-z]{1,3}|[ivxlcdm]+)\).*)$",
        value,
        re.IGNORECASE,
    )

    if footnote_wrapped:
        value = footnote_wrapped.group(1)

    if relative_line == 1:
        # Remove ordinary or editorially-prefixed top-level number.
        value = re.sub(
            r"^\s*(?:\d+\*+\[\s*)?\[?\s*"
            + re.escape(str(parent_number))
            + r"\.\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )

        # A provision title can precede the first subsection on the same line.
        marker = re.search(
            r"(?:"
            r"\(\d+[A-Z]?\)|"
            r"\([a-z]{1,3}\)|"
            r"\([ivxlcdm]+\)|"
            r"\bProvided\b|"
            r"\bExplanation\b|"
            r"\bIllustrations?\b"
            r")",
            value,
            re.IGNORECASE,
        )

        if marker:
            value = value[marker.start():]

    return normalize_inline(value)


def split_child_units(
    statute_id: str,
    parent: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract child legal units from the selected body occurrence.

    v1.0.2 additionally handles child markers located on the same line as the
    top-level provision number and common footnote-wrapped markers.
    """
    lines = normalize_lines(parent["text"])

    children: list[dict[str, Any]] = []
    counters = {
        "PROVISO": 0,
        "EXPLANATION": 0,
        "ILLUSTRATION": 0,
    }

    current_subsection: str | None = None
    current_clause: str | None = None

    for relative_line, raw_line in enumerate(lines, start=1):
        if not raw_line:
            continue

        line = normalize_child_line(
            parent["number"],
            raw_line,
            relative_line,
        )

        if not line:
            continue

        subsection = SUBSECTION_RE.match(line)

        if subsection:
            number = subsection.group(1)
            current_subsection = number
            current_clause = None

            children.append(
                {
                    "statute_id": statute_id,
                    "provision_id": make_provision_id(
                        statute_id,
                        "SUBSECTION",
                        parent["number"],
                        number,
                    ),
                    "provision_type": "SUBSECTION",
                    "number": number,
                    "title": None,
                    "text": line,
                    "parent_provision_id": parent["provision_id"],
                    "parent_number": parent["number"],
                    "part": parent["part"],
                    "chapter": parent["chapter"],
                    "schedule": parent["schedule"],
                    "source_start_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "source_end_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "parser_confidence": 0.95,
                    "needs_review": False,
                    "parser_version": PARSER_VERSION,
                }
            )
            continue

        # Roman markers can be clauses or subclauses. When we are already
        # inside a clause, treat them as subclauses first.
        subclause = SUBCLAUSE_RE.match(line)

        if subclause and current_clause:
            number = subclause.group(1).lower()

            suffix_bits = [
                bit
                for bit in (
                    current_subsection,
                    current_clause,
                    number,
                )
                if bit
            ]

            parent_suffix = "_".join(
                bit
                for bit in (
                    current_subsection,
                    current_clause,
                )
                if bit
            )

            children.append(
                {
                    "statute_id": statute_id,
                    "provision_id": make_provision_id(
                        statute_id,
                        "SUBCLAUSE",
                        parent["number"],
                        "_".join(suffix_bits),
                    ),
                    "provision_type": "SUBCLAUSE",
                    "number": number,
                    "title": None,
                    "text": line,
                    "parent_provision_id": make_provision_id(
                        statute_id,
                        "CLAUSE",
                        parent["number"],
                        parent_suffix,
                    ),
                    "parent_number": parent["number"],
                    "part": parent["part"],
                    "chapter": parent["chapter"],
                    "schedule": parent["schedule"],
                    "source_start_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "source_end_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "parser_confidence": 0.90,
                    "needs_review": False,
                    "parser_version": PARSER_VERSION,
                }
            )
            continue

        clause = CLAUSE_RE.match(line)

        if clause:
            number = clause.group(1).lower()
            current_clause = number

            suffix = (
                f"{current_subsection}_{number}"
                if current_subsection
                else number
            )

            parent_id = (
                make_provision_id(
                    statute_id,
                    "SUBSECTION",
                    parent["number"],
                    current_subsection,
                )
                if current_subsection
                else parent["provision_id"]
            )

            children.append(
                {
                    "statute_id": statute_id,
                    "provision_id": make_provision_id(
                        statute_id,
                        "CLAUSE",
                        parent["number"],
                        suffix,
                    ),
                    "provision_type": "CLAUSE",
                    "number": number,
                    "title": None,
                    "text": line,
                    "parent_provision_id": parent_id,
                    "parent_number": parent["number"],
                    "part": parent["part"],
                    "chapter": parent["chapter"],
                    "schedule": parent["schedule"],
                    "source_start_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "source_end_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "parser_confidence": 0.92,
                    "needs_review": False,
                    "parser_version": PARSER_VERSION,
                }
            )
            continue

        explanation = EXPLANATION_RE.match(line)

        if explanation:
            counters["EXPLANATION"] += 1
            label = (
                explanation.group(1)
                or str(counters["EXPLANATION"])
            )

            children.append(
                {
                    "statute_id": statute_id,
                    "provision_id": make_provision_id(
                        statute_id,
                        "EXPLANATION",
                        parent["number"],
                        label,
                    ),
                    "provision_type": "EXPLANATION",
                    "number": label,
                    "title": None,
                    "text": line,
                    "parent_provision_id": parent["provision_id"],
                    "parent_number": parent["number"],
                    "part": parent["part"],
                    "chapter": parent["chapter"],
                    "schedule": parent["schedule"],
                    "source_start_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "source_end_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "parser_confidence": 0.93,
                    "needs_review": False,
                    "parser_version": PARSER_VERSION,
                }
            )
            continue

        proviso = PROVISO_RE.match(line)

        if proviso:
            counters["PROVISO"] += 1
            number = str(counters["PROVISO"])

            children.append(
                {
                    "statute_id": statute_id,
                    "provision_id": make_provision_id(
                        statute_id,
                        "PROVISO",
                        parent["number"],
                        number,
                    ),
                    "provision_type": "PROVISO",
                    "number": number,
                    "title": None,
                    "text": line,
                    "parent_provision_id": parent["provision_id"],
                    "parent_number": parent["number"],
                    "part": parent["part"],
                    "chapter": parent["chapter"],
                    "schedule": parent["schedule"],
                    "source_start_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "source_end_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "parser_confidence": 0.92,
                    "needs_review": False,
                    "parser_version": PARSER_VERSION,
                }
            )
            continue

        illustration = ILLUSTRATION_RE.match(line)

        if illustration:
            counters["ILLUSTRATION"] += 1
            number = str(counters["ILLUSTRATION"])

            children.append(
                {
                    "statute_id": statute_id,
                    "provision_id": make_provision_id(
                        statute_id,
                        "ILLUSTRATION",
                        parent["number"],
                        number,
                    ),
                    "provision_type": "ILLUSTRATION",
                    "number": number,
                    "title": None,
                    "text": line,
                    "parent_provision_id": parent["provision_id"],
                    "parent_number": parent["number"],
                    "part": parent["part"],
                    "chapter": parent["chapter"],
                    "schedule": parent["schedule"],
                    "source_start_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "source_end_line": (
                        parent["source_start_line"]
                        + relative_line
                        - 1
                    ),
                    "parser_confidence": 0.90,
                    "needs_review": False,
                    "parser_version": PARSER_VERSION,
                }
            )

    return children



# ---------------------------------------------------------------------
# Structural nodes
# ---------------------------------------------------------------------

def build_structure_nodes(
    statute: dict[str, Any],
    structures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    statute_id = statute["statute_id"]

    rows: list[dict[str, Any]] = []

    for structure in structures:
        structure_type = structure["type"]
        number = structure["number"]

        rows.append(
            {
                "statute_id": statute_id,
                "provision_id": make_provision_id(
                    statute_id,
                    structure_type,
                    number,
                ),
                "provision_type": structure_type,
                "number": number,
                "title": structure.get("title") or None,
                "text": None,
                "parent_provision_id": None,
                "parent_number": None,
                "part": (
                    number
                    if structure_type == "PART"
                    else None
                ),
                "chapter": (
                    number
                    if structure_type == "CHAPTER"
                    else None
                ),
                "schedule": (
                    number
                    if structure_type == "SCHEDULE"
                    else None
                ),
                "source_start_line": structure["line"],
                "source_end_line": structure["line"],
                "parser_confidence": 0.98,
                "needs_review": False,
                "parser_version": PARSER_VERSION,
            }
        )

    return rows




# ---------------------------------------------------------------------
# Per-statute parser
# ---------------------------------------------------------------------

def parse_statute(
    statute: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    statute_id = statute["statute_id"]

    rule = statute_rule(statute_id)

    provision_type = rule.get(
        "top_level_type",
        "ARTICLE" if is_constitution(statute) else "SECTION",
    )

    raw_candidates, structures, diagnostics = scan_top_level_candidates(
        statute
    )

    total_lines = max(
        1,
        len(
            normalize_lines(
                str(statute.get("text", ""))
            )
        ),
    )

    # v1.0.5: prefer non-schedule occurrences, but retain a schedule-context
    # candidate when it independently looks like real legislative body text.
    (
        candidates,
        duplicate_count,
        schedule_candidates_rejected,
    ) = deduplicate_with_schedule_awareness(
        raw_candidates,
        total_lines,
    )

    inactive_recovered = recover_explicit_inactive_candidates(
        statute,
        candidates,
    )

    candidate_by_number = {
        candidate.number.upper(): candidate
        for candidate in candidates
    }

    for recovered_candidate in inactive_recovered:
        key = recovered_candidate.number.upper()
        existing = candidate_by_number.get(key)

        if existing is None:
            candidate_by_number[key] = recovered_candidate
            continue

        recovered_status = infer_provision_status(
            recovered_candidate.text,
            recovered_candidate.title,
            recovered_candidate.heading_style,
        )
        existing_status = infer_provision_status(
            existing.text,
            existing.title,
            existing.heading_style,
        )

        # Prefer explicit inactive evidence over a weak/partial ACTIVE
        # candidate such as a TOC heading or a split PDF heading.
        if (
            recovered_status in {"OMITTED", "REPEALED"}
            and existing_status == "ACTIVE"
        ):
            candidate_by_number[key] = recovered_candidate

    candidates = sorted(
        candidate_by_number.values(),
        key=lambda candidate: provision_sort_key(
            candidate.number
        ),
    )

    body_diagnostics = derive_body_bounds_from_selected(
        statute_id,
        candidates,
    )

    (
        body_structures,
        structures_rejected,
    ) = filter_structure_nodes_to_selected_body(
        structures,
        body_diagnostics["body_start_line"],
        body_diagnostics["body_end_line"],
    )

    rows: list[dict[str, Any]] = []

    rows.extend(
        build_structure_nodes(
            statute,
            body_structures,
        )
    )

    top_level_rows: list[dict[str, Any]] = []

    for sequence, candidate in enumerate(candidates, start=1):
        text = candidate.text.strip()
        words = len(normalize_inline(text).split())

        title = derive_title(candidate)

        provision_status = infer_provision_status(
            text,
            title,
            candidate.heading_style,
        )

        # Legitimate provisions can be very short:
        #   "147. Interpretation."
        #   "5. Saving."
        #   "[238. Omitted.]"
        needs_review = (
            words < 2
            and provision_status == "ACTIVE"
        )

        confidence = 0.99

        if candidate.number_repaired:
            confidence = 0.90

        if words < 10:
            confidence = min(confidence, 0.82)

        row = {
            "statute_id": statute_id,
            "provision_id": make_provision_id(
                statute_id,
                provision_type,
                candidate.number,
            ),
            "provision_type": provision_type,
            "number": candidate.number,
            "title": title,
            "text": text,
            "provision_status": provision_status,
            "parent_provision_id": None,
            "parent_number": None,
            "part": candidate.part,
            "chapter": candidate.chapter,
            "schedule": candidate.schedule,
            "sequence": sequence,
            "source_start_line": candidate.start_line,
            "source_end_line": candidate.end_line,
            "word_count": words,
            "heading_style": candidate.heading_style,
            "number_repaired": candidate.number_repaired,
            "parser_confidence": confidence,
            "needs_review": needs_review,
            "parser_version": PARSER_VERSION,
        }

        top_level_rows.append(row)
        rows.append(row)

    child_rows: list[dict[str, Any]] = []

    for parent in top_level_rows:
        child_rows.extend(
            split_child_units(
                statute_id,
                parent,
            )
        )

    rows.extend(child_rows)

    top_count = len(top_level_rows)
    review_count = sum(
        1
        for row in rows
        if row.get("needs_review")
    )

    active_count = sum(
        1
        for row in top_level_rows
        if row.get("provision_status") == "ACTIVE"
    )

    omitted_count = sum(
        1
        for row in top_level_rows
        if row.get("provision_status") == "OMITTED"
    )

    repealed_count = sum(
        1
        for row in top_level_rows
        if row.get("provision_status") == "REPEALED"
    )

    max_numeric = rule.get("max_numeric_base")

    present_bases = {
        numeric_base(row["number"])
        for row in top_level_rows
        if numeric_base(row["number"]) is not None
    }

    missing_bases: list[int] = []

    if max_numeric is not None:
        missing_bases = [
            number
            for number in range(1, max_numeric + 1)
            if number not in present_bases
        ]

    coverage = (
        round(
            (max_numeric - len(missing_bases))
            / max_numeric,
            6,
        )
        if max_numeric
        else None
    )

    qa_issues: list[str] = []

    if top_count == 0:
        qa_issues.append("no_top_level_provisions_found")

    if max_numeric and missing_bases:
        qa_issues.append(
            f"missing_numeric_bases:{len(missing_bases)}"
        )

    if max_numeric and coverage is not None and coverage < 0.98:
        qa_issues.append(
            f"low_numeric_coverage:{coverage:.3f}"
        )

    if top_count >= 20 and len(child_rows) == 0:
        qa_issues.append(
            "no_child_units_detected"
        )

    report = {
        "statute_id": statute_id,
        "title": statute.get("title"),
        "top_level_type": provision_type,
        "expected_max_numeric_base": max_numeric,
        "raw_top_level_candidates": len(raw_candidates),
        "body_window_candidates": len(candidates),
        "schedule_candidates_rejected": schedule_candidates_rejected,
        "inactive_provisions_recovered": len(inactive_recovered),
        "deduplicated_top_level_provisions": top_count,
        "active_provisions": active_count,
        "omitted_provisions": omitted_count,
        "repealed_provisions": repealed_count,
        "duplicate_candidates_dropped": duplicate_count,
        "body_start_line": body_diagnostics[
            "body_start_line"
        ],
        "body_end_line": body_diagnostics[
            "body_end_line"
        ],
        "outside_body_candidates_rejected": 0,
        "structure_nodes_outside_body_rejected": structures_rejected,
        "numeric_coverage": coverage,
        "missing_numeric_bases": ",".join(
            str(number)
            for number in missing_bases
        ),
        "ocr_number_repairs": diagnostics[
            "ocr_number_repairs"
        ],
        "out_of_range_rejected": diagnostics[
            "out_of_range_rejected"
        ],
        "editorial_notes_rejected": diagnostics[
            "editorial_notes_rejected"
        ],
        "structural_nodes": len(
            [
                row
                for row in rows
                if row["provision_type"]
                in {"PART", "CHAPTER", "SCHEDULE"}
            ]
        ),
        "child_units": len(child_rows),
        "review_rows": review_count,
        "qa_issues": ";".join(qa_issues),
    }

    return rows, report


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def write_report(
    reports: list[dict[str, Any]],
) -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "statute_id",
        "title",
        "top_level_type",
        "expected_max_numeric_base",
        "raw_top_level_candidates",
        "body_window_candidates",
        "schedule_candidates_rejected",
        "inactive_provisions_recovered",
        "deduplicated_top_level_provisions",
        "active_provisions",
        "omitted_provisions",
        "repealed_provisions",
        "duplicate_candidates_dropped",
        "body_start_line",
        "body_end_line",
        "outside_body_candidates_rejected",
        "structure_nodes_outside_body_rejected",
        "numeric_coverage",
        "missing_numeric_bases",
        "ocr_number_repairs",
        "out_of_range_rejected",
        "editorial_notes_rejected",
        "structural_nodes",
        "child_units",
        "review_rows",
        "qa_issues",
    ]

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(reports)


def write_qa(
    reports: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> None:
    total_top_level = sum(
        report["deduplicated_top_level_provisions"]
        for report in reports
    )

    total_duplicates = sum(
        report["duplicate_candidates_dropped"]
        for report in reports
    )

    total_children = sum(
        report["child_units"]
        for report in reports
    )

    total_review = sum(
        1
        for row in all_rows
        if row.get("needs_review")
    )

    lines = [
        "# Statute Parser QA",
        "",
        f"Parser version: `{PARSER_VERSION}`",
        "",
        "## Summary",
        "",
        f"- Statutes processed: **{len(reports)}**",
        f"- Top-level provisions: **{total_top_level}**",
        f"- Duplicate candidates dropped: **{total_duplicates}**",
        f"- Child units extracted: **{total_children}**",
        f"- Rows needing review: **{total_review}**",
        "",
        "## Per-statute results",
        "",
    ]

    for report in reports:
        lines.extend(
            [
                f"### {report['statute_id']} — {report['title']}",
                "",
                f"- Top-level type: **{report['top_level_type']}**",
                (
                    f"- Expected max numeric base: "
                    f"**{report['expected_max_numeric_base']}**"
                ),
                (
                    f"- Raw candidates: "
                    f"**{report['raw_top_level_candidates']}**"
                ),
                (
                    f"- Main-body candidates: "
                    f"**{report['body_window_candidates']}**"
                ),
                (
                    f"- Schedule-context candidates rejected: "
                    f"**{report['schedule_candidates_rejected']}**"
                ),
                (
                    f"- Inactive provisions recovered: "
                    f"**{report['inactive_provisions_recovered']}**"
                ),
                (
                    f"- ACTIVE provisions: "
                    f"**{report['active_provisions']}**"
                ),
                (
                    f"- OMITTED provisions: "
                    f"**{report['omitted_provisions']}**"
                ),
                (
                    f"- REPEALED provisions: "
                    f"**{report['repealed_provisions']}**"
                ),
                (
                    f"- Body start line: "
                    f"**{report['body_start_line']}**"
                ),
                (
                    f"- Body end line: "
                    f"**{report['body_end_line']}**"
                ),
                (
                    f"- Outside-body candidates rejected: "
                    f"**{report['outside_body_candidates_rejected']}**"
                ),
                (
                    f"- Structure nodes outside selected body rejected: "
                    f"**{report['structure_nodes_outside_body_rejected']}**"
                ),
                (
                    f"- Deduplicated provisions: "
                    f"**{report['deduplicated_top_level_provisions']}**"
                ),
                (
                    f"- Duplicate candidates dropped: "
                    f"**{report['duplicate_candidates_dropped']}**"
                ),
                (
                    f"- Numeric coverage: "
                    f"**{report['numeric_coverage']}**"
                ),
                (
                    f"- Missing numeric bases: "
                    f"**{report['missing_numeric_bases'] or 'none'}**"
                ),
                (
                    f"- OCR number repairs: "
                    f"**{report['ocr_number_repairs']}**"
                ),
                (
                    f"- Out-of-range candidates rejected: "
                    f"**{report['out_of_range_rejected']}**"
                ),
                (
                    f"- Editorial notes rejected: "
                    f"**{report['editorial_notes_rejected']}**"
                ),
                (
                    f"- Structural nodes: "
                    f"**{report['structural_nodes']}**"
                ),
                (
                    f"- Child units: "
                    f"**{report['child_units']}**"
                ),
                (
                    f"- Review rows: "
                    f"**{report['review_rows']}**"
                ),
                (
                    f"- QA issues: "
                    f"**{report['qa_issues'] or 'none'}**"
                ),
                "",
            ]
        )

    QA_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def run() -> None:
    statutes = load_jsonl(INPUT_PATH)

    all_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for statute in statutes:
        rows, report = parse_statute(statute)

        all_rows.extend(rows)
        reports.append(report)

    write_jsonl(
        OUTPUT_PATH,
        all_rows,
    )

    write_report(
        reports,
    )

    write_qa(
        reports,
        all_rows,
    )

    total_top_level = sum(
        report["deduplicated_top_level_provisions"]
        for report in reports
    )

    total_children = sum(
        report["child_units"]
        for report in reports
    )

    total_duplicates = sum(
        report["duplicate_candidates_dropped"]
        for report in reports
    )

    total_review = sum(
        1
        for row in all_rows
        if row.get("needs_review")
    )

    print()
    print("Statute Parser")
    print("=" * 70)
    print(f"Parser version: {PARSER_VERSION}")
    print(f"Statutes processed: {len(statutes)}")
    print(f"Top-level provisions: {total_top_level}")
    print(f"Child units extracted: {total_children}")
    print(f"Duplicate candidates dropped: {total_duplicates}")
    print(f"Rows needing review: {total_review}")
    print()
    print(f"Output: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"QA report: {QA_PATH}")
    print("=" * 70)

    print()
    print("Per-statute")
    print("-" * 70)

    for report in reports:
        print(
            f"{report['statute_id']} | "
            f"{report['top_level_type']}="
            f"{report['deduplicated_top_level_provisions']} | "
            f"active={report['active_provisions']} | "
            f"omitted={report['omitted_provisions']} | "
            f"repealed={report['repealed_provisions']} | "
            f"coverage={report['numeric_coverage']} | "
            f"missing={report['missing_numeric_bases'] or 'none'} | "
            f"repairs={report['ocr_number_repairs']} | "
            f"children={report['child_units']} | "
            f"qa={report['qa_issues'] or 'OK'}"
        )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
