from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LEGAL_CHUNKS_PATH = (
    PROCESSED_DIR / "legal_chunks.jsonl"
)

TREATMENTS_PATH = (
    PROCESSED_DIR / "citation_treatments.jsonl"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORT_PATH = (
    REPORTS_DIR / "citation_treatment_report.csv"
)


DETECTOR_VERSION = "1.6.0"


# ============================================================
# Labels
# ============================================================

TREATMENT_LABELS = (
    "FOLLOWED",
    "RELIED_ON",
    "APPROVED",
    "DISTINGUISHED",
    "DOUBTED",
    "OVERRULED",
    "DISAPPROVED",
    "REFERRED_TO",
)


TREATMENT_ACTORS = (
    "COURT",
    "PARTY",
    "QUOTED_AUTHORITY",
    "UNKNOWN",
)


STRONG_TREATMENTS = {
    "FOLLOWED",
    "RELIED_ON",
    "APPROVED",
    "DISTINGUISHED",
    "DOUBTED",
    "OVERRULED",
    "DISAPPROVED",
}


# ============================================================
# Citation patterns
# ============================================================

CASE_NAME_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9])

    (?P<left>
        [A-Z]
        [A-Za-z0-9
        .,'’()&/@+\-:\s]{1,180}?
    )

    \s+

    (?:
        (?i:v\.?|vs\.?|versus)
    )

    \s+

    (?P<right>
        [A-Z]
        [A-Za-z0-9
        .,'’()&/@+\-:\s]{1,180}?
    )

    (?=
        \s*,\s*
        |
        \s+reported\s+in\b
        |
        \s+reported\s+as\b
        |
        \s+\[
        |
        \s+\(
        |
        \s+\d{4}\s+(?:INSC|SCC|AIR)\b
        |
        \s+\(?\d{4}\)?\s+
        Supp\.?\s*
        \(\s*\d+\s*\)
        \s+SCC\b
        |
        ;
        |
        :
        |
        \n
        |
        \.(?=\s+[A-Z0-9]|\s*$)
        |
        $
    )
    """,
    re.VERBOSE,
)


SCC_PATTERN = re.compile(
    r"""
    \(
    (?P<year>\d{4})
    \)
    \s*
    (?P<volume>\d+)
    \s+
    SCC
    \s+
    (?P<page>\d+)
    """,
    re.IGNORECASE | re.VERBOSE,
)


SCC_SUPP_PATTERN = re.compile(
    r"""
    (?<!\d)
    \(?
    (?P<year>\d{4})
    \)?
    \s+
    Supp\.?
    \s*
    \(
    \s*
    (?P<volume>\d+)
    \s*
    \)
    \s+
    SCC
    \s+
    (?P<page>\d+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


SCC_ONLINE_PATTERN = re.compile(
    r"""
    \b
    (?P<year>\d{4})
    \s+
    SCC
    \s+
    OnLine
    \s+
    (?P<court>[A-Z]{2,12})
    \s+
    (?P<number>\d+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


AIR_PATTERN = re.compile(
    r"""
    \bAIR
    \s+
    (?P<year>\d{4})
    \s+
    (?P<court>[A-Z&]{2,12})
    \s+
    (?P<page>\d+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


INSC_PATTERN = re.compile(
    r"""
    \b
    (?P<year>\d{4})
    \s+
    INSC
    \s+
    (?P<number>\d+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# Treatment patterns
# ============================================================

TREATMENT_PATTERNS: dict[
    str,
    list[tuple[str, float]],
] = {

    "OVERRULED": [
        (
            r"\bwas expressly overruled\b",
            0.99,
        ),
        (
            r"\bhas been expressly overruled\b",
            0.99,
        ),
        (
            r"\bhas been overruled\b",
            0.98,
        ),
        (
            r"\bstands overruled\b",
            0.99,
        ),
        (
            r"\boverruled by\b",
            0.98,
        ),
        (
            r"\bwe overrule\b",
            0.99,
        ),
        (
            r"\bare therefore overruled\b",
            0.98,
        ),
        (
            r"\bis therefore overruled\b",
            0.98,
        ),
    ],

    "DOUBTED": [
        (
            r"\bcorrectness .* doubted\b",
            0.97,
        ),
        (
            r"\bwas doubted\b",
            0.96,
        ),
        (
            r"\bhas been doubted\b",
            0.96,
        ),
        (
            r"\bwe doubt the correctness\b",
            0.97,
        ),
        (
            r"\bdoubted the correctness\b",
            0.95,
        ),
        (
            r"\bserious doubt .* correctness\b",
            0.94,
        ),
    ],

    "DISAPPROVED": [
        (
            r"\bwas disapproved\b",
            0.97,
        ),
        (
            r"\bhas been disapproved\b",
            0.96,
        ),
        (
            r"\bwe disapprove\b",
            0.97,
        ),
        (
            r"\bcannot be approved\b",
            0.90,
        ),
        (
            r"\bview .* cannot be accepted\b",
            0.88,
        ),
    ],

    "DISTINGUISHED": [
        (
            r"\bwas distinguished\b",
            0.97,
        ),
        (
            r"\bhas been distinguished\b",
            0.96,
        ),
        (
            r"\bwe distinguish\b",
            0.96,
        ),
        (
            r"\bdistinguishable on facts\b",
            0.94,
        ),
        (
            r"\bis distinguishable\b",
            0.93,
        ),
        (
            r"\bdistinguished on facts\b",
            0.94,
        ),
        (
            r"\bfacts .* distinguishable\b",
            0.90,
        ),
    ],

    "APPROVED": [
        (
            r"\bwas approved\b",
            0.95,
        ),
        (
            r"\bhas been approved\b",
            0.94,
        ),
        (
            r"\bwe approve\b",
            0.96,
        ),
        (
            r"\bapproved the view\b",
            0.93,
        ),
        (
            r"\bapproved the principle\b",
            0.93,
        ),
        (
            r"\bview .* approved\b",
            0.91,
        ),
    ],

    "FOLLOWED": [
        (
            r"\bwas followed\b",
            0.95,
        ),
        (
            r"\bhas been followed\b",
            0.94,
        ),
        (
            r"\bfollowing the decision\b",
            0.92,
        ),
        (
            r"\bfollowing the judgment\b",
            0.92,
        ),
        (
            r"\bwe follow\b",
            0.95,
        ),
        (
            r"\bfollowed by this court\b",
            0.93,
        ),
        (
            r"\bfollowing the law laid down\b",
            0.92,
        ),
    ],

    "RELIED_ON": [
        (
            r"\breliance was placed on\b",
            0.93,
        ),
        (
            r"\breliance has been placed on\b",
            0.92,
        ),
        (
            r"\breliance is placed on\b",
            0.92,
        ),
        (
            r"\brelying upon\b",
            0.91,
        ),
        (
            r"\brelying on\b",
            0.90,
        ),
        (
            r"\brelied upon\b",
            0.92,
        ),
        (
            r"\brelied on\b",
            0.91,
        ),
        (
            r"\bplaces reliance on\b",
            0.90,
        ),
        (
            r"\bplaced reliance on\b",
            0.92,
        ),
        (
            r"\bwe rely upon\b",
            0.96,
        ),
        (
            r"\bwe rely on\b",
            0.96,
        ),
    ],

    "REFERRED_TO": [
        (
            r"\bit was held as follows\b",
            0.84,
        ),
        (
            r"\bit has been held as follows\b",
            0.84,
        ),
        (
            r"\bit has been held that\b",
            0.82,
        ),
        (
            r"\bheld as follows\b",
            0.82,
        ),
        (
            r"\bwas explained as follows\b",
            0.84,
        ),
        (
            r"\bwas specified as follows\b",
            0.82,
        ),
        (
            r"\bwas observed as follows\b",
            0.82,
        ),
        (
            r"\bobserved as follows\b",
            0.80,
        ),
        (
            r"\bwas held that\b",
            0.80,
        ),
        (
            r"\bit was observed that\b",
            0.80,
        ),
        (
            r"\bwherein it was held\b",
            0.82,
        ),
        (
            r"\bwherein it has been held\b",
            0.82,
        ),
        (
            r"\bin which it was held\b",
            0.80,
        ),
        (
            r"\bin which it has been held\b",
            0.80,
        ),
        (
            r"\bin view of the decision in\b",
            0.78,
        ),
        (
            r"\bthese aspects were highlighted .* in\b",
            0.78,
        ),
        (
            r"\bprinciple .* laid down in\b",
            0.82,
        ),
        (
            r"\blaw laid down in\b",
            0.80,
        ),
        (
            r"\breferred to\b",
            0.75,
        ),
        (
            r"\breference was made to\b",
            0.76,
        ),
        (
            r"\bconsidered in\b",
            0.70,
        ),
        (
            r"\bobserved in\b",
            0.72,
        ),
        (
            r"\bheld in\b",
            0.72,
        ),
        (
            r"\bin .* supra\b",
            0.65,
        ),
    ],
}


TREATMENT_PRIORITY = {
    "OVERRULED": 100,
    "DOUBTED": 95,
    "DISAPPROVED": 90,
    "DISTINGUISHED": 85,
    "APPROVED": 80,
    "FOLLOWED": 75,
    "RELIED_ON": 70,
    "REFERRED_TO": 10,
}


# ============================================================
# Actor / scope patterns
# ============================================================

PARTY_CUE_PATTERNS = (
    r"\blearned counsel\b",
    r"\blearned senior counsel\b",
    r"\bcounsel for the petitioner\b",
    r"\bcounsel for the respondent\b",
    r"\bcounsel for the appellant\b",
    r"\bcounsel for the accused\b",
    r"\bcounsel for the revenue\b",
    r"\bcounsel for the prosecution\b",
    r"\bpetitioner submits\b",
    r"\brespondent submits\b",
    r"\bappellant submits\b",
    r"\brevenue submits\b",
    r"\bprosecution submits\b",
    r"\bit was submitted\b",
    r"\bit is submitted\b",
    r"\bsubmitted that\b",
    r"\bargued that\b",
    r"\bcontended that\b",
    r"\bsubmission of\b",
)


COURT_CUE_PATTERNS = (
    r"\bthis court\b",
    r"\bwe hold\b",
    r"\bwe find\b",
    r"\bwe observe\b",
    r"\bwe are of the view\b",
    r"\bwe are of the considered view\b",
    r"\bwe follow\b",
    r"\bwe rely\b",
    r"\bwe distinguish\b",
    r"\bwe approve\b",
    r"\bwe overrule\b",
    r"\bwe doubt\b",
    r"\bthe court finds\b",
    r"\bthe court holds\b",
    r"\bthe court observes\b",
)


QUOTE_INTRO_PATTERNS = (
    r"\bheld as under\b",
    r"\bheld as follows\b",
    r"\bobserved as under\b",
    r"\bobserved as follows\b",
    r"\bhas held as under\b",
    r"\bhas held as follows\b",
    r"\bhas observed as under\b",
    r"\bhas observed as follows\b",
    r"\breproduced below\b",
    r"\breproduced hereunder\b",
    r"\bquoted below\b",
    r"\breads as follows\b",
    r"\breads thus\b",
)


PREVIOUS_BRIDGE_PATTERNS = (
    r"\breliance was placed on the following\b",
    r"\breliance has been placed on the following\b",
    r"\bthe following judgments were relied\b",
    r"\bthe following decisions were relied\b",
    r"\bthe following authorities were relied\b",
    r"\bfollowing judgments\b",
    r"\bfollowing decisions\b",
)


ANAPHORIC_PATTERNS = (
    r"\bthe said decision\b",
    r"\bthe aforesaid decision\b",
    r"\bthe above decision\b",
    r"\bthis decision\b",
    r"\bthe said judgment\b",
    r"\bthe aforesaid judgment\b",
    r"\bthe above judgment\b",
    r"\bthe same\b",
)


# ============================================================
# Generic helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_spaces(
    value: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def matches_any(
    text: str,
    patterns: tuple[str, ...],
) -> bool:

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(
                    line
                )
            )

    return rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )


# ============================================================
# Sentence splitting
# ============================================================

def split_sentences(
    text: str,
) -> list[str]:

    if not text:
        return []

    text = normalize_spaces(
        text
    )

    if not text:
        return []

    # --------------------------------------------------------
    # Protect case-name connectors
    # --------------------------------------------------------

    protected = re.sub(
        r"""
        (?<=\w)
        \s+
        v\.
        \s+
        (?=[A-Z])
        """,
        " v<DOT> ",
        text,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    protected = re.sub(
        r"""
        (?<=\w)
        \s+
        vs\.
        \s+
        (?=[A-Z])
        """,
        " vs<DOT> ",
        protected,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    # --------------------------------------------------------
    # Protect common legal abbreviations
    # --------------------------------------------------------

    abbreviation_map = {
        "S.C.": "S<DOT>C<DOT>",
        "T.N.": "T<DOT>N<DOT>",
        "U.P.": "U<DOT>P<DOT>",
        "A.P.": "A<DOT>P<DOT>",
        "M.P.": "M<DOT>P<DOT>",
        "NCT.": "NCT<DOT>",
        "Ltd.": "Ltd<DOT>",
        "Pvt.": "Pvt<DOT>",
        "No.": "No<DOT>",
        "Nos.": "Nos<DOT>",
        "Dr.": "Dr<DOT>",
        "Mr.": "Mr<DOT>",
        "Mrs.": "Mrs<DOT>",
        "Ms.": "Ms<DOT>",
        "Sr.": "Sr<DOT>",
        "J.": "J<DOT>",
        "JJ.": "JJ<DOT>",
    }

    for abbreviation, replacement in (
        abbreviation_map.items()
    ):

        protected = re.sub(
            re.escape(
                abbreviation
            ),
            replacement,
            protected,
            flags=re.IGNORECASE,
        )

    # --------------------------------------------------------
    # Split sentence boundaries
    # --------------------------------------------------------

    parts = re.split(
        r"""
        (?<=[?!])
        \s+
        |
        (?<=\.)
        \s+
        (?=[A-Z0-9])
        """,
        protected,
        flags=re.VERBOSE,
    )

    sentences: list[
        str
    ] = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        part = part.replace(
            "<DOT>",
            ".",
        )

        part = normalize_spaces(
            part
        )

        if len(part) >= 20:

            sentences.append(
                part
            )

    return sentences


# ============================================================
# Case-name cleanup
# ============================================================

def clean_case_party(
    value: str,
) -> str:

    value = normalize_spaces(
        value
    )

    return value.strip(
        " ,.;:-"
    )


def clean_case_left_party(
    value: str,
) -> str:

    value = clean_case_party(
        value
    )

    introducer_patterns = (

        # ----------------------------------------------------
        # Decision / judgment introductions
        # ----------------------------------------------------

        r"^.*\bin view of the decision in\s+",
        r"^.*\bin view of the judgment in\s+",

        r"^.*\bfollowing the decision in\s+",
        r"^.*\bfollowing the judgment in\s+",

        # ----------------------------------------------------
        # Reliance language
        # ----------------------------------------------------

        r"^.*\breliance was placed on\s+",
        r"^.*\breliance was placed upon\s+",

        r"^.*\breliance has been placed on\s+",
        r"^.*\breliance has been placed upon\s+",

        r"^.*\breliance is placed on\s+",
        r"^.*\breliance is placed upon\s+",

        r"^.*\bplaced reliance on\s+",
        r"^.*\bplaced reliance upon\s+",

        r"^.*\bplaces reliance on\s+",
        r"^.*\bplaces reliance upon\s+",

        r"^.*\brelied upon\s+",
        r"^.*\brelied on\s+",

        r"^.*\brelying upon\s+",
        r"^.*\brelying on\s+",

        # ----------------------------------------------------
        # Principle/test language
        # ----------------------------------------------------

        r"^.*\bthe hallowed principle in\s+",
        r"^.*\bthe principle in\s+",

        r"^.*\btest enunciated in\s+",
        r"^.*\btest laid down in\s+",

        r"^.*\bprinciple enunciated in\s+",
        r"^.*\bprinciple laid down in\s+",

        r"^.*\blaw laid down in\s+",

        # ----------------------------------------------------
        # Multiple authorities
        # ----------------------------------------------------

        r"^.*\bsupra and\s+",

        # ----------------------------------------------------
        # Generic decision/judgment wording
        # ----------------------------------------------------

        r"^.*\bthe decision in\s+",
        r"^.*\bdecision in\s+",

        r"^.*\bthe judgment in\s+",
        r"^.*\bjudgment in\s+",

        r"^.*\bthe case of\s+",
        r"^.*\bcase of\s+",

        # ----------------------------------------------------
        # Authoritative-court wording
        # ----------------------------------------------------

        (
            r"^.*\bauthoritative judgment "
            r"rendered by "
            r"(?:the\s+)?"
            r"(?:Apex|Supreme|High) Court in\s+"
        ),

        (
            r"^.*\bjudgment rendered by "
            r"(?:the\s+)?"
            r"(?:Apex|Supreme|High) Court in\s+"
        ),

        r"^.*\bjudgment of this court in\s+",
        r"^.*\bjudgment of the court in\s+",

        # ----------------------------------------------------
        # Other introducers
        # ----------------------------------------------------

        r"^.*\bhighlighted recently in\s+",
        r"^.*\bhighlighted in\s+",

        r"^.*\badverted to\s+",
        r"^.*\badverts to\s+",

        r"^.*\breferred to\s+",
        r"^.*\breference was made to\s+",

        r"^.*\bconsidered in\s+",
        r"^.*\bobserved in\s+",
        r"^.*\bheld in\s+",
    )

    for pattern in introducer_patterns:

        cleaned = re.sub(
            pattern,
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()

        if cleaned != value:

            value = cleaned
            break

    simple_prefixes = (
        "In the case of ",
        "In ",
        "See also ",
        "See ",
    )

    changed = True

    while changed:

        changed = False

        for prefix in simple_prefixes:

            if value.lower().startswith(
                prefix.lower()
            ):

                value = value[
                    len(prefix):
                ].strip()

                changed = True
                break

    return clean_case_party(
        value
    )


# ============================================================
# Citation extraction
# ============================================================

def extract_case_names(
    text: str,
) -> list[str]:

    citations: list[
        str
    ] = []

    for match in CASE_NAME_PATTERN.finditer(
        text
    ):

        left = clean_case_left_party(
            match.group(
                "left"
            )
        )

        right = clean_case_party(
            match.group(
                "right"
            )
        )

        if not left or not right:
            continue

        if (
            len(left) > 140
            or len(right) > 180
        ):
            continue

        if (
            len(left) < 2
            or len(right) < 2
        ):
            continue

        if not re.search(
            r"[A-Za-z]",
            left,
        ):
            continue

        if not re.search(
            r"[A-Za-z]",
            right,
        ):
            continue

        citations.append(
            f"{left} v. {right}"
        )

    return citations


def extract_reporter_citations(
    text: str,
) -> list[str]:

    citations: list[
        str
    ] = []

    for pattern in (
        SCC_PATTERN,
        SCC_SUPP_PATTERN,
        SCC_ONLINE_PATTERN,
        AIR_PATTERN,
        INSC_PATTERN,
    ):

        for match in pattern.finditer(
            text
        ):

            citations.append(
                normalize_spaces(
                    match.group(0)
                )
            )

    return citations


def extract_citation_mentions(
    text: str,
) -> list[str]:

    mentions: list[
        str
    ] = []

    mentions.extend(
        extract_case_names(
            text
        )
    )

    mentions.extend(
        extract_reporter_citations(
            text
        )
    )

    seen: set[
        str
    ] = set()

    unique: list[
        str
    ] = []

    for mention in mentions:

        key = normalize_spaces(
            mention
        ).lower()

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            mention
        )

    return unique


# ============================================================
# Treatment signal extraction
# ============================================================

def is_negated_reliance(
    text: str,
    match_start: int,
) -> bool:
    """
    Reject evidentiary / negative constructions such as:

        cannot be relied upon
        could not be relied on
        should not be relied upon
        would not be relied on
        must not be relied upon
        may not be relied on
        not be relied upon

    These phrases do not mean that the cited precedent itself was
    RELIED_ON.
    """

    prefix = text[
        max(
            0,
            match_start - 60,
        ):
        match_start
    ]

    return bool(
        re.search(
            r"""
            \b
            (?:
                cannot
                |
                can't
                |
                could\s+not
                |
                should\s+not
                |
                would\s+not
                |
                must\s+not
                |
                may\s+not
                |
                shall\s+not
                |
                not
            )
            \s+
            (?:be\s+)?
            $
            """,
            prefix,
            flags=(
                re.IGNORECASE
                | re.VERBOSE
            ),
        )
    )


def find_treatment_signal(
    text: str,
) -> tuple[
    str,
    float,
    str,
] | None:

    candidates: list[
        tuple[
            str,
            float,
            str,
            int,
        ]
    ] = []

    for label, patterns in (
        TREATMENT_PATTERNS.items()
    ):

        for pattern, confidence in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            # ------------------------------------------------
            # Do not interpret negative evidentiary language
            # such as "cannot be relied upon" as precedent
            # reliance.
            # ------------------------------------------------

            if (
                label == "RELIED_ON"
                and is_negated_reliance(
                    text,
                    match.start(),
                )
            ):
                continue

            candidates.append(
                (
                    label,
                    confidence,
                    normalize_spaces(
                        match.group(0)
                    ),
                    TREATMENT_PRIORITY[
                        label
                    ],
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[3],
            item[1],
        ),
        reverse=True,
    )

    label, confidence, trigger, _ = (
        candidates[0]
    )

    return (
        label,
        confidence,
        trigger,
    )


def resolve_treatment(
    sentence: str,
    previous_sentence: str,
    next_sentence: str,
) -> tuple[
    str,
    float,
    str,
    str,
]:
    """
    Resolve treatment using citation-local context.

    Priority:

        1. Citation sentence itself
        2. Previous sentence only when it clearly introduces
           the cited authorities
        3. Next sentence only when it clearly refers back
        4. REFERRED_TO neutral fallback
    """

    # --------------------------------------------------------
    # Direct sentence
    # --------------------------------------------------------

    direct = find_treatment_signal(
        sentence
    )

    if direct is not None:

        label, confidence, trigger = (
            direct
        )

        return (
            label,
            confidence,
            trigger,
            "DIRECT_SENTENCE",
        )

    # --------------------------------------------------------
    # Previous-sentence bridge
    # --------------------------------------------------------

    previous_signal = (
        find_treatment_signal(
            previous_sentence
        )
        if previous_sentence
        else None
    )

    if (
        previous_signal is not None
        and matches_any(
            previous_sentence,
            PREVIOUS_BRIDGE_PATTERNS,
        )
    ):

        label, confidence, trigger = (
            previous_signal
        )

        return (
            label,
            max(
                0.50,
                confidence - 0.03,
            ),
            trigger,
            "PREVIOUS_SENTENCE",
        )

    # --------------------------------------------------------
    # Next-sentence anaphora
    # --------------------------------------------------------

    next_signal = (
        find_treatment_signal(
            next_sentence
        )
        if next_sentence
        else None
    )

    if (
        next_signal is not None
        and matches_any(
            next_sentence,
            ANAPHORIC_PATTERNS,
        )
    ):

        label, confidence, trigger = (
            next_signal
        )

        return (
            label,
            max(
                0.50,
                confidence - 0.05,
            ),
            trigger,
            "NEXT_SENTENCE",
        )

    # --------------------------------------------------------
    # Conservative default
    # --------------------------------------------------------

    return (
        "REFERRED_TO",
        0.50,
        "",
        "NEUTRAL_FALLBACK",
    )


# ============================================================
# Actor detection
# ============================================================

def quoted_authority_context(
    sentence: str,
    previous_sentence: str,
) -> bool:

    # --------------------------------------------------------
    # Explicit quotation
    # --------------------------------------------------------

    if sentence.lstrip().startswith(
        (
            '"',
            "'",
            "“",
            "‘",
        )
    ):
        return True

    if not previous_sentence:
        return False

    # --------------------------------------------------------
    # Previous sentence must introduce quoted material
    # --------------------------------------------------------

    if not matches_any(
        previous_sentence,
        QUOTE_INTRO_PATTERNS,
    ):
        return False

    # --------------------------------------------------------
    # Previous sentence should also identify an authority
    # --------------------------------------------------------

    has_authority_reference = bool(
        extract_citation_mentions(
            previous_sentence
        )
    ) or bool(
        re.search(
            r"""
            \b
            (?:
                Supreme\s+Court
                |
                Apex\s+Court
                |
                High\s+Court
                |
                Privy\s+Council
            )
            \b
            """,
            previous_sentence,
            flags=(
                re.IGNORECASE
                | re.VERBOSE
            ),
        )
    )

    return has_authority_reference


def infer_treatment_actor(
    section_type: str,
    sentence: str,
    previous_sentence: str,
    treatment: str,
    trigger: str,
) -> str:

    section_type = (
        section_type
        or ""
    ).upper()

    # --------------------------------------------------------
    # Quoted authority gets precedence
    # --------------------------------------------------------

    if quoted_authority_context(
        sentence,
        previous_sentence,
    ):

        return "QUOTED_AUTHORITY"

    # --------------------------------------------------------
    # Explicit party/counsel language
    # --------------------------------------------------------

    if matches_any(
        sentence,
        PARTY_CUE_PATTERNS,
    ):

        return "PARTY"

    # --------------------------------------------------------
    # ARGUMENTS generally represents party material
    # --------------------------------------------------------

    if (
        section_type == "ARGUMENTS"
        and not matches_any(
            sentence,
            COURT_CUE_PATTERNS,
        )
    ):

        return "PARTY"

    # --------------------------------------------------------
    # Passive reliance normally reflects submissions unless
    # current-court language explicitly adopts the authority.
    # --------------------------------------------------------

    if (
        treatment == "RELIED_ON"
        and trigger.lower()
        in {
            "reliance was placed on",
            "reliance has been placed on",
            "reliance is placed on",
            "placed reliance on",
            "places reliance on",
            "relying on",
            "relying upon",
            "relied on",
            "relied upon",
        }
        and not matches_any(
            sentence,
            COURT_CUE_PATTERNS,
        )
    ):

        return "PARTY"

    # --------------------------------------------------------
    # Explicit current-court language
    # --------------------------------------------------------

    if matches_any(
        sentence,
        COURT_CUE_PATTERNS,
    ):

        return "COURT"

    # --------------------------------------------------------
    # Strong treatment in court-analysis sections
    # --------------------------------------------------------

    if (
        section_type
        in {
            "REASONING",
            "HOLDING",
            "ORDER",
        }
        and treatment
        in STRONG_TREATMENTS
    ):

        return "COURT"

    return "UNKNOWN"


# ============================================================
# Citation context creation
# ============================================================

def citation_contexts(
    sentences: list[str],
) -> list[
    tuple[
        int,
        str,
        str,
        str,
        str,
        list[str],
    ]
]:

    results: list[
        tuple[
            int,
            str,
            str,
            str,
            str,
            list[str],
        ]
    ] = []

    for index, sentence in enumerate(
        sentences
    ):

        mentions = extract_citation_mentions(
            sentence
        )

        if not mentions:
            continue

        previous_sentence = (
            sentences[index - 1]
            if index > 0
            else ""
        )

        next_sentence = (
            sentences[index + 1]
            if index + 1 < len(sentences)
            else ""
        )

        context = normalize_spaces(
            " ".join(
                part
                for part in (
                    previous_sentence,
                    sentence,
                    next_sentence,
                )
                if part
            )
        )

        results.append(
            (
                index,
                previous_sentence,
                sentence,
                next_sentence,
                context,
                mentions,
            )
        )

    return results


# ============================================================
# Citation-local treatment window
# ============================================================

def citation_local_window(
    sentence: str,
    citation_text: str,
    radius: int = 140,
) -> str:
    """
    Return a bounded piece of the sentence around one citation.

    This prevents a treatment phrase that refers to a different
    proposition, piece of evidence, or another citation elsewhere
    in the same sentence from being copied to every citation in
    that sentence.
    """

    if not sentence:
        return ""

    if not citation_text:
        return sentence

    normalized_sentence = (
        sentence.lower()
    )

    normalized_citation = (
        citation_text.lower()
    )

    start = normalized_sentence.find(
        normalized_citation
    )

    if start < 0:
        return sentence

    end = (
        start
        + len(citation_text)
    )

    window_start = max(
        0,
        start - radius,
    )

    window_end = min(
        len(sentence),
        end + radius,
    )

    return sentence[
        window_start:
        window_end
    ]


# ============================================================
# Chunk processing
# ============================================================

def process_chunk(
    chunk: dict[str, Any],
) -> list[dict[str, Any]]:

    text = str(
        chunk.get(
            "text",
            "",
        )
    )

    sentences = split_sentences(
        text
    )

    contexts = citation_contexts(
        sentences
    )

    output: list[
        dict[str, Any]
    ] = []

    local_counter = 0

    section_type = str(
        chunk.get(
            "section_type",
            "",
        )
    )

    for (
        sentence_index,
        previous_sentence,
        citation_sentence,
        next_sentence,
        context,
        mentions,
    ) in contexts:

        for citation_text in mentions:

            # ------------------------------------------------
            # Treatment is resolved for this citation mention,
            # not once for the whole sentence.
            # ------------------------------------------------

            local_sentence = (
                citation_local_window(
                    citation_sentence,
                    citation_text,
                )
            )

            (
                treatment,
                confidence,
                trigger,
                treatment_scope,
            ) = resolve_treatment(
                local_sentence,
                previous_sentence,
                next_sentence,
            )

            actor = infer_treatment_actor(
                section_type=section_type,
                sentence=citation_sentence,
                previous_sentence=(
                    previous_sentence
                ),
                treatment=treatment,
                trigger=trigger,
            )

            is_judicial_treatment = (
                actor == "COURT"
                and treatment
                in STRONG_TREATMENTS
            )

            local_counter += 1

            output.append(
                {
                    "treatment_id": (
                        f"{chunk.get('chunk_id', '')}"
                        f"_T_{local_counter:03d}"
                    ),

                    "document_id": (
                        chunk.get(
                            "document_id",
                            "",
                        )
                    ),

                    "chunk_id": (
                        chunk.get(
                            "chunk_id",
                            "",
                        )
                    ),

                    "section_type": (
                        section_type
                    ),

                    "page_start": (
                        chunk.get(
                            "page_start"
                        )
                    ),

                    "page_end": (
                        chunk.get(
                            "page_end"
                        )
                    ),

                    "citation_text": (
                        citation_text
                    ),

                    # Canonical resolution happens later.
                    "target_case_id": None,

                    "treatment": (
                        treatment
                    ),

                    "treatment_actor": (
                        actor
                    ),

                    "treatment_scope": (
                        treatment_scope
                    ),

                    "is_judicial_treatment": (
                        is_judicial_treatment
                    ),

                    "confidence": round(
                        confidence,
                        3,
                    ),

                    "trigger": (
                        trigger
                    ),

                    "previous_sentence": (
                        previous_sentence
                    ),

                    "citation_sentence": (
                        citation_sentence
                    ),

                    "next_sentence": (
                        next_sentence
                    ),

                    "context": (
                        context
                    ),

                    "sentence_index": (
                        sentence_index
                    ),

                    "court": (
                        chunk.get(
                            "court",
                            "",
                        )
                    ),

                    "year": (
                        chunk.get(
                            "year"
                        )
                    ),

                    "detector_version": (
                        DETECTOR_VERSION
                    ),

                    "detected_at": (
                        utc_now()
                    ),
                }
            )

    return output


# ============================================================
# Deduplication
# ============================================================

def treatment_dedupe_key(
    row: dict[str, Any],
) -> tuple[
    str,
    str,
    str,
    str,
    str,
]:

    return (
        str(
            row.get(
                "document_id",
                "",
            )
        ),

        normalize_spaces(
            str(
                row.get(
                    "citation_text",
                    "",
                )
            )
        ).lower(),

        str(
            row.get(
                "treatment",
                "",
            )
        ).upper(),

        str(
            row.get(
                "treatment_actor",
                "",
            )
        ).upper(),

        normalize_spaces(
            str(
                row.get(
                    "citation_sentence",
                    "",
                )
            )
        ).lower(),
    )


def deduplicate_treatments(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    best: dict[
        tuple[
            str,
            str,
            str,
            str,
            str,
        ],
        dict[str, Any],
    ] = {}

    order: list[
        tuple[
            str,
            str,
            str,
            str,
            str,
        ]
    ] = []

    for row in rows:

        key = treatment_dedupe_key(
            row
        )

        existing = best.get(
            key
        )

        if existing is None:

            best[
                key
            ] = row

            order.append(
                key
            )

            continue

        if float(
            row.get(
                "confidence",
                0.0,
            )
        ) > float(
            existing.get(
                "confidence",
                0.0,
            )
        ):

            best[
                key
            ] = row

    return [
        best[
            key
        ]
        for key
        in order
    ]


# ============================================================
# Reporting
# ============================================================

REPORT_FIELDS = [
    "document_id",
    "citation_mentions",

    "followed",
    "relied_on",
    "approved",
    "distinguished",
    "doubted",
    "overruled",
    "disapproved",
    "referred_to",

    "court_actor",
    "party_actor",
    "quoted_authority_actor",
    "unknown_actor",

    "judicial_treatments",
]


def write_report(
    treatments: list[dict[str, Any]],
) -> None:

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    grouped: dict[
        str,
        Counter,
    ] = {}

    for row in treatments:

        document_id = str(
            row.get(
                "document_id",
                "",
            )
        )

        if document_id not in grouped:

            grouped[
                document_id
            ] = Counter()

        counts = grouped[
            document_id
        ]

        counts[
            "citation_mentions"
        ] += 1

        treatment = str(
            row.get(
                "treatment",
                "REFERRED_TO",
            )
        ).lower()

        counts[
            treatment
        ] += 1

        actor = str(
            row.get(
                "treatment_actor",
                "UNKNOWN",
            )
        ).lower()

        counts[
            f"{actor}_actor"
        ] += 1

        if bool(
            row.get(
                "is_judicial_treatment",
                False,
            )
        ):

            counts[
                "judicial_treatments"
            ] += 1

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=REPORT_FIELDS,
        )

        writer.writeheader()

        for document_id in sorted(
            grouped
        ):

            counts = grouped[
                document_id
            ]

            output_row: dict[
                str,
                Any,
            ] = {
                "document_id": (
                    document_id
                )
            }

            for field in REPORT_FIELDS:

                if field == "document_id":
                    continue

                output_row[
                    field
                ] = counts.get(
                    field,
                    0,
                )

            writer.writerow(
                output_row
            )


# ============================================================
# Chunk filtering
# ============================================================

def filter_chunks(
    chunks: list[dict[str, Any]],
    limit: int | None,
    document_id: str | None,
) -> list[dict[str, Any]]:

    if document_id:

        chunks = [
            chunk
            for chunk in chunks
            if str(
                chunk.get(
                    "document_id",
                    "",
                )
            )
            == document_id
        ]

        if not chunks:

            raise ValueError(
                "No chunks found for "
                f"{document_id}"
            )

    if limit is None:

        return chunks

    if limit <= 0:

        raise ValueError(
            "--limit must be greater "
            "than zero"
        )

    selected_documents: list[
        str
    ] = []

    filtered_chunks: list[
        dict[str, Any]
    ] = []

    for chunk in chunks:

        doc_id = str(
            chunk.get(
                "document_id",
                "",
            )
        )

        if (
            doc_id
            not in selected_documents
        ):

            if (
                len(
                    selected_documents
                )
                >= limit
            ):
                continue

            selected_documents.append(
                doc_id
            )

        if doc_id in selected_documents:

            filtered_chunks.append(
                chunk
            )

    return filtered_chunks


# ============================================================
# Pipeline
# ============================================================

def run_detection(
    limit: int | None = None,
    document_id: str | None = None,
    min_confidence: float = 0.0,
) -> None:

    if not (
        0.0
        <= min_confidence
        <= 1.0
    ):

        raise ValueError(
            "--min-confidence must be "
            "between 0.0 and 1.0"
        )

    # --------------------------------------------------------
    # Load legal chunks
    # --------------------------------------------------------

    chunks = read_jsonl(
        LEGAL_CHUNKS_PATH
    )

    chunks = filter_chunks(
        chunks=chunks,
        limit=limit,
        document_id=document_id,
    )

    documents = sorted(
        {
            str(
                chunk.get(
                    "document_id",
                    "",
                )
            )
            for chunk in chunks
            if chunk.get(
                "document_id"
            )
        }
    )

    print()

    print(
        "Citation Treatment Detection"
    )

    print("=" * 70)

    print(
        f"Detector version: "
        f"{DETECTOR_VERSION}"
    )

    print(
        f"Documents selected: "
        f"{len(documents)}"
    )

    print(
        f"Chunks selected: "
        f"{len(chunks)}"
    )

    print(
        f"Minimum confidence: "
        f"{min_confidence:.2f}"
    )

    print("=" * 70)

    treatments: list[
        dict[str, Any]
    ] = []

    raw_detection_count = 0

    duplicate_count = 0

    # --------------------------------------------------------
    # Process each judgment independently
    # --------------------------------------------------------

    for index, doc_id in enumerate(
        documents,
        start=1,
    ):

        document_chunks = [
            chunk
            for chunk in chunks
            if str(
                chunk.get(
                    "document_id",
                    "",
                )
            )
            == doc_id
        ]

        raw_rows: list[
            dict[str, Any]
        ] = []

        for chunk in document_chunks:

            raw_rows.extend(
                process_chunk(
                    chunk
                )
            )

        raw_detection_count += len(
            raw_rows
        )

        document_rows = (
            deduplicate_treatments(
                raw_rows
            )
        )

        duplicates_removed = (
            len(
                raw_rows
            )
            - len(
                document_rows
            )
        )

        duplicate_count += (
            duplicates_removed
        )

        document_rows = [
            row
            for row in document_rows
            if float(
                row.get(
                    "confidence",
                    0.0,
                )
            )
            >= min_confidence
        ]

        treatments.extend(
            document_rows
        )

        treatment_counts = Counter(
            row[
                "treatment"
            ]
            for row
            in document_rows
        )

        actor_counts = Counter(
            row[
                "treatment_actor"
            ]
            for row
            in document_rows
        )

        judicial_count = sum(
            1
            for row in document_rows
            if row[
                "is_judicial_treatment"
            ]
        )

        print(
            f"[{index}/{len(documents)}] "
            f"{doc_id}"
            f" | citations="
            f"{len(document_rows)}"
            f" | deduped="
            f"{duplicates_removed}"
            f" | judicial="
            f"{judicial_count}"
            f" | party="
            f"{actor_counts.get('PARTY', 0)}"
            f" | quoted="
            f"{actor_counts.get('QUOTED_AUTHORITY', 0)}"
            f" | followed="
            f"{treatment_counts.get('FOLLOWED', 0)}"
            f" | relied="
            f"{treatment_counts.get('RELIED_ON', 0)}"
        )

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    write_jsonl(
        TREATMENTS_PATH,
        treatments,
    )

    write_report(
        treatments
    )

    # --------------------------------------------------------
    # Corpus totals
    # --------------------------------------------------------

    total_treatments = Counter(
        row[
            "treatment"
        ]
        for row
        in treatments
    )

    total_actors = Counter(
        row[
            "treatment_actor"
        ]
        for row
        in treatments
    )

    judicial_rows = [
        row
        for row in treatments
        if row.get(
            "is_judicial_treatment"
        )
    ]

    judicial_labels = Counter(
        row[
            "treatment"
        ]
        for row
        in judicial_rows
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()

    print(
        "Citation Treatment Report"
    )

    print("=" * 70)

    print(
        f"Documents processed: "
        f"{len(documents)}"
    )

    print(
        f"Raw citation detections: "
        f"{raw_detection_count}"
    )

    print(
        f"Overlap duplicates removed: "
        f"{duplicate_count}"
    )

    print(
        f"Citation mentions retained: "
        f"{len(treatments)}"
    )

    print()

    for label in TREATMENT_LABELS:

        print(
            f"{label}: "
            f"{total_treatments.get(label, 0)}"
        )

    print()

    print(
        "Treatment actors"
    )

    print("-" * 70)

    for actor in TREATMENT_ACTORS:

        print(
            f"{actor}: "
            f"{total_actors.get(actor, 0)}"
        )

    print()

    print(
        f"Judicial non-neutral treatments: "
        f"{len(judicial_rows)}"
    )

    for label in (
        "FOLLOWED",
        "RELIED_ON",
        "APPROVED",
        "DISTINGUISHED",
        "DOUBTED",
        "OVERRULED",
        "DISAPPROVED",
    ):

        print(
            f"JUDICIAL_{label}: "
            f"{judicial_labels.get(label, 0)}"
        )

    # --------------------------------------------------------
    # Canonical targets are intentionally unresolved here
    # --------------------------------------------------------

    unresolved = sum(
        1
        for row in treatments
        if not row.get(
            "target_case_id"
        )
    )

    print()

    print(
        f"Canonical targets unresolved: "
        f"{unresolved}"
    )

    print(
        f"Treatments: "
        f"{TREATMENTS_PATH}"
    )

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Detect actor-aware legal "
            "citation treatments."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process the first N "
            "documents."
        ),
    )

    parser.add_argument(
        "--document-id",
        default=None,
        help=(
            "Process one document only."
        ),
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "Minimum confidence "
            "to retain."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    try:

        run_detection(
            limit=args.limit,
            document_id=(
                args.document_id
            ),
            min_confidence=(
                args.min_confidence
            ),
        )

    except Exception as exc:

        print()

        print(
            "Citation treatment "
            "detection failed"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print("=" * 70)

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()