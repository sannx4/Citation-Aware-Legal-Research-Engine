from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_TREATMENTS_PATH = (
    PROCESSED_DIR
    / "citation_treatments.jsonl"
)

RESOLVED_TREATMENTS_PATH = (
    PROCESSED_DIR
    / "citation_treatments_resolved.jsonl"
)

AUTHORITY_REGISTRY_PATH = (
    PROCESSED_DIR
    / "canonical_citation_registry.jsonl"
)

DOCUMENTS_DIR = (
    PROCESSED_DIR
    / "documents"
)

RAW_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "manifest_deduped.csv"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_PATH = (
    REPORTS_DIR
    / "citation_resolution_report.csv"
)


RESOLVER_VERSION = "1.4.1"


# ============================================================
# Citation patterns
# ============================================================

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


CASE_CONNECTOR_PATTERN = re.compile(
    r"""
    \s+
    (?:
        v\.?
        |
        vs\.?
        |
        versus
    )
    \s+
    """,
    re.IGNORECASE | re.VERBOSE,
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
# Mention classification
# ============================================================

def reporter_type(
    value: str,
) -> str | None:

    value = normalize_spaces(
        value
    )

    if SCC_PATTERN.fullmatch(
        value
    ):
        return "SCC"

    if SCC_SUPP_PATTERN.fullmatch(
        value
    ):
        return "SCC_SUPP"

    if SCC_ONLINE_PATTERN.fullmatch(
        value
    ):
        return "SCC_ONLINE"

    if AIR_PATTERN.fullmatch(
        value
    ):
        return "AIR"

    if INSC_PATTERN.fullmatch(
        value
    ):
        return "INSC"

    return None


def is_reporter_citation(
    value: str,
) -> bool:

    return (
        reporter_type(
            value
        )
        is not None
    )


def is_case_name(
    value: str,
) -> bool:

    if is_reporter_citation(
        value
    ):
        return False

    return bool(
        CASE_CONNECTOR_PATTERN.search(
            value
        )
    )


# ============================================================
# Reporter normalization
# ============================================================

def normalize_reporter(
    value: str,
) -> str:

    value = normalize_spaces(
        value
    )

    # --------------------------------------------------------
    # Standard SCC
    # --------------------------------------------------------

    match = SCC_PATTERN.fullmatch(
        value
    )

    if match:

        return (
            f"({match.group('year')}) "
            f"{int(match.group('volume'))} "
            f"SCC "
            f"{int(match.group('page'))}"
        )

    # --------------------------------------------------------
    # Supplement SCC
    # --------------------------------------------------------

    match = SCC_SUPP_PATTERN.fullmatch(
        value
    )

    if match:

        return (
            f"{match.group('year')} "
            f"Supp ({int(match.group('volume'))}) "
            f"SCC "
            f"{int(match.group('page'))}"
        )

    # --------------------------------------------------------
    # SCC OnLine
    # --------------------------------------------------------

    match = SCC_ONLINE_PATTERN.fullmatch(
        value
    )

    if match:

        return (
            f"{match.group('year')} "
            f"SCC OnLine "
            f"{match.group('court').upper()} "
            f"{int(match.group('number'))}"
        )

    # --------------------------------------------------------
    # AIR
    # --------------------------------------------------------

    match = AIR_PATTERN.fullmatch(
        value
    )

    if match:

        return (
            f"AIR "
            f"{match.group('year')} "
            f"{match.group('court').upper()} "
            f"{int(match.group('page'))}"
        )

    # --------------------------------------------------------
    # INSC
    # --------------------------------------------------------

    match = INSC_PATTERN.fullmatch(
        value
    )

    if match:

        return (
            f"{match.group('year')} "
            f"INSC "
            f"{int(match.group('number'))}"
        )

    return value


# ============================================================
# Case-name normalization
# ============================================================

def remove_footnote_noise(
    value: str,
) -> str:
    """
    Remove common footnote / OCR artifacts.

    Examples:

        Uttarakhand 26
            ->
        Uttarakhand

        Bhartiya18
            ->
        Bhartiya

        another1
            ->
        another
    """

    # --------------------------------------------------------
    # Number directly attached to a word
    # --------------------------------------------------------

    value = re.sub(
        r"(?<=[A-Za-z)])\d+\b",
        "",
        value,
    )

    # --------------------------------------------------------
    # Trailing standalone footnote
    # --------------------------------------------------------

    value = re.sub(
        r"\s+\d+\s*$",
        "",
        value,
    )

    return normalize_spaces(
        value
    )


def clean_case_prefix(
    value: str,
) -> str:
    """
    Remove legal prose accidentally captured before the true
    first party name.

    Examples:

        Learned Senior Counsel submitted that all four limbs of
        the test enunciated in Rajiv Thapar v. Madan Lal Kapoor

            ->

        Rajiv Thapar v. Madan Lal Kapoor


        Now it is relevant to consider the authoritative judgment
        rendered by the Apex Court in State of Haryana v. Bhajan Lal

            ->

        State of Haryana v. Bhajan Lal


        Reliance is properly placed ... on Mahmood Ali supra and
        Abhishek v. State of M.P.

            ->

        Abhishek v. State of M.P.


        Reliance on Union of India v. NHPC Ltd.

            ->

        Union of India v. NHPC Ltd.
    """

    value = normalize_spaces(
        value
    )

    # --------------------------------------------------------
    # Strong legal introducers
    # --------------------------------------------------------
    #
    # These use greedy prefix matching deliberately. The aim is
    # to keep the text occurring after the legally meaningful
    # introducer nearest the actual case name.
    # --------------------------------------------------------

    patterns = (

        # ----------------------------------------------------
        # Reliance formulations
        # ----------------------------------------------------

        r"^.*\breliance\s+was\s+placed\s+upon\s+",
        r"^.*\breliance\s+was\s+placed\s+on\s+",

        r"^.*\breliance\s+has\s+been\s+placed\s+upon\s+",
        r"^.*\breliance\s+has\s+been\s+placed\s+on\s+",

        r"^.*\breliance\s+is\s+placed\s+upon\s+",
        r"^.*\breliance\s+is\s+placed\s+on\s+",

        r"^.*\breliance\s+on\s+",

        r"^.*\bplaced\s+reliance\s+upon\s+",
        r"^.*\bplaced\s+reliance\s+on\s+",

        r"^.*\bplaces\s+reliance\s+upon\s+",
        r"^.*\bplaces\s+reliance\s+on\s+",

        r"^.*\brelying\s+upon\s+",
        r"^.*\brelying\s+on\s+",

        r"^.*\brelied\s+upon\s+",
        r"^.*\brelied\s+on\s+",

        r"^.*\brelies\s+upon\s+",
        r"^.*\brelies\s+on\s+",

        r"^.*\bplacing\s+reliance\s+upon\s+",
        r"^.*\bplacing\s+reliance\s+on\s+",

        r"^.*\breliance\s+being\s+placed\s+upon\s+",
        r"^.*\breliance\s+being\s+placed\s+on\s+",

        (
            r"^.*\breliance\s+is\s+placed\b.*?\bupon\s+"
            r"(?:the\s+judgments?\s+of\s+)?"
        ),

        (
            r"^.*\breliance\s+is\s+placed\b.*?\bon\s+"
            r"(?:the\s+judgments?\s+of\s+)?"
        ),

        # ----------------------------------------------------
        # Multiple-authority constructions
        #
        # Example:
        #
        # Mahmood Ali supra and Abhishek v. State of M.P.
        # ----------------------------------------------------

        r"^.*\bsupra\s+and\s+",

        # ----------------------------------------------------
        # Test / principle / proposition language
        # ----------------------------------------------------

        r"^.*\btest\s+enunciated\s+in\s+",
        r"^.*\btest\s+laid\s+down\s+in\s+",

        r"^.*\bprinciple\s+enunciated\s+in\s+",
        r"^.*\bprinciple\s+laid\s+down\s+in\s+",

        r"^.*\bprinciples\s+enunciated\s+in\s+",
        r"^.*\bprinciples\s+laid\s+down\s+in\s+",

        r"^.*\blaw\s+laid\s+down\s+in\s+",
        r"^.*\blaw\s+laid\s+down\s+by\s+this\s+court\s+in\s+",

        r"^.*\bratios?\s+in\s+",

        (
            r"^.*\bprinciples?\s+enunciated\s+by\s+"
            r"(?:the\s+)?(?:Constitution\s+Bench|Court)\s+in\s+"
        ),

        r"^.*\blegal\s+position\s+was\s+later\s+revisited\s+in\s+",
        r"^.*\brevisited\s+in\s+",

        # ----------------------------------------------------
        # Authoritative judgment introductions
        # ----------------------------------------------------

        (
            r"^.*\bauthoritative\s+judgment\s+"
            r"rendered\s+by\s+"
            r"(?:the\s+)?"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\bjudgment\s+rendered\s+by\s+"
            r"(?:Hon['’]ble\s+)?"
            r"(?:the\s+)?"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\bauthoritative\s+pronouncement\s+of\s+"
            r"(?:the\s+)?(?:Hon['’]ble\s+)?"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\b(?:Hon['’]ble\s+)?"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\bdecision\s+of\s+"
            r"(?:the\s+)?(?:Hon['’]ble\s+)?"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\bjudgment\s+of\s+"
            r"(?:the\s+)?(?:Hon['’]ble\s+)?"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\bdecision\s+of\s+the\s+"
            r"[A-Za-z ]+High\s+Court\s+in\s+"
        ),

        (
            r"^.*\b(?:a\s+)?learned\s+"
            r"(?:Single\s+Judge|Division\s+Bench)\s+of\s+"
            r"(?:the\s+)?[A-Za-z ]+High\s+Court\s+in\s+"
        ),

        (
            r"^.*\bobservations?\s+of\s+"
            r"(?:a\s+|the\s+)?"
            r"(?:Full\s+Bench|Division\s+Bench|Single\s+Judge)"
            r".*?\bin\s+"
        ),

        (
            r"^.*\bdecision\s+of\s+"
            r"(?:a\s+|the\s+)?"
            r"(?:Full\s+Bench|Division\s+Bench|Single\s+Judge)"
            r".*?\bin\s+"
        ),

        # ----------------------------------------------------
        # This-court judgment references
        # ----------------------------------------------------

        r"^.*\bjudgment\s+of\s+this\s+court\s+in\s+",
        r"^.*\bjudgment\s+of\s+the\s+court\s+in\s+",
        r"^.*\bdecision\s+of\s+this\s+court\s+in\s+",
        r"^.*\bdecision\s+of\s+the\s+court\s+in\s+",

        (
            r"^.*\bjudgment\s+of\s+"
            r"(?:a\s+|the\s+)?"
            r"(?:Full\s+Bench|Division\s+Bench|Single\s+Judge)"
            r"\s+of\s+this\s+Court\s+in\s+"
        ),

        (
            r"^.*\bdecision\s+of\s+"
            r"(?:a\s+|the\s+)?"
            r"(?:Full\s+Bench|Division\s+Bench|Single\s+Judge)"
            r"\s+of\s+this\s+Court\s+in\s+"
        ),

        (
            r"^.*\brefer\s+the\s+judgment\s+"
            r"of\s+this\s+court\s+in\s+"
        ),

        (
            r"^.*\brefer\s+to\s+the\s+judgment\s+"
            r"of\s+this\s+court\s+in\s+"
        ),

        (
            r"^.*\brefer\s+the\s+decision\s+"
            r"of\s+this\s+court\s+in\s+"
        ),

        (
            r"^.*\brefer\s+to\s+the\s+decision\s+"
            r"of\s+this\s+court\s+in\s+"
        ),

        # ----------------------------------------------------
        # In-view-of forms
        # ----------------------------------------------------

        r"^.*\bin\s+view\s+of\s+the\s+decision\s+in\s+",
        r"^.*\bin\s+view\s+of\s+the\s+judgment\s+in\s+",

        # ----------------------------------------------------
        # Rendered / reported / titled forms
        # ----------------------------------------------------

        r"^.*\bdecisions?\s+rendered\s+in\s+",
        r"^.*\bjudgments?\s+rendered\s+in\s+",

        r"^.*\bjudgment\s+reported\s+in\s+",
        r"^.*\bdecision\s+reported\s+in\s+",

        r"^.*\bcase\s+titled\s+",
        r"^.*\btitled\s+",

        # reporter followed by case name
        (
            r"^.*\bAIR\s+\d{4}\s+[A-Z&]{2,12}\s+\d+"
            r"(?:\s*\(para\s+\d+\))?\s+"
        ),

        # Company Cases citation followed by parenthesized case name
        r"^.*\bCompany\s+Cases\s+\d+\s*\(\s*",

        # ----------------------------------------------------
        # Generic decision / judgment forms
        # ----------------------------------------------------

        r"^.*\bdecision\s+in\s+the\s+case\s+of\s+",

        r"^.*\bthe\s+decision\s+in\s+",
        r"^.*\bdecision\s+in\s+",

        r"^.*\bthe\s+judgment\s+in\s+",
        r"^.*\bjudgment\s+in\s+",

        r"^.*\bthe\s+case\s+of\s+",
        r"^.*\bcase\s+of\s+",

        # ----------------------------------------------------
        # Other common judicial citation introducers
        # ----------------------------------------------------

        r"^.*\bhighlighted\s+recently\s+in\s+",
        r"^.*\bhighlighted\s+in\s+",

        r"^.*\badverted\s+to\s+",
        r"^.*\badverts\s+to\s+",

        r"^.*\breferred\s+to\s+",
        r"^.*\breference\s+was\s+made\s+to\s+",

        r"^.*\bconsidered\s+in\s+",
        r"^.*\bobserved\s+in\s+",
        r"^.*\bheld\s+in\s+",

        r"^.*\bquoted\s+.*?\bjudgment\s+.*?\bin\s+",
        r"^.*\bfind\s+the\s+ratio\s+in\s+",
        r"^.*\bfind\s+the\s+ratios\s+in\s+",

        (
            r"^.*\bsubsequent\s+decision\s+of\s+"
            r"(?:the\s+)?Supreme\s+Court\s+in\s+"
        ),

        (
            r"^.*\brefer\s+the\s+judgment\s+of\s+"
            r"(?:the\s+)?Hon['’]ble\s+"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\breference\s+(?:in\s+this\s+regard\s+)?"
            r"may\s+be\s+made\s+to\s+the\s+decision\s+of\s+"
            r"(?:the\s+)?Hon['’]ble\s+"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\breferring\s+to\s+para\s+\d+\s+of\s+"
            r"judgment\s+of\s+(?:the\s+)?"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*\b(?:principle|proposition)\s+"
            r"(?:has\s+been\s+)?reiterated\s+by\s+"
            r"(?:the\s+)?Hon['’]ble\s+"
            r"(?:Apex|Supreme)\s+Court\s+"
            r"in\s+the\s+judgment\s+of\s+"
        ),

        # Parenthetical prose immediately before a real case name.
        (
            r"^.*?\(\s*"
            r"(?="
            r"(?-i:[A-Z])"
            r"[A-Za-z0-9 .,'’()&/@+\-:]{1,180}"
            r"\s+(?:v\.?|vs\.?|versus)\s+"
            r"(?-i:[A-Z])"
            r")"
        ),

        # Final guarded fallback:
        # remove prose ending in "in" only when what follows is
        # clearly another case-name expression containing v./vs./versus.
        (
            r"^.*\bin\s+"
            r"(?="
            r"(?-i:[A-Z])"
            r"[A-Za-z0-9 .,'’()&/@+\-:]{1,180}?"
            r"\s+(?:v\.?|vs\.?|versus)\s+"
            r"(?-i:[A-Z])"
            r")"
        ),
    )

    for pattern in patterns:

        cleaned = re.sub(
            pattern,
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()

        if cleaned != value:

            value = cleaned
            break

    # --------------------------------------------------------
    # Safe second-stage cleanup
    # --------------------------------------------------------
    #
    # A first legal introducer can expose another wrapper.
    # Example:
    #
    #     Supreme Court in the judgment of X v. Y
    #
    # becomes:
    #
    #     the judgment of X v. Y
    #
    # This pass removes only tightly bounded wrappers.
    # --------------------------------------------------------

    second_stage_patterns = (

        r"^(?:the\s+)?judgment\s+of\s+",
        r"^(?:the\s+)?decision\s+of\s+",

        (
            r"^(?:Secondly,\s*)?"
            r"(?:the\s+)?judgment\s+"
            r"(?=[A-Z])"
        ),

        (
            r"^.*?\bin\s+"
            r"(?="
            r"(?-i:[A-Z])"
            r"[A-Za-z0-9 .,'’()&/@+\-:]{1,180}?"
            r"\s+(?:v\.?|vs\.?|versus)\s+"
            r"(?-i:[A-Z])"
            r")"
        ),

        (
            r"^(?:the\s+)?authoritative\s+pronouncement\s+of\s+"
            r"(?:the\s+)?(?:Hon['’]ble\s+)?"
            r"(?:Apex|Supreme|High)\s+Court\s+in\s+"
        ),

        (
            r"^.*?\(\s*"
            r"(?="
            r"(?-i:[A-Z])"
            r"[A-Za-z0-9 .,'’()&/@+\-:]{1,180}"
            r"\s+(?:v\.?|vs\.?|versus)\s+"
            r"(?-i:[A-Z])"
            r")"
        ),
    )

    second_stage_changed = True

    while second_stage_changed:

        second_stage_changed = False

        for pattern in second_stage_patterns:

            cleaned = re.sub(
                pattern,
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()

            if cleaned != value:

                value = cleaned
                second_stage_changed = True
                break

    # --------------------------------------------------------
    # Simple wrappers
    # --------------------------------------------------------

    simple_prefixes = (
        "In the case of ",
        "The case of ",
        "In ",
        "The judgment of ",
        "The decision of ",
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

    return normalize_spaces(
        value
    )


def clean_case_suffix(
    value: str,
) -> str:
    """
    Remove prose accidentally captured after the second party.

    v1.4 safety rule:
        aggressive suffix cleanup is applied only to the text
        occurring after the case connector (v./vs./versus).

    This prevents a phrase appearing before the actual case name,
    such as:

        Further, the Hon'ble Apex Court in
        Malkiat Singh v. Joginder Singh

    from being reduced to just "Further".
    """

    value = normalize_spaces(
        value
    )

    connector_match = (
        CASE_CONNECTOR_PATTERN.search(
            value
        )
    )

    # A canonical case-name candidate should contain a connector.
    # If it does not, do not perform aggressive suffix stripping.
    if connector_match is None:
        return value

    left = value[
        :connector_match.start()
    ].strip()

    right = value[
        connector_match.end():
    ].strip()

    suffix_patterns = (

        r"\s+which\s+consistently\s+held.*$",
        r"\s+which\s+held.*$",

        r"\s+wherein\s+.*$",
        r"\s+where\s+the\s+court\s+.*$",

        r"\s+in\s+support\s+of\s+such\s+contention.*$",
        r"\s+in\s+support\s+of\s+his\s+contentions.*$",
        r"\s+in\s+support\s+of\s+her\s+contentions.*$",
        r"\s+in\s+support\s+of\s+their\s+contentions.*$",

        (
            r"\s+the\s+Hon['’]ble\s+"
            r"Supreme\s+Court\s+held.*$"
        ),

        (
            r"\s+the\s+Hon['’]ble\s+"
            r"Supreme\s+Court\s+observed.*$"
        ),

        # Bench / court explanatory prose appearing after
        # the second party.
        r"\s+a\s+learned\s+Division\s+Bench.*$",
        r"\s+a\s+learned\s+Single\s+Judge.*$",
        r"\s+a\s+three[-\s]?Judge\s+Bench.*$",
        r"\s+a\s+Constitution\s+Bench.*$",

        (
            r"\s+the\s+Hon['’]ble\s+"
            r"Apex\s+Court\s+.*$"
        ),

        (
            r"\s+the\s+Hon['’]ble\s+"
            r"High\s+Court\s+.*$"
        ),

        # Explanatory tails.
        r"\s+in\s+view\s+thereof.*$",
        r"\s+to\s+hold\s+that.*$",
        r"\s+is\s+that\s+.*$",
        r"\s+are\s+instructive.*$",
        r"\s+that\s+no\s+final\s+order.*$",

        # Common prose after the actual authority.
        r"\s+set\s+aside\s+the\s+order.*$",
        r"\s+directed\s+the\s+release.*$",
        r"\s+dealt\s+with\s+.*$",
        r"\s+held\s+thus.*$",

        # Narrow v1.4.1 fix:
        # Basavaraj v. Canara Bank) to indicate the law...
        r"\)\s+to\s+indicate\b.*$",

        # External citation identifiers accidentally captured
        # after the second party.
        r"\s+MANU/[A-Z]+/\d+/\d+.*$",
        r"\s+AIR\s+\d{4}\s+[A-Z&]{2,12}\s+\d+.*$",

        # List / parenthetical continuation.
        r"\)\s+and\s+three\s+Judges?\s+Bench.*$",
    )

    for pattern in suffix_patterns:

        right = re.sub(
            pattern,
            "",
            right,
            flags=re.IGNORECASE,
        ).strip()

    # Remove a closing parenthesis left over from:
    #   (Case A v. Case B)
    right = re.sub(
        r"\)\s*$",
        "",
        right,
    ).strip()

    return normalize_spaces(
        f"{left} v. {right}"
    )


def normalize_case_name(
    value: str,
) -> str:

    value = normalize_spaces(
        value
    )

    # --------------------------------------------------------
    # v1.4.1 surgical pre-clean
    # --------------------------------------------------------
    #
    # Some citations already begin with the true case name but
    # continue with post-citation bench prose:
    #
    #   Shabana v. NCT of Delhi a learned Division Bench ...
    #
    # v1.4's prefix cleaner can mistake the later "in ..." phrase
    # for a prefix introducer. Trim only this very narrow tail,
    # and only when it occurs AFTER the case connector.
    # --------------------------------------------------------

    connector_match = (
        CASE_CONNECTOR_PATTERN.search(
            value
        )
    )

    if connector_match is not None:

        left_of_connector = value[
            :connector_match.start()
        ]

        right_of_connector = value[
            connector_match.end():
        ]

        right_of_connector = re.sub(
            r"\s+a\s+learned\s+Division\s+Bench\b.*$",
            "",
            right_of_connector,
            flags=re.IGNORECASE,
        ).strip()

        value = normalize_spaces(
            f"{left_of_connector} v. {right_of_connector}"
        )

    # --------------------------------------------------------
    # Remove preceding prose
    # --------------------------------------------------------

    value = clean_case_prefix(
        value
    )

    # --------------------------------------------------------
    # Remove trailing prose
    # --------------------------------------------------------

    value = clean_case_suffix(
        value
    )

    # --------------------------------------------------------
    # Remove footnote markers
    # --------------------------------------------------------

    value = remove_footnote_noise(
        value
    )

    # --------------------------------------------------------
    # Normalize versus connector
    # --------------------------------------------------------

    value = CASE_CONNECTOR_PATTERN.sub(
        " v. ",
        value,
        count=1,
    )

    value = normalize_spaces(
        value
    )

    value = value.strip(
        " ,.;:-[]()"
    )

    return value


def case_name_key(
    value: str,
) -> str:

    value = normalize_case_name(
        value
    ).lower()

    # --------------------------------------------------------
    # Normalize ampersands
    # --------------------------------------------------------

    value = value.replace(
        "&",
        " and ",
    )

    # --------------------------------------------------------
    # Remove punctuation for alias comparison
    # --------------------------------------------------------

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = normalize_spaces(
        value
    )

    # --------------------------------------------------------
    # Normalize connector variants if any survive
    # --------------------------------------------------------

    value = re.sub(
        r"\bvs\b",
        "v",
        value,
    )

    value = re.sub(
        r"\bversus\b",
        "v",
        value,
    )

    return value


def reporter_key(
    value: str,
) -> str:

    return (
        normalize_reporter(
            value
        )
        .lower()
        .strip()
    )


# ============================================================
# Alias IDs
# ============================================================

def mention_alias_key(
    citation_text: str,
) -> str:

    # --------------------------------------------------------
    # Reporter alias
    # --------------------------------------------------------

    if is_reporter_citation(
        citation_text
    ):

        return (
            "REPORTER::"
            + reporter_key(
                citation_text
            )
        )

    # --------------------------------------------------------
    # Case-name alias
    # --------------------------------------------------------

    if is_case_name(
        citation_text
    ):

        return (
            "CASE_NAME::"
            + case_name_key(
                citation_text
            )
        )

    # --------------------------------------------------------
    # Unknown/raw alias
    # --------------------------------------------------------

    return (
        "RAW::"
        + normalize_spaces(
            citation_text
        ).lower()
    )


# ============================================================
# Union-find
# ============================================================

class UnionFind:

    def __init__(
        self,
    ) -> None:

        self.parent: dict[
            str,
            str,
        ] = {}

        self.rank: dict[
            str,
            int,
        ] = {}

    def add(
        self,
        item: str,
    ) -> None:

        if item in self.parent:
            return

        self.parent[
            item
        ] = item

        self.rank[
            item
        ] = 0

    def find(
        self,
        item: str,
    ) -> str:

        if item not in self.parent:

            self.add(
                item
            )

        if (
            self.parent[
                item
            ]
            != item
        ):

            self.parent[
                item
            ] = self.find(
                self.parent[
                    item
                ]
            )

        return self.parent[
            item
        ]

    def union(
        self,
        first: str,
        second: str,
    ) -> None:

        first_root = self.find(
            first
        )

        second_root = self.find(
            second
        )

        if (
            first_root
            == second_root
        ):
            return

        first_rank = self.rank[
            first_root
        ]

        second_rank = self.rank[
            second_root
        ]

        if (
            first_rank
            < second_rank
        ):

            self.parent[
                first_root
            ] = second_root

        elif (
            first_rank
            > second_rank
        ):

            self.parent[
                second_root
            ] = first_root

        else:

            self.parent[
                second_root
            ] = first_root

            self.rank[
                first_root
            ] += 1


# ============================================================
# Citation event grouping
# ============================================================

def citation_event_key(
    row: dict[str, Any],
) -> tuple[
    str,
    str,
]:
    """
    Group citation aliases appearing in the same citation sentence.

    Conservative strategy:

        one case name + reporter(s)
            -> safe to merge

        multiple case names + reporter(s)
            -> ambiguous, do not guess
    """

    document_id = str(
        row.get(
            "document_id",
            "",
        )
    )

    citation_sentence = normalize_spaces(
        str(
            row.get(
                "citation_sentence",
                "",
            )
        )
    ).lower()

    return (
        document_id,
        citation_sentence,
    )


def group_citation_events(
    rows: list[dict[str, Any]],
) -> dict[
    tuple[str, str],
    list[dict[str, Any]],
]:

    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    for row in rows:

        key = citation_event_key(
            row
        )

        groups[
            key
        ].append(
            row
        )

    return groups


# ============================================================
# Optional local corpus index
# ============================================================

TITLE_FIELDS = (
    "case_name",
    "case_title",
    "title",
    "judgment_title",
    "name",
)


def get_document_id(
    data: dict[str, Any],
    fallback: str,
) -> str:

    candidates = (
        data.get(
            "document_id"
        ),
        data.get(
            "doc_id"
        ),
        data.get(
            "id"
        ),
        fallback,
    )

    for candidate in candidates:

        if candidate:

            return str(
                candidate
            )

    return fallback


def extract_titles_from_mapping(
    data: dict[str, Any],
) -> list[str]:

    titles: list[
        str
    ] = []

    # --------------------------------------------------------
    # Top-level metadata
    # --------------------------------------------------------

    for field in TITLE_FIELDS:

        value = data.get(
            field
        )

        if (
            isinstance(
                value,
                str,
            )
            and is_case_name(
                value
            )
        ):

            titles.append(
                normalize_case_name(
                    value
                )
            )

    # --------------------------------------------------------
    # Nested metadata
    # --------------------------------------------------------

    metadata = data.get(
        "metadata"
    )

    if isinstance(
        metadata,
        dict,
    ):

        for field in TITLE_FIELDS:

            value = metadata.get(
                field
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and is_case_name(
                    value
                )
            ):

                titles.append(
                    normalize_case_name(
                        value
                    )
                )

    return titles


def load_local_case_index() -> dict[
    str,
    set[str],
]:
    """
    Build an optional exact-title lookup against the local canonical
    judgment corpus.

    This does not block citation resolution when document metadata
    lacks a case title.
    """

    index: dict[
        str,
        set[str],
    ] = defaultdict(
        set
    )

    # --------------------------------------------------------
    # Processed document metadata
    # --------------------------------------------------------

    if DOCUMENTS_DIR.exists():

        for path in DOCUMENTS_DIR.glob(
            "*.json"
        ):

            try:

                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception:
                continue

            if not isinstance(
                data,
                dict,
            ):
                continue

            document_id = (
                get_document_id(
                    data,
                    path.stem,
                )
            )

            titles = (
                extract_titles_from_mapping(
                    data
                )
            )

            for title in titles:

                key = case_name_key(
                    title
                )

                if not key:
                    continue

                index[
                    key
                ].add(
                    document_id
                )

    # --------------------------------------------------------
    # Raw canonical manifest
    # --------------------------------------------------------

    if RAW_MANIFEST_PATH.exists():

        try:

            with RAW_MANIFEST_PATH.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:

                reader = csv.DictReader(
                    file
                )

                for row in reader:

                    document_id = (
                        row.get(
                            "document_id"
                        )
                        or row.get(
                            "doc_id"
                        )
                        or row.get(
                            "id"
                        )
                        or ""
                    )

                    if not document_id:
                        continue

                    for field in TITLE_FIELDS:

                        value = row.get(
                            field
                        )

                        if (
                            value
                            and is_case_name(
                                value
                            )
                        ):

                            key = case_name_key(
                                value
                            )

                            if not key:
                                continue

                            index[
                                key
                            ].add(
                                document_id
                            )

        except Exception:
            # Optional index:
            # failure here must not kill resolution.
            pass

    return index


# ============================================================
# Canonical ID generation
# ============================================================

def build_case_id(
    seed: str,
) -> str:
    """
    Generate deterministic canonical authority IDs.

    Reporter citations are preferred as cluster seeds when available.
    """

    digest = hashlib.sha1(
        seed.encode(
            "utf-8"
        )
    ).hexdigest()[
        :12
    ].upper()

    return (
        f"CASE_{digest}"
    )


# ============================================================
# Suspicious-name QA
# ============================================================

SUSPICIOUS_CASE_TERMS = (

    "learned counsel",

    "learned senior counsel",

    "submits",

    "submitted",

    "relying",

    "reliance",

    "contention",

    "the decision",

    "the judgment",

    "the case of",

    "the in ",

    "claimants",

    "eyewitness",

    "authoritative judgment",

    "test enunciated",

    "counsel for the petitioner",
    "counsel for the respondent",
    "the court made the following order",
    "marriage register book",
    "advocate for the petitioner",
    "advocate for the respondent",
    "civil appeal no.",
    "civil appeal no ",
    "slp (c)",
    "w.p.(c)",
    "no uncertain terms",
    "secondly, the judgment",
)


def case_name_needs_review(
    value: str,
) -> bool:

    normalized = normalize_spaces(
        value
    )

    lower = normalized.lower()

    # --------------------------------------------------------
    # Excessively long names are usually extraction pollution
    # --------------------------------------------------------

    if len(
        normalized
    ) > 130:

        return True

    # --------------------------------------------------------
    # Residual surrounding prose
    # --------------------------------------------------------

    if any(
        term in lower
        for term
        in SUSPICIOUS_CASE_TERMS
    ):

        return True

    # --------------------------------------------------------
    # Must still contain a case connector
    # --------------------------------------------------------

    if not CASE_CONNECTOR_PATTERN.search(
        normalized
    ):

        return True

    return False


# ============================================================
# Authority building
# ============================================================

def build_alias_clusters(
    rows: list[dict[str, Any]],
) -> tuple[
    UnionFind,
    int,
]:
    """
    Conservatively merge aliases.

    Safe case:

        Gangadhar Behera v. State of Orissa
        (2002) 8 SCC 381

        -> one canonical authority

    Unsafe case:

        multiple case names + multiple reporter citations
        in the same sentence

        -> do not guess reporter/name pairings
    """

    union_find = UnionFind()

    # --------------------------------------------------------
    # Register every observed alias
    # --------------------------------------------------------

    for row in rows:

        citation_text = str(
            row.get(
                "citation_text",
                "",
            )
        )

        alias = mention_alias_key(
            citation_text
        )

        union_find.add(
            alias
        )

    events = group_citation_events(
        rows
    )

    ambiguous_events = 0

    # --------------------------------------------------------
    # Merge safe name + reporter citation events
    # --------------------------------------------------------

    for event_rows in events.values():

        case_aliases: list[
            str
        ] = []

        reporter_aliases: list[
            str
        ] = []

        for row in event_rows:

            citation_text = str(
                row.get(
                    "citation_text",
                    "",
                )
            )

            alias = mention_alias_key(
                citation_text
            )

            if is_case_name(
                citation_text
            ):

                case_aliases.append(
                    alias
                )

            elif is_reporter_citation(
                citation_text
            ):

                reporter_aliases.append(
                    alias
                )

        # ----------------------------------------------------
        # Stable deduplication
        # ----------------------------------------------------

        case_aliases = list(
            dict.fromkeys(
                case_aliases
            )
        )

        reporter_aliases = list(
            dict.fromkeys(
                reporter_aliases
            )
        )

        # ----------------------------------------------------
        # Safe merge
        # ----------------------------------------------------

        if (
            len(
                case_aliases
            )
            == 1
            and reporter_aliases
        ):

            anchor = (
                case_aliases[
                    0
                ]
            )

            for reporter_alias in (
                reporter_aliases
            ):

                union_find.union(
                    anchor,
                    reporter_alias,
                )

        # ----------------------------------------------------
        # Ambiguous list of cases/reporters
        # ----------------------------------------------------

        elif (
            len(
                case_aliases
            )
            > 1
            and reporter_aliases
        ):

            ambiguous_events += 1

    return (
        union_find,
        ambiguous_events,
    )


def build_clusters(
    union_find: UnionFind,
) -> dict[
    str,
    list[str],
]:

    clusters: dict[
        str,
        list[str],
    ] = defaultdict(
        list
    )

    for alias in (
        union_find.parent
    ):

        root = union_find.find(
            alias
        )

        clusters[
            root
        ].append(
            alias
        )

    return clusters


# ============================================================
# Registry construction helpers
# ============================================================

def alias_display_value(
    alias: str,
) -> str:

    if alias.startswith(
        "REPORTER::"
    ):

        return alias[
            len(
                "REPORTER::"
            ):
        ]

    if alias.startswith(
        "CASE_NAME::"
    ):

        return alias[
            len(
                "CASE_NAME::"
            ):
        ]

    if alias.startswith(
        "RAW::"
    ):

        return alias[
            len(
                "RAW::"
            ):
        ]

    return alias


def choose_cluster_seed(
    aliases: list[str],
) -> str:
    """
    Prefer a normalized reporter citation as the deterministic case-ID
    seed because reporter citations are normally more stable than
    extracted case-name text.
    """

    reporters = sorted(
        alias
        for alias in aliases
        if alias.startswith(
            "REPORTER::"
        )
    )

    if reporters:

        return reporters[
            0
        ]

    case_names = sorted(
        alias
        for alias in aliases
        if alias.startswith(
            "CASE_NAME::"
        )
    )

    if case_names:

        return case_names[
            0
        ]

    return sorted(
        aliases
    )[0]


# ============================================================
# Registry construction
# ============================================================

def build_registry(
    rows: list[dict[str, Any]],
    union_find: UnionFind,
    local_case_index: dict[
        str,
        set[str],
    ],
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
]:
    """
    Build canonical legal authority entities.

    Returns:

        registry rows

        alias -> canonical CASE_* ID
    """

    clusters = build_clusters(
        union_find
    )

    alias_to_case_id: dict[
        str,
        str,
    ] = {}

    citation_text_by_alias: dict[
        str,
        set[str],
    ] = defaultdict(
        set
    )

    mention_count_by_alias: Counter = (
        Counter()
    )

    # --------------------------------------------------------
    # Collect original surface forms
    # --------------------------------------------------------

    for row in rows:

        citation_text = str(
            row.get(
                "citation_text",
                "",
            )
        )

        alias = mention_alias_key(
            citation_text
        )

        citation_text_by_alias[
            alias
        ].add(
            citation_text
        )

        mention_count_by_alias[
            alias
        ] += 1

    registry: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Build canonical entity per alias cluster
    # --------------------------------------------------------

    for aliases in (
        clusters.values()
    ):

        seed = choose_cluster_seed(
            aliases
        )

        case_id = build_case_id(
            seed
        )

        for alias in aliases:

            alias_to_case_id[
                alias
            ] = case_id

        case_names: set[
            str
        ] = set()

        reporters: set[
            str
        ] = set()

        raw_aliases: set[
            str
        ] = set()

        all_original_mentions: set[
            str
        ] = set()

        mention_count = 0

        # Source-level QA is tracked separately from final
        # authority-level QA. A dirty historical surface form
        # should not automatically poison an otherwise clean
        # canonical authority.
        dirty_case_names: set[str] = set()

        needs_review = False

        # ----------------------------------------------------
        # Collect aliases
        # ----------------------------------------------------

        for alias in aliases:

            mention_count += (
                mention_count_by_alias[
                    alias
                ]
            )

            all_original_mentions.update(
                citation_text_by_alias[
                    alias
                ]
            )

            # ------------------------------------------------
            # Case-name alias
            # ------------------------------------------------

            if alias.startswith(
                "CASE_NAME::"
            ):

                for original in (
                    citation_text_by_alias[
                        alias
                    ]
                ):

                    cleaned = normalize_case_name(
                        original
                    )

                    if not cleaned:
                        continue

                    case_names.add(
                        cleaned
                    )

                    if case_name_needs_review(
                        cleaned
                    ):

                        dirty_case_names.add(
                            cleaned
                        )

            # ------------------------------------------------
            # Reporter alias
            # ------------------------------------------------

            elif alias.startswith(
                "REPORTER::"
            ):

                for original in (
                    citation_text_by_alias[
                        alias
                    ]
                ):

                    reporters.add(
                        normalize_reporter(
                            original
                        )
                    )

            # ------------------------------------------------
            # Raw/unclassified alias
            # ------------------------------------------------

            else:

                raw_aliases.update(
                    citation_text_by_alias[
                        alias
                    ]
                )

                needs_review = True

        # ----------------------------------------------------
        # Canonical display name
        # ----------------------------------------------------

        canonical_name = None

        if case_names:

            # Prefer a clean normalized name over a shorter dirty
            # extraction. This prevents values such as "Further"
            # or residual legal prose from becoming the canonical
            # display name merely because they are shorter.
            clean_case_names = [
                name
                for name in case_names
                if not case_name_needs_review(
                    name
                )
            ]

            candidate_names = (
                clean_case_names
                if clean_case_names
                else list(
                    case_names
                )
            )

            canonical_name = sorted(
                candidate_names,
                key=lambda value: (
                    len(
                        value
                    ),
                    value.lower(),
                ),
            )[0]

        # ----------------------------------------------------
        # Authority-level review semantics
        # ----------------------------------------------------

        # The selected canonical name must itself be credible.
        # Dirty alternative source aliases are retained for audit
        # but do not automatically lower a clean authority to
        # review status.
        if (
            canonical_name is not None
            and case_name_needs_review(
                canonical_name
            )
        ):
            needs_review = True

        # No usable case name and no reporter means the authority
        # is intrinsically weak.
        if (
            canonical_name is None
            and not reporters
        ):
            needs_review = True

        # ----------------------------------------------------
        # Attempt exact local-corpus resolution
        # ----------------------------------------------------

        local_document_ids: set[
            str
        ] = set()

        for name in case_names:

            key = case_name_key(
                name
            )

            local_document_ids.update(
                local_case_index.get(
                    key,
                    set(),
                )
            )

        local_document_id = None

        if (
            len(
                local_document_ids
            )
            == 1
        ):

            local_document_id = next(
                iter(
                    local_document_ids
                )
            )

        elif (
            len(
                local_document_ids
            )
            > 1
        ):

            needs_review = True

        # ----------------------------------------------------
        # Resolution quality
        # ----------------------------------------------------

        if (
            reporters
            and case_names
        ):

            resolution_basis = (
                "CASE_NAME_PLUS_REPORTER"
            )

            resolution_confidence = (
                0.98
            )

        elif reporters:

            resolution_basis = (
                "REPORTER_ONLY"
            )

            resolution_confidence = (
                0.94
            )

        elif case_names:

            resolution_basis = (
                "CASE_NAME_ONLY"
            )

            resolution_confidence = (
                0.85
            )

        else:

            resolution_basis = (
                "RAW_ONLY"
            )

            resolution_confidence = (
                0.50
            )

        # ----------------------------------------------------
        # Review penalty
        # ----------------------------------------------------

        if needs_review:

            resolution_confidence = min(
                resolution_confidence,
                0.70,
            )

        # ----------------------------------------------------
        # Registry row
        # ----------------------------------------------------

        registry.append(
            {
                "case_id": (
                    case_id
                ),

                "canonical_name": (
                    canonical_name
                ),

                "case_names": sorted(
                    case_names
                ),

                "reporter_citations": sorted(
                    reporters
                ),

                "raw_aliases": sorted(
                    raw_aliases
                ),

                "original_mentions": sorted(
                    all_original_mentions
                ),

                "local_document_id": (
                    local_document_id
                ),

                "in_local_corpus": (
                    local_document_id
                    is not None
                ),

                "resolution_basis": (
                    resolution_basis
                ),

                "resolution_confidence": round(
                    resolution_confidence,
                    3,
                ),

                "needs_review": (
                    needs_review
                ),

                "dirty_source_aliases": sorted(
                    dirty_case_names
                ),

                "dirty_source_alias_count": (
                    len(
                        dirty_case_names
                    )
                ),

                "mention_count": (
                    mention_count
                ),

                "resolver_version": (
                    RESOLVER_VERSION
                ),

                "resolved_at": (
                    utc_now()
                ),
            }
        )

    registry.sort(
        key=lambda row: (
            str(
                row.get(
                    "canonical_name"
                )
                or ""
            ).lower(),

            str(
                row.get(
                    "case_id",
                    "",
                )
            ),
        )
    )

    return (
        registry,
        alias_to_case_id,
    )


# ============================================================
# Treatment row resolution
# ============================================================

def registry_by_case_id(
    registry: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    dict[str, Any],
]:

    return {
        str(
            row[
                "case_id"
            ]
        ): row
        for row
        in registry
    }


def resolve_treatment_rows(
    rows: list[dict[str, Any]],
    alias_to_case_id: dict[
        str,
        str,
    ],
    registry: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:

    registry_index = (
        registry_by_case_id(
            registry
        )
    )

    output: list[
        dict[str, Any]
    ] = []

    for row in rows:

        resolved = dict(
            row
        )

        citation_text = str(
            row.get(
                "citation_text",
                "",
            )
        )

        alias = mention_alias_key(
            citation_text
        )

        case_id = (
            alias_to_case_id.get(
                alias
            )
        )

        # ----------------------------------------------------
        # Resolved authority
        # ----------------------------------------------------

        if case_id:

            authority = (
                registry_index[
                    case_id
                ]
            )

            resolved[
                "target_case_id"
            ] = case_id

            resolved[
                "target_case_name"
            ] = authority.get(
                "canonical_name"
            )

            resolved[
                "target_reporters"
            ] = authority.get(
                "reporter_citations",
                [],
            )

            resolved[
                "target_document_id"
            ] = authority.get(
                "local_document_id"
            )

            resolved[
                "resolution_basis"
            ] = authority.get(
                "resolution_basis"
            )

            resolved[
                "resolution_confidence"
            ] = authority.get(
                "resolution_confidence"
            )

            resolved[
                "resolution_needs_review"
            ] = authority.get(
                "needs_review",
                False,
            )

        # ----------------------------------------------------
        # Truly unresolved authority
        # ----------------------------------------------------

        else:

            resolved[
                "target_case_id"
            ] = None

            resolved[
                "target_case_name"
            ] = None

            resolved[
                "target_reporters"
            ] = []

            resolved[
                "target_document_id"
            ] = None

            resolved[
                "resolution_basis"
            ] = "UNRESOLVED"

            resolved[
                "resolution_confidence"
            ] = 0.0

            resolved[
                "resolution_needs_review"
            ] = True

        resolved[
            "resolver_version"
        ] = RESOLVER_VERSION

        resolved[
            "resolved_at"
        ] = utc_now()

        output.append(
            resolved
        )

    return output


# ============================================================
# Filtering
# ============================================================

def filter_rows(
    rows: list[dict[str, Any]],
    limit: int | None,
    document_id: str | None,
) -> list[dict[str, Any]]:

    # --------------------------------------------------------
    # Specific document
    # --------------------------------------------------------

    if document_id:

        rows = [
            row
            for row in rows
            if str(
                row.get(
                    "document_id",
                    "",
                )
            )
            == document_id
        ]

        if not rows:

            raise ValueError(
                "No citation treatments found for "
                f"{document_id}"
            )

    # --------------------------------------------------------
    # No document limit
    # --------------------------------------------------------

    if limit is None:

        return rows

    if limit <= 0:

        raise ValueError(
            "--limit must be greater than zero"
        )

    # --------------------------------------------------------
    # First N documents
    # --------------------------------------------------------

    selected_documents: list[
        str
    ] = []

    output: list[
        dict[str, Any]
    ] = []

    for row in rows:

        doc_id = str(
            row.get(
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

            output.append(
                row
            )

    return output


# ============================================================
# Report
# ============================================================

REPORT_FIELDS = [
    "metric",
    "value",
]


def write_report(
    metrics: dict[
        str,
        Any,
    ],
) -> None:

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        for metric, value in (
            metrics.items()
        ):

            writer.writerow(
                {
                    "metric": (
                        metric
                    ),

                    "value": (
                        value
                    ),
                }
            )


# ============================================================
# Pipeline
# ============================================================

def run_resolution(
    limit: int | None = None,
    document_id: str | None = None,
) -> None:

    # --------------------------------------------------------
    # Load citation treatments
    # --------------------------------------------------------

    rows = read_jsonl(
        INPUT_TREATMENTS_PATH
    )

    rows = filter_rows(
        rows=rows,
        limit=limit,
        document_id=document_id,
    )

    documents = sorted(
        {
            str(
                row.get(
                    "document_id",
                    "",
                )
            )
            for row in rows
            if row.get(
                "document_id"
            )
        }
    )

    print()

    print(
        "Canonical Citation Resolution"
    )

    print("=" * 70)

    print(
        "Resolver version: "
        f"{RESOLVER_VERSION}"
    )

    print(
        "Documents selected: "
        f"{len(documents)}"
    )

    print(
        "Citation mentions: "
        f"{len(rows)}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Build alias graph
    # --------------------------------------------------------

    (
        union_find,
        ambiguous_events,
    ) = build_alias_clusters(
        rows
    )

    # --------------------------------------------------------
    # Build local judgment-title index
    # --------------------------------------------------------

    local_case_index = (
        load_local_case_index()
    )

    # --------------------------------------------------------
    # Build canonical registry
    # --------------------------------------------------------

    (
        registry,
        alias_to_case_id,
    ) = build_registry(
        rows=rows,
        union_find=(
            union_find
        ),
        local_case_index=(
            local_case_index
        ),
    )

    # --------------------------------------------------------
    # Attach canonical IDs to treatment rows
    # --------------------------------------------------------

    resolved_rows = (
        resolve_treatment_rows(
            rows=rows,
            alias_to_case_id=(
                alias_to_case_id
            ),
            registry=registry,
        )
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    write_jsonl(
        RESOLVED_TREATMENTS_PATH,
        resolved_rows,
    )

    write_jsonl(
        AUTHORITY_REGISTRY_PATH,
        registry,
    )

    # ========================================================
    # Metrics
    # ========================================================

    resolved_count = sum(
        1
        for row in resolved_rows
        if row.get(
            "target_case_id"
        )
    )

    unresolved_count = (
        len(
            resolved_rows
        )
        - resolved_count
    )

    name_plus_reporter = sum(
        1
        for row in registry
        if row.get(
            "resolution_basis"
        )
        == "CASE_NAME_PLUS_REPORTER"
    )

    reporter_only = sum(
        1
        for row in registry
        if row.get(
            "resolution_basis"
        )
        == "REPORTER_ONLY"
    )

    case_name_only = sum(
        1
        for row in registry
        if row.get(
            "resolution_basis"
        )
        == "CASE_NAME_ONLY"
    )

    raw_only = sum(
        1
        for row in registry
        if row.get(
            "resolution_basis"
        )
        == "RAW_ONLY"
    )

    review_count = sum(
        1
        for row in registry
        if row.get(
            "needs_review"
        )
    )

    local_match_count = sum(
        1
        for row in registry
        if row.get(
            "in_local_corpus"
        )
    )

    alias_count = len(
        union_find.parent
    )

    merged_alias_count = (
        alias_count
        - len(
            registry
        )
    )

    metrics = {

        "resolver_version": (
            RESOLVER_VERSION
        ),

        "documents_processed": (
            len(
                documents
            )
        ),

        "citation_mentions": (
            len(
                rows
            )
        ),

        "resolved_mentions": (
            resolved_count
        ),

        "unresolved_mentions": (
            unresolved_count
        ),

        "unique_canonical_authorities": (
            len(
                registry
            )
        ),

        "unique_aliases": (
            alias_count
        ),

        "aliases_merged": (
            merged_alias_count
        ),

        "case_name_plus_reporter": (
            name_plus_reporter
        ),

        "case_name_only": (
            case_name_only
        ),

        "reporter_only": (
            reporter_only
        ),

        "raw_only": (
            raw_only
        ),

        "ambiguous_multi_case_events": (
            ambiguous_events
        ),

        "authorities_needing_review": (
            review_count
        ),

        "local_corpus_matches": (
            local_match_count
        ),
    }

    write_report(
        metrics
    )

    # ========================================================
    # Console report
    # ========================================================

    print()

    print(
        "Citation Resolution Report"
    )

    print("=" * 70)

    print(
        "Documents processed: "
        f"{len(documents)}"
    )

    print(
        "Citation mentions: "
        f"{len(rows)}"
    )

    print(
        "Resolved mentions: "
        f"{resolved_count}"
    )

    print(
        "Unresolved mentions: "
        f"{unresolved_count}"
    )

    print()

    print(
        "Canonical authorities: "
        f"{len(registry)}"
    )

    print(
        "Unique aliases: "
        f"{alias_count}"
    )

    print(
        "Aliases merged: "
        f"{merged_alias_count}"
    )

    print()

    print(
        "Name + reporter authorities: "
        f"{name_plus_reporter}"
    )

    print(
        "Case-name-only authorities: "
        f"{case_name_only}"
    )

    print(
        "Reporter-only authorities: "
        f"{reporter_only}"
    )

    print(
        "Raw-only authorities: "
        f"{raw_only}"
    )

    print()

    print(
        "Ambiguous multi-case events: "
        f"{ambiguous_events}"
    )

    print(
        "Authorities needing review: "
        f"{review_count}"
    )

    print(
        "Matched to local corpus: "
        f"{local_match_count}"
    )

    print()

    print(
        "Resolved treatments: "
        f"{RESOLVED_TREATMENTS_PATH}"
    )

    print(
        "Authority registry: "
        f"{AUTHORITY_REGISTRY_PATH}"
    )

    print(
        "Report: "
        f"{REPORT_PATH}"
    )

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Resolve citation mentions "
            "into canonical legal "
            "authority entities."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process citation mentions "
            "from the first N documents."
        ),
    )

    parser.add_argument(
        "--document-id",
        default=None,
        help=(
            "Resolve citations from "
            "one document only."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    try:

        run_resolution(
            limit=(
                args.limit
            ),

            document_id=(
                args.document_id
            ),
        )

    except Exception as exc:

        print()

        print(
            "Citation resolution failed"
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