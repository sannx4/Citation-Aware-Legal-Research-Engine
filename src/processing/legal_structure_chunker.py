from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCUMENTS_DIR = PROCESSED_DIR / "documents"

CHUNKS_DIR = PROCESSED_DIR / "chunks"

GLOBAL_CHUNKS_PATH = (
    PROCESSED_DIR / "legal_chunks.jsonl"
)

CHUNKING_MANIFEST_PATH = (
    PROCESSED_DIR / "legal_chunking_manifest.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

CHUNKING_REPORT_PATH = (
    REPORTS_DIR
    / "legal_structure_chunking_report.csv"
)

CHUNKER_VERSION = "1.2.0"


# Approximate chunk sizes.
# We intentionally avoid requiring a tokenizer dependency.
TARGET_TOKENS = 850
MAX_TOKENS = 1100
OVERLAP_TOKENS = 120

MIN_PARAGRAPH_CHARS = 20


SECTION_TYPES = (
    "PREAMBLE",
    "FACTS",
    "ISSUES",
    "ARGUMENTS",
    "REASONING",
    "HOLDING",
    "ORDER",
    "GENERAL",
)


# ============================================================
# Data structures
# ============================================================

@dataclass
class TextUnit:
    page: int
    text: str
    section: str
    detection_method: str
    token_estimate: int


# ============================================================
# Time helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# Token estimation
# ============================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate.

    English legal text averages roughly 1.2-1.5 tokens per
    whitespace-delimited word. This avoids introducing a tokenizer
    dependency into the corpus processing layer.
    """

    if not text:
        return 0

    words = re.findall(
        r"\S+",
        text,
    )

    return max(
        1,
        math.ceil(
            len(words) * 1.33
        ),
    )


# ============================================================
# Text normalization
# ============================================================

def normalize_line(
    line: str,
) -> str:

    line = line.replace(
        "\x00",
        "",
    )

    line = re.sub(
        r"[ \t]+",
        " ",
        line,
    )

    return line.strip()


def normalized_upper(
    text: str,
) -> str:

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().upper()


# ============================================================
# Heading detection
# ============================================================

HEADING_PATTERNS: list[
    tuple[str, list[str]]
] = [
    (
        "FACTS",
        [
            r"FACTS",
            r"BRIEF FACTS",
            r"FACTS OF THE CASE",
            r"FACTUAL BACKGROUND",
            r"BACKGROUND FACTS",
            r"FACTUAL MATRIX",
            r"FACTUAL ASPECTS",
            r"PROSECUTION CASE",
            r"CASE OF THE PROSECUTION",
        ],
    ),

    (
        "ISSUES",
        [
            r"ISSUE",
            r"ISSUES",
            r"ISSUES FOR CONSIDERATION",
            r"QUESTION FOR CONSIDERATION",
            r"QUESTIONS FOR CONSIDERATION",
            r"QUESTION OF LAW",
            r"QUESTIONS OF LAW",
            r"POINT FOR DETERMINATION",
            r"POINTS FOR DETERMINATION",
            r"QUESTIONS? THAT ARISE",
        ],
    ),

    (
        "ARGUMENTS",
        [
            r"ARGUMENTS",
            r"SUBMISSIONS",
            r"CONTENTIONS",
            r"SUBMISSIONS OF THE APPELLANT",
            r"SUBMISSIONS OF THE RESPONDENT",
            r"SUBMISSIONS ON BEHALF OF .*",
            r"ARGUMENTS ON BEHALF OF .*",
            r"CONTENTIONS OF THE APPELLANT",
            r"CONTENTIONS OF THE RESPONDENT",
        ],
    ),

    (
        "REASONING",
        [
            r"ANALYSIS",
            r"ANALYSIS OF THIS COURT",
            r"ANALYSIS BY THIS COURT",
            r"OUR ANALYSIS",
            r"COURT'?S ANALYSIS",
            r"DISCUSSION",
            r"REASONS",
            r"REASONING",
            r"ANALYSIS AND FINDINGS",
            r"DISCUSSION AND FINDINGS",
            r"CONSIDERATION",
            r"FINDINGS",
            r"FINDINGS OF THIS COURT",
            r"OUR FINDINGS",
            r"CONSIDERATION BY THIS COURT",
        ],
    ),

    (
        "HOLDING",
        [
            r"CONCLUSION",
            r"CONCLUSIONS",
            r"HELD",
            r"DECISION",
            r"OUR CONCLUSION",
            r"FINAL CONCLUSION",
            r"CONCLUSION OF THIS COURT",
        ],
    ),

    (
        "ORDER",
        [
            r"FINAL ORDER",
            r"OPERATIVE ORDER",
            r"RELIEF",
            r"DIRECTIONS",
            r"DISPOSITION",
        ],
    ),
]

def detect_explicit_heading(
    line: str,
    position_ratio: float,
) -> str | None:
    """
    Detect short explicit section headings.

    Plain ORDER/JUDGMENT near the beginning of many Indian judgments
    describes the document type, not necessarily the final disposition.
    """

    cleaned = normalized_upper(
        line
    )

    cleaned = cleaned.strip(
        ":-–—.()[]{} "
    )

    if not cleaned:
        return None

    # A genuine heading is normally short.
    if len(cleaned) > 100:
        return None

    for section, patterns in HEADING_PATTERNS:
        for pattern in patterns:
            if re.fullmatch(
                pattern,
                cleaned,
                flags=re.IGNORECASE,
            ):
                return section

    # Plain ORDER is only treated as final disposition when it appears
    # sufficiently late in the judgment.
    if cleaned == "ORDER":
        if position_ratio >= 0.65:
            return "ORDER"

        return "GENERAL"

    # JUDGMENT appearing near the top is document metadata.
    if cleaned in {
        "JUDGMENT",
        "JUDGEMENT",
    }:
        return "GENERAL"

    return None


# ============================================================
# Heuristic section signals
# ============================================================

SECTION_SIGNALS: dict[
    str,
    list[
        tuple[str, float]
    ],
] = {
    "FACTS": [
        (
            r"\bbrief facts\b",
            4.0,
        ),
        (
            r"\bfacts of the case\b",
            4.0,
        ),
        (
            r"\bfactual matrix\b",
            4.0,
        ),
        (
            r"\bfactual background\b",
            4.0,
        ),
        (
            r"\bcase of the petitioner\b",
            2.0,
        ),
        (
            r"\bcase of the prosecution\b",
            2.0,
        ),
    ],

    "ISSUES": [
        (
            r"\bquestion for consideration\b",
            4.0,
        ),
        (
            r"\bquestions? that arise\b",
            4.0,
        ),
        (
            r"\bissue for consideration\b",
            4.0,
        ),
        (
            r"\bpoint for determination\b",
            4.0,
        ),
        (
            r"\bquestion of law\b",
            3.0,
        ),
        (
            r"\bissue before (?:us|this court)\b",
            3.0,
        ),
    ],

    "ARGUMENTS": [
        (
            r"\blearned counsel .* submitted\b",
            4.0,
        ),
        (
            r"\blearned counsel .* contended\b",
            4.0,
        ),
        (
            r"\bit was argued\b",
            3.0,
        ),
        (
            r"\bit was submitted\b",
            3.0,
        ),
        (
            r"\bcounsel for .* submitted\b",
            3.5,
        ),
        (
            r"\bpetitioner contends\b",
            3.0,
        ),
        (
            r"\brespondent contends\b",
            3.0,
        ),
    ],

    "REASONING": [
        (
            r"\bwe have considered\b",
            4.0,
        ),
        (
            r"\bhaving heard\b",
            3.5,
        ),
        (
            r"\bin our considered (?:view|opinion)\b",
            4.0,
        ),
        (
            r"\bit is well settled\b",
            3.0,
        ),
        (
            r"\bwe find that\b",
            3.5,
        ),
        (
            r"\bthis court finds\b",
            3.5,
        ),
        (
            r"\bon examination of\b",
            2.5,
        ),
        (
            r"\bwe are unable to accept\b",
            3.5,
        ),
        (
    r"\bthe (?:hon'?ble )?(?:supreme|apex|high)?\s*"
    r"court (?:has )?(?:held|observed|emphasized|"
    r"reaffirmed|clarified|underscored)\b",
    4.0,
),

(
    r"\bin the case at hand\b",
    3.5,
),

(
    r"\bthis court (?:finds|holds|observes|"
    r"is of the view|is of the considered view)\b",
    4.0,
),

(
    r"\bwe (?:find|observe|note|consider|hold)\b",
    3.5,
),

(
    r"\bthe court underscored\b",
    4.0,
),

(
    r"\bthe court emphasized\b",
    4.0,
),

(
    r"\bthe court reaffirmed\b",
    4.0,
),
    ],

"HOLDING": [
    (
        r"\bwe hold that\b",
        5.0,
    ),
    (
        r"\bwe therefore hold\b",
        5.0,
    ),
    (
        r"\bwe conclude that\b",
        5.0,
    ),
    (
        r"\bwe are of the opinion that\b",
        4.0,
    ),
    (
        r"\bwe are of the considered view that\b",
        4.5,
    ),
    (
        r"\bthis court is of the considered view\b",
        4.5,
    ),
    (
        r"\bit is held that\b",
        4.5,
    ),
    (
        r"\bwe find no merit\b",
        4.0,
    ),
    (
        r"\bwe find merit\b",
        4.0,
    ),
],

"ORDER": [
    (
        r"\bpetition is dismissed\b",
        5.0,
    ),
    (
        r"\bpetition is allowed\b",
        5.0,
    ),
    (
        r"\bapplication is dismissed\b",
        5.0,
    ),
    (
        r"\bapplication is allowed\b",
        5.0,
    ),
    (
        r"\bappeal is dismissed\b",
        5.0,
    ),
    (
        r"\bappeal is allowed\b",
        5.0,
    ),
    (
        r"\bappeal succeeds\b",
        5.0,
    ),
    (
        r"\bwrit petition is dismissed\b",
        5.0,
    ),
    (
        r"\bwrit petition is allowed\b",
        5.0,
    ),
    (
        r"\bbail application is dismissed\b",
        5.0,
    ),
    (
        r"\bbail application is allowed\b",
        5.0,
    ),
    (
        r"\bstands dismissed\b",
        4.5,
    ),
    (
        r"\bstands allowed\b",
        4.5,
    ),
    (
        r"\bstands disposed of\b",
        4.5,
    ),
    (
        r"\baccordingly.*disposed of\b",
        4.5,
    ),
    (
        r"\bconviction .* set aside\b",
        5.0,
    ),
    (
        r"\bsentence .* set aside\b",
        5.0,
    ),
    (
        r"\bjudgment .* set aside\b",
        5.0,
    ),
    (
        r"\bappellant[s]? .* acquitted\b",
        5.0,
    ),
    (
        r"\baccused .* acquitted\b",
        5.0,
    ),
    (
        r"\breleased on bail\b",
        4.5,
    ),
    (
        r"\bno order as to costs\b",
        3.5,
    ),
],
}


def heuristic_section(
    text: str,
    position_ratio: float,
) -> tuple[
    str | None,
    float,
]:
    """
    Return the strongest legal-section signal.

    Uses textual evidence first, then small positional adjustments.
    Weak evidence returns None rather than forcing an incorrect
    legal-section label.
    """

    lowered = text.lower()

    scores: dict[
        str,
        float,
    ] = {
        section: 0.0
        for section
        in SECTION_SIGNALS
    }

    # ---------------------------------------------------------
    # Textual signals
    # ---------------------------------------------------------

    for section, signals in SECTION_SIGNALS.items():

        for pattern, weight in signals:

            if re.search(
                pattern,
                lowered,
                flags=re.IGNORECASE,
            ):
                scores[
                    section
                ] += weight

    # ---------------------------------------------------------
    # Positional validity constraints
    # ---------------------------------------------------------

    # FACTS signals late in a judgment are often phrases such as
    # "in the facts of the case" occurring inside judicial reasoning.
    if position_ratio > 0.60:
        scores[
            "FACTS"
        ] = 0.0

    # Issues are normally framed before the final stages.
    if position_ratio > 0.75:
        scores[
            "ISSUES"
        ] *= 0.25

    # Party arguments normally appear before the final disposition.
    if position_ratio > 0.85:
        scores[
            "ARGUMENTS"
        ] *= 0.25

    # ---------------------------------------------------------
    # Conservative positional biases
    # ---------------------------------------------------------

    # Early part of a judgment is somewhat more likely to contain
    # factual background, but position alone is not enough to classify.
    if position_ratio < 0.35:
        scores[
            "FACTS"
        ] += 0.35

    # Arguments frequently occur after factual background.
    if (
        0.15
        <= position_ratio
        <= 0.65
    ):
        scores[
            "ARGUMENTS"
        ] += 0.20

    # Judicial reasoning usually occupies the middle/later body.
    if (
        0.25
        <= position_ratio
        <= 0.90
    ):
        scores[
            "REASONING"
        ] += 0.25

    # Holdings are more likely in the later stages.
    if position_ratio >= 0.60:
        scores[
            "HOLDING"
        ] += 0.25

    # Operative orders are especially likely near the end.
    if position_ratio >= 0.75:
        scores[
            "ORDER"
        ] += 0.35

    # ---------------------------------------------------------
    # Select strongest candidate
    # ---------------------------------------------------------

    section = max(
        scores,
        key=scores.get,
    )

    score = scores[
        section
    ]

    # ---------------------------------------------------------
    # Confidence thresholds
    # ---------------------------------------------------------

    # Stronger protection against false FACTS labels because generic
    # phrases containing "facts" are common throughout judgments.
    if (
        section == "FACTS"
        and score < 3.0
    ):
        return None, score

    # HOLDING and ORDER should require meaningful legal language,
    # not merely their position near the end of the judgment.
    if (
        section in {
            "HOLDING",
            "ORDER",
        }
        and score < 3.5
    ):
        return None, score

    # General minimum confidence.
    if score < 2.5:
        return None, score

    return section, score


# ============================================================
# Document loading
# ============================================================

def load_documents() -> list[
    dict[str, Any]
]:
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            "Processed document directory "
            f"does not exist: {DOCUMENTS_DIR}"
        )

    paths = sorted(
        DOCUMENTS_DIR.glob(
            "*.json"
        )
    )

    documents: list[
        dict[str, Any]
    ] = []

    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            document = json.load(
                file
            )

        document[
            "_document_path"
        ] = str(path)

        documents.append(
            document
        )

    return documents


# ============================================================
# Page extraction
# ============================================================

def get_page_data(
    document: dict[str, Any],
) -> list[dict[str, Any]]:

    page_data = document.get(
        "page_data"
    )

    if isinstance(
        page_data,
        list,
    ) and page_data:
        return page_data

    # Fallback for an older structured-document format.
    text = str(
        document.get(
            "text",
            "",
        )
    )

    return [
        {
            "page_number": 1,
            "text": text,
        }
    ]


# ============================================================
# Paragraph creation
# ============================================================

def lines_to_paragraphs(
    text: str,
) -> list[str]:
    """
    Convert noisy PDF-extracted lines into paragraph-like units.

    Legal PDFs frequently break a sentence across multiple visual lines,
    so splitting strictly on newline would create unusably tiny chunks.
    """

    lines = [
        normalize_line(line)
        for line
        in text.splitlines()
    ]

    paragraphs: list[str] = []

    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return

        paragraph = " ".join(
            buffer
        ).strip()

        buffer.clear()

        if (
            len(paragraph)
            >= MIN_PARAGRAPH_CHARS
        ):
            paragraphs.append(
                paragraph
            )

    for line in lines:

        if not line:
            flush()
            continue

        # Preserve potential headings as independent units.
        if (
            len(line) <= 100
            and line.upper() == line
            and len(
                line.split()
            ) <= 12
        ):
            flush()

            paragraphs.append(
                line
            )

            continue

        buffer.append(
            line
        )

        # Natural paragraph boundary.
        if re.search(
            r"[.!?;:]$",
            line,
        ):
            flush()

        # Prevent enormous malformed blocks.
        elif sum(
            len(part)
            for part in buffer
        ) >= 900:
            flush()

    flush()

    return paragraphs


# ============================================================
# Section annotation
# ============================================================

def annotate_document(
    document: dict[str, Any],
) -> tuple[
    list[TextUnit],
    int,
]:
    page_data = get_page_data(
        document
    )

    total_pages = max(
        1,
        len(page_data),
    )

    units: list[
        TextUnit
    ] = []

    explicit_heading_count = 0

    current_explicit_section: (
        str | None
    ) = None

    heuristic_section_state: (
        str | None
    ) = None

    heuristic_decay = 0

    for page_index, page in enumerate(
        page_data,
        start=1,
    ):

        page_number = int(
            page.get(
                "page_number",
                page_index,
            )
        )

        text = str(
            page.get(
                "text",
                "",
            )
        )

        paragraphs = lines_to_paragraphs(
            text
        )

        for paragraph in paragraphs:

            position_ratio = (
                page_index
                / total_pages
            )

            heading = (
                detect_explicit_heading(
                    paragraph,
                    position_ratio,
                )
            )

            if heading is not None:

                explicit_heading_count += 1

                # GENERAL headings such as JUDGMENT or an early ORDER
                # should not force all later content into GENERAL.
                if heading != "GENERAL":
                    current_explicit_section = (
                        heading
                    )

                    heuristic_section_state = (
                        None
                    )

                    heuristic_decay = 0

                continue

            inferred, score = (
                heuristic_section(
                    paragraph,
                    position_ratio,
                )
            )

            if inferred is not None:

                # -----------------------------------------------------
                # Strong terminal transitions
                # -----------------------------------------------------
                # Strong HOLDING / ORDER language near the end of the
                # judgment can override an earlier section such as
                # REASONING or ARGUMENTS.

                if (
                    inferred in {
                        "HOLDING",
                        "ORDER",
                    }
                    and score >= 4.0
                    and position_ratio >= 0.60
                ):
                    section = inferred
                    method = "heuristic_strong"

                    heuristic_section_state = inferred
                    heuristic_decay = 2

                    # ORDER normally represents a terminal section.
                    if inferred == "ORDER":
                        current_explicit_section = "ORDER"

                # -----------------------------------------------------
                # ARGUMENTS -> REASONING transition
                # -----------------------------------------------------
                # Long judgments may have an explicit SUBMISSIONS or
                # ARGUMENTS heading. That heading must not label the
                # entire remaining judgment as ARGUMENTS.
                #
                # Strong judicial-analysis language can transition the
                # document into REASONING.

                elif (
                    current_explicit_section == "ARGUMENTS"
                    and inferred == "REASONING"
                    and score >= 3.5
                    and position_ratio >= 0.30
                ):
                    section = "REASONING"
                    method = "heuristic_transition"

                    current_explicit_section = "REASONING"

                    heuristic_section_state = "REASONING"
                    heuristic_decay = 3

                # -----------------------------------------------------
                # No explicit section currently active
                # -----------------------------------------------------

                elif (
                    current_explicit_section
                    is None
                ):
                    heuristic_section_state = inferred
                    heuristic_decay = 3

                    section = inferred
                    method = "heuristic"

                # -----------------------------------------------------
                # Existing explicit section remains authoritative
                # -----------------------------------------------------

                else:
                    section = (
                        current_explicit_section
                    )

                    method = (
                        "explicit_heading"
                    )

            # ---------------------------------------------------------
            # Continue explicit section
            # ---------------------------------------------------------

            elif (
                current_explicit_section
                is not None
            ):
                section = (
                    current_explicit_section
                )

                method = (
                    "explicit_heading"
                )

            # ---------------------------------------------------------
            # Continue short-lived heuristic context
            # ---------------------------------------------------------

            elif (
                heuristic_section_state
                is not None
                and heuristic_decay > 0
            ):
                section = (
                    heuristic_section_state
                )

                method = (
                    "heuristic_context"
                )

                heuristic_decay -= 1

            # ---------------------------------------------------------
            # Safe fallback
            # ---------------------------------------------------------

            else:

                # Page 1 usually contains:
                #
                # court name
                # case number
                # parties
                # advocates
                # coram / bench
                # dates
                #
                # Treat unclassified page-1 material as PREAMBLE.

                if (
                    page_index == 1
                    and current_explicit_section
                    is None
                ):
                    section = "PREAMBLE"
                    method = "position_fallback"

                else:
                    section = "GENERAL"
                    method = "fallback"

            units.append(
                TextUnit(
                    page=page_number,
                    text=paragraph,
                    section=section,
                    detection_method=method,
                    token_estimate=(
                        estimate_tokens(
                            paragraph
                        )
                    ),
                )
            )

    return (
        units,
        explicit_heading_count,
    )


# ============================================================
# Oversized-unit splitting
# ============================================================

def split_oversized_unit(
    unit: TextUnit,
) -> list[TextUnit]:

    if (
        unit.token_estimate
        <= MAX_TOKENS
    ):
        return [unit]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        unit.text,
    )

    output: list[
        TextUnit
    ] = []

    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        tokens = estimate_tokens(
            sentence
        )

        if (
            buffer
            and (
                buffer_tokens
                + tokens
                > TARGET_TOKENS
            )
        ):
            text = " ".join(
                buffer
            )

            output.append(
                TextUnit(
                    page=unit.page,
                    text=text,
                    section=unit.section,
                    detection_method=(
                        unit.detection_method
                    ),
                    token_estimate=(
                        estimate_tokens(
                            text
                        )
                    ),
                )
            )

            buffer = []
            buffer_tokens = 0

        buffer.append(
            sentence
        )

        buffer_tokens += tokens

    if buffer:
        text = " ".join(
            buffer
        )

        output.append(
            TextUnit(
                page=unit.page,
                text=text,
                section=unit.section,
                detection_method=(
                    unit.detection_method
                ),
                token_estimate=(
                    estimate_tokens(
                        text
                    )
                ),
            )
        )

    return output


# ============================================================
# Chunk building
# ============================================================

def chunk_units(
    units: list[TextUnit],
    document: dict[str, Any],
) -> list[dict[str, Any]]:

    expanded_units: list[
        TextUnit
    ] = []

    for unit in units:
        expanded_units.extend(
            split_oversized_unit(
                unit
            )
        )

    chunks: list[
        dict[str, Any]
    ] = []

    current: list[
        TextUnit
    ] = []

    current_tokens = 0

    chunk_index = 1

    document_id = str(
        document.get(
            "document_id",
            "",
        )
    )

    def emit_chunk() -> None:
        nonlocal current
        nonlocal current_tokens
        nonlocal chunk_index

        if not current:
            return

        section_counts = Counter(
            unit.section
            for unit in current
        )

        section = (
            section_counts.most_common(
                1
            )[0][0]
        )

        method_counts = Counter(
            unit.detection_method
            for unit in current
        )

        detection_method = (
            method_counts.most_common(
                1
            )[0][0]
        )

        text = "\n\n".join(
            unit.text
            for unit in current
        ).strip()

        page_start = min(
            unit.page
            for unit in current
        )

        page_end = max(
            unit.page
            for unit in current
        )

        chunks.append(
            {
                "chunk_id": (
                    f"{document_id}"
                    f"_CH_{chunk_index:04d}"
                ),

                "document_id": (
                    document_id
                ),

                "chunk_index": (
                    chunk_index
                ),

                "section_type": (
                    section
                ),

                "section_detection": (
                    detection_method
                ),

                "page_start": (
                    page_start
                ),

                "page_end": (
                    page_end
                ),

                "token_estimate": (
                    estimate_tokens(
                        text
                    )
                ),

                "characters": len(
                    text
                ),

                "text": text,

                "title": document.get(
                    "title",
                    "",
                ),

                "court": document.get(
                    "court",
                    "",
                ),

                "year": document.get(
                    "year"
                ),

                "source": document.get(
                    "source",
                    "",
                ),

                "source_url": (
                    document.get(
                        "source_url",
                        "",
                    )
                ),

                "document_sha256": (
                    document.get(
                        "sha256",
                        "",
                    )
                ),

                "chunker_version": (
                    CHUNKER_VERSION
                ),
            }
        )

        chunk_index += 1

    index = 0

    while index < len(
        expanded_units
    ):

        unit = expanded_units[
            index
        ]

        # Do not mix legal sections unnecessarily.
        section_changed = (
            current
            and (
                unit.section
                != current[-1].section
            )
        )

        would_exceed = (
            current_tokens
            + unit.token_estimate
            > MAX_TOKENS
        )

        if (
            current
            and (
                section_changed
                or would_exceed
            )
        ):
            previous = list(
                current
            )

            emit_chunk()

            # Section transitions should not carry overlap across the
            # legal boundary.
            if section_changed:
                current = []
                current_tokens = 0

            else:
                overlap: list[
                    TextUnit
                ] = []

                overlap_tokens = 0

                for previous_unit in reversed(
                    previous
                ):
                    if (
                        overlap_tokens
                        >= OVERLAP_TOKENS
                    ):
                        break

                    overlap.insert(
                        0,
                        previous_unit,
                    )

                    overlap_tokens += (
                        previous_unit
                        .token_estimate
                    )

                current = overlap

                current_tokens = sum(
                    item.token_estimate
                    for item in current
                )

        current.append(
            unit
        )

        current_tokens += (
            unit.token_estimate
        )

        # Emit around the target size rather than waiting for the
        # absolute maximum.
        if (
            current_tokens
            >= TARGET_TOKENS
        ):
            previous = list(
                current
            )

            emit_chunk()

            overlap = []
            overlap_tokens = 0

            for previous_unit in reversed(
                previous
            ):
                if (
                    overlap_tokens
                    >= OVERLAP_TOKENS
                ):
                    break

                overlap.insert(
                    0,
                    previous_unit,
                )

                overlap_tokens += (
                    previous_unit
                    .token_estimate
                )

            current = overlap

            current_tokens = sum(
                item.token_estimate
                for item in current
            )

        index += 1

    emit_chunk()

    return chunks


# ============================================================
# File writing
# ============================================================

def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

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


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        return []

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


# ============================================================
# Manifest handling
# ============================================================

MANIFEST_FIELDS = [
    "document_id",
    "source_sha256",
    "chunker_version",
    "chunks",
    "explicit_headings",
    "structured_chunks",
    "general_chunks",
    "status",
    "error",
    "processed_at",
]


def load_previous_manifest() -> dict[
    str,
    dict[str, str],
]:

    if not CHUNKING_MANIFEST_PATH.exists():
        return {}

    with CHUNKING_MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        return {
            row["document_id"]: dict(
                row
            )
            for row in reader
            if row.get(
                "document_id"
            )
        }


def write_manifest(
    rows: dict[
        str,
        dict[str, Any],
    ],
) -> None:

    with CHUNKING_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                MANIFEST_FIELDS
            ),
        )

        writer.writeheader()

        for document_id in sorted(
            rows
        ):
            row = rows[
                document_id
            ]

            writer.writerow(
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key
                    in MANIFEST_FIELDS
                }
            )


# ============================================================
# Report
# ============================================================

REPORT_FIELDS = [
    "document_id",
    "chunks",
    "preamble_chunks",
    "facts_chunks",
    "issues_chunks",
    "arguments_chunks",
    "reasoning_chunks",
    "holding_chunks",
    "order_chunks",
    "general_chunks",
    "structured_chunks",
    "explicit_headings",
    "status",
    "error",
]


def build_report_row(
    document_id: str,
    chunks: list[dict[str, Any]],
    explicit_headings: int,
    status: str = "success",
    error: str = "",
) -> dict[str, Any]:

    counts = Counter(
        chunk.get(
            "section_type",
            "GENERAL",
        )
        for chunk in chunks
    )

    structured_chunks = sum(
        counts.get(
            section,
            0,
        )
        for section in (
            "FACTS",
            "ISSUES",
            "ARGUMENTS",
            "REASONING",
            "HOLDING",
            "ORDER",
        )
    )

    return {
        "document_id": document_id,
        "chunks": len(
            chunks
        ),
        "preamble_chunks": (
            counts.get(
                "PREAMBLE",
                0,
            )
        ),
        "facts_chunks": (
            counts.get(
                "FACTS",
                0,
            )
        ),
        "issues_chunks": (
            counts.get(
                "ISSUES",
                0,
            )
        ),
        "arguments_chunks": (
            counts.get(
                "ARGUMENTS",
                0,
            )
        ),
        "reasoning_chunks": (
            counts.get(
                "REASONING",
                0,
            )
        ),
        "holding_chunks": (
            counts.get(
                "HOLDING",
                0,
            )
        ),
        "order_chunks": (
            counts.get(
                "ORDER",
                0,
            )
        ),
        "general_chunks": (
            counts.get(
                "GENERAL",
                0,
            )
        ),
        "structured_chunks": (
            structured_chunks
        ),
        "explicit_headings": (
            explicit_headings
        ),
        "status": status,
        "error": error,
    }


def write_report(
    rows: list[
        dict[str, Any]
    ],
) -> None:

    with CHUNKING_REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                REPORT_FIELDS
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(
                        key,
                        "",
                    )
                    for key
                    in REPORT_FIELDS
                }
            )


# ============================================================
# Incremental check
# ============================================================

def can_skip(
    document: dict[str, Any],
    previous: dict[
        str,
        dict[str, str],
    ],
    force: bool,
) -> bool:

    if force:
        return False

    document_id = str(
        document.get(
            "document_id",
            "",
        )
    )

    old = previous.get(
        document_id
    )

    if not old:
        return False

    if (
        old.get(
            "status"
        )
        != "success"
    ):
        return False

    if (
        old.get(
            "chunker_version"
        )
        != CHUNKER_VERSION
    ):
        return False

    if (
        old.get(
            "source_sha256"
        )
        != str(
            document.get(
                "sha256",
                "",
            )
        )
    ):
        return False

    chunk_path = (
        CHUNKS_DIR
        / f"{document_id}.jsonl"
    )

    return chunk_path.exists()


# ============================================================
# Global chunk file
# ============================================================

def rebuild_global_chunks(
    document_ids: list[str],
) -> list[dict[str, Any]]:

    all_chunks: list[
        dict[str, Any]
    ] = []

    for document_id in sorted(
        document_ids
    ):

        path = (
            CHUNKS_DIR
            / f"{document_id}.jsonl"
        )

        if not path.exists():
            continue

        all_chunks.extend(
            read_jsonl(
                path
            )
        )

    write_jsonl(
        GLOBAL_CHUNKS_PATH,
        all_chunks,
    )

    return all_chunks


# ============================================================
# Main processing
# ============================================================

def process_documents(
    force: bool = False,
    limit: int | None = None,
    document_id_filter: str | None = None,
) -> None:

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents = load_documents()

    all_document_ids = [
        str(
            document.get(
                "document_id",
                "",
            )
        )
        for document in documents
    ]

    if document_id_filter:

        documents = [
            document
            for document
            in documents
            if str(
                document.get(
                    "document_id",
                    "",
                )
            )
            == document_id_filter
        ]

        if not documents:
            raise ValueError(
                "Document not found: "
                f"{document_id_filter}"
            )

    if limit is not None:
        documents = (
            documents[:limit]
        )

    previous = (
        load_previous_manifest()
    )

    manifest = {
        key: dict(value)
        for key, value
        in previous.items()
    }

    report_rows: list[
        dict[str, Any]
    ] = []

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    print()
    print(
        "Legal Structure-Aware Chunking"
    )
    print("=" * 70)

    print(
        f"Documents selected: "
        f"{len(documents)}"
    )

    print(
        f"Chunker version: "
        f"{CHUNKER_VERSION}"
    )

    print(
        f"Target tokens/chunk: "
        f"{TARGET_TOKENS}"
    )

    print(
        f"Maximum tokens/chunk: "
        f"{MAX_TOKENS}"
    )

    print(
        f"Overlap tokens: "
        f"{OVERLAP_TOKENS}"
    )

    print("=" * 70)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        document_id = str(
            document.get(
                "document_id",
                "",
            )
        )

        print(
            f"[{index}/{len(documents)}] "
            f"{document_id}",
            end="",
        )

        try:

            chunk_path = (
                CHUNKS_DIR
                / f"{document_id}.jsonl"
            )

            if can_skip(
                document=document,
                previous=previous,
                force=force,
            ):

                chunks = read_jsonl(
                    chunk_path
                )

                old = previous[
                    document_id
                ]

                explicit_headings = int(
                    old.get(
                        "explicit_headings",
                        0,
                    )
                    or 0
                )

                skipped_count += 1

                print(
                    " | SKIPPED unchanged"
                )

            else:

                units, explicit_headings = (
                    annotate_document(
                        document
                    )
                )

                chunks = chunk_units(
                    units=units,
                    document=document,
                )

                write_jsonl(
                    chunk_path,
                    chunks,
                )

                processed_count += 1

                section_counts = Counter(
                    chunk[
                        "section_type"
                    ]
                    for chunk in chunks
                )

                print(
                    f" | chunks="
                    f"{len(chunks)}"
                    f" | structured="
                    f"{sum(section_counts.get(s, 0) for s in ['FACTS', 'ISSUES', 'ARGUMENTS', 'REASONING', 'HOLDING', 'ORDER'])}"
                    f" | general="
                    f"{section_counts.get('GENERAL', 0)}"
                )

            report_row = (
                build_report_row(
                    document_id=(
                        document_id
                    ),
                    chunks=chunks,
                    explicit_headings=(
                        explicit_headings
                    ),
                )
            )

            report_rows.append(
                report_row
            )

            manifest[
                document_id
            ] = {
                "document_id": (
                    document_id
                ),

                "source_sha256": (
                    document.get(
                        "sha256",
                        "",
                    )
                ),

                "chunker_version": (
                    CHUNKER_VERSION
                ),

                "chunks": len(
                    chunks
                ),

                "explicit_headings": (
                    explicit_headings
                ),

                "structured_chunks": (
                    report_row[
                        "structured_chunks"
                    ]
                ),

                "general_chunks": (
                    report_row[
                        "general_chunks"
                    ]
                ),

                "status": "success",

                "error": "",

                "processed_at": (
                    utc_now()
                ),
            }

        except Exception as exc:

            failed_count += 1

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print(
                f" | FAILED | {error}"
            )

            report_rows.append(
                build_report_row(
                    document_id=(
                        document_id
                    ),
                    chunks=[],
                    explicit_headings=0,
                    status="failed",
                    error=error,
                )
            )

            manifest[
                document_id
            ] = {
                "document_id": (
                    document_id
                ),
                "source_sha256": (
                    document.get(
                        "sha256",
                        "",
                    )
                ),
                "chunker_version": (
                    CHUNKER_VERSION
                ),
                "chunks": 0,
                "explicit_headings": 0,
                "structured_chunks": 0,
                "general_chunks": 0,
                "status": "failed",
                "error": error,
                "processed_at": (
                    utc_now()
                ),
            }

    write_manifest(
        manifest
    )

    write_report(
        report_rows
    )

    global_chunks = (
        rebuild_global_chunks(
            all_document_ids
        )
    )

    counts = Counter(
        chunk.get(
            "section_type",
            "GENERAL",
        )
        for chunk
        in global_chunks
    )

    structured_total = sum(
        counts.get(
            section,
            0,
        )
        for section in (
            "FACTS",
            "ISSUES",
            "ARGUMENTS",
            "REASONING",
            "HOLDING",
            "ORDER",
        )
    )

    print()
    print(
        "Legal Chunking Report"
    )
    print("=" * 70)

    print(
        f"Processed this run: "
        f"{processed_count}"
    )

    print(
        f"Skipped unchanged: "
        f"{skipped_count}"
    )

    print(
        f"Failed this run: "
        f"{failed_count}"
    )

    print(
        f"Total chunks available: "
        f"{len(global_chunks)}"
    )

    print(
        f"Structured legal chunks: "
        f"{structured_total}"
    )

    for section in SECTION_TYPES:
        print(
            f"{section}: "
            f"{counts.get(section, 0)}"
        )

    print(
        f"Global chunks: "
        f"{GLOBAL_CHUNKS_PATH}"
    )

    print(
        f"Manifest: "
        f"{CHUNKING_MANIFEST_PATH}"
    )

    print(
        f"Report: "
        f"{CHUNKING_REPORT_PATH}"
    )

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Create legal-structure-aware chunks "
            "from processed judgments."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N "
            "documents."
        ),
    )

    parser.add_argument(
        "--document-id",
        default=None,
        help=(
            "Process a single document."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reprocess even when source "
            "hash and chunker version "
            "are unchanged."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    try:

        process_documents(
            force=args.force,
            limit=args.limit,
            document_id_filter=(
                args.document_id
            ),
        )

    except Exception as exc:

        print()
        print(
            "Legal structure chunking failed"
        )
        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print("=" * 70)

        sys.exit(1)


if __name__ == "__main__":
    main()