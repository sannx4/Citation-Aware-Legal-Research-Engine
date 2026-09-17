import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import (
    DEDUPED_MANIFEST_PATH,
    FILE_HASH_INDEX_PATH,
)


# ============================================================
# Helpers
# ============================================================

def clean(value) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() == "nan":
        return ""

    return value


def sha256_file(
    file_path: Path,
) -> str:

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:

        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


# ============================================================
# Existing hash index
# ============================================================

def load_old_hash_index() -> dict[str, dict]:

    if not FILE_HASH_INDEX_PATH.exists():
        return {}

    try:
        old_index = pd.read_csv(
            FILE_HASH_INDEX_PATH
        )

    except pd.errors.EmptyDataError:
        return {}

    result = {}

    for _, row in old_index.iterrows():

        document_id = clean(
            row.get(
                "document_id",
                "",
            )
        )

        if not document_id:
            continue

        result[document_id] = {
            "sha256_hash": clean(
                row.get(
                    "sha256_hash",
                    "",
                )
            ),

            "processed_at": clean(
                row.get(
                    "processed_at",
                    "",
                )
            ),

            "status": clean(
                row.get(
                    "status",
                    "",
                )
            ),
        }

    return result


# ============================================================
# Load canonical corpus
# ============================================================

def load_deduped_corpus() -> pd.DataFrame:
    """
    Gap 36 produces manifest_deduped.csv.

    That file contains only:

        valid PDFs
        minus physical duplicates

    Current expected size:

        20 valid PDFs
        - 1 duplicate
        = 19 canonical documents
    """

    if not DEDUPED_MANIFEST_PATH.exists():

        raise FileNotFoundError(
            "Run duplicate detection first. "
            f"Missing: {DEDUPED_MANIFEST_PATH}"
        )

    try:

        corpus = pd.read_csv(
            DEDUPED_MANIFEST_PATH
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame()

    required_columns = {
        "document_id",
        "file_name",
        "file_path",
    }

    missing_columns = (
        required_columns
        - set(corpus.columns)
    )

    if missing_columns:

        raise ValueError(
            "Deduped corpus is missing "
            f"columns: "
            f"{sorted(missing_columns)}"
        )

    return corpus


# ============================================================
# Build hash index
# ============================================================

def build_hash_index(
    corpus: pd.DataFrame,
) -> list[dict]:

    old_index = (
        load_old_hash_index()
    )

    rows = []

    now = datetime.now().isoformat(
        timespec="seconds"
    )

    for _, row in corpus.iterrows():

        document_id = clean(
            row.get(
                "document_id",
                "",
            )
        )

        file_name = clean(
            row.get(
                "file_name",
                "",
            )
        )

        file_path_value = clean(
            row.get(
                "file_path",
                "",
            )
        )

        if not document_id:
            continue

        # ----------------------------------------------------
        # Use exact path recorded by integrity/dedup pipeline
        # ----------------------------------------------------

        if file_path_value:

            file_path = Path(
                file_path_value
            )

        else:

            file_path = Path(
                file_name
            )

        # ----------------------------------------------------
        # Missing physical file
        # ----------------------------------------------------

        if not file_path.exists():

            rows.append(
                {
                    "document_id":
                        document_id,

                    "file_name":
                        file_name,

                    "file_path":
                        str(file_path),

                    "sha256_hash":
                        "",

                    "file_size":
                        0,

                    "processed_at":
                        "",

                    "status":
                        "missing_file",

                    "reprocess_needed":
                        "yes",
                }
            )

            continue

        # ----------------------------------------------------
        # Current hash
        # ----------------------------------------------------

        current_hash = sha256_file(
            file_path
        )

        file_size = (
            file_path.stat().st_size
        )

        old = old_index.get(
            document_id
        )

        # ----------------------------------------------------
        # Unchanged
        # ----------------------------------------------------

        if (
            old
            and
            old.get(
                "sha256_hash"
            )
            ==
            current_hash
        ):

            status = "unchanged"

            processed_at = (
                old.get(
                    "processed_at"
                )
                or now
            )

            reprocess_needed = "no"

        # ----------------------------------------------------
        # New or modified
        # ----------------------------------------------------

        else:

            status = (
                "new_or_changed"
            )

            processed_at = now

            reprocess_needed = "yes"

        rows.append(
            {
                "document_id":
                    document_id,

                "file_name":
                    file_name,

                "file_path":
                    str(file_path),

                "sha256_hash":
                    current_hash,

                "file_size":
                    file_size,

                "processed_at":
                    processed_at,

                "status":
                    status,

                "reprocess_needed":
                    reprocess_needed,
            }
        )

    return rows


# ============================================================
# Write index
# ============================================================

def write_hash_index(
    rows: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "document_id",
        "file_name",
        "file_path",
        "sha256_hash",
        "file_size",
        "processed_at",
        "status",
        "reprocess_needed",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    corpus = load_deduped_corpus()

    rows = build_hash_index(
        corpus
    )

    write_hash_index(
        rows,
        FILE_HASH_INDEX_PATH,
    )

    total = len(rows)

    unchanged = sum(
        1
        for row in rows
        if row["status"]
        == "unchanged"
    )

    changed = sum(
        1
        for row in rows
        if row["status"]
        == "new_or_changed"
    )

    missing = sum(
        1
        for row in rows
        if row["status"]
        == "missing_file"
    )

    print(
        "\nGap 39 Hash Index Report"
    )

    print("=" * 70)

    print(
        f"Canonical documents indexed: "
        f"{total}"
    )

    print(
        f"Unchanged files: "
        f"{unchanged}"
    )

    print(
        f"New or changed files: "
        f"{changed}"
    )

    print(
        f"Missing files: "
        f"{missing}"
    )

    print(
        f"Hash index saved to: "
        f"{FILE_HASH_INDEX_PATH}"
    )

    print("=" * 70)

    for row in rows:

        print(
            f"{row['document_id']} | "
            f"{row['status']} | "
            f"reprocess="
            f"{row['reprocess_needed']} | "
            f"{row['sha256_hash'][:12]}"
        )


if __name__ == "__main__":
    main()