from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

VERSIONS_DIR = RAW_DIR / "corpus_versions"

REGISTRY_PATH = RAW_DIR / "corpus_registry.csv"
DEDUPED_MANIFEST_PATH = RAW_DIR / "manifest_deduped.csv"
SEED_URLS_PATH = RAW_DIR / "seed_urls.csv"

CORPUS_STATS_PATH = REPORTS_DIR / "corpus_stats.json"
METADATA_REPORT_PATH = REPORTS_DIR / "metadata_backfill_report.csv"

DUPLICATE_REPORT_PATH = (
    RAW_DIR
    / "sample_cases"
    / "duplicate_downloads.csv"
)

HASH_INDEX_PATH = (
    RAW_DIR
    / "sample_cases"
    / "file_hash_index.csv"
)


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    """
    Calculate SHA-256 for a file.
    """

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def count_csv_rows(path: Path) -> int:
    """
    Count data rows in a CSV, excluding the header.
    """

    if not path.exists():
        return 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        return sum(1 for _ in reader)


def copy_snapshot_file(
    source: Path,
    destination_dir: Path,
) -> dict | None:
    """
    Copy one snapshot source file and return its metadata.
    """

    if not source.exists():
        return None

    destination = destination_dir / source.name

    shutil.copy2(
        source,
        destination,
    )

    return {
        "file_name": source.name,
        "source_path": str(source),
        "snapshot_path": str(destination),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def load_corpus_stats() -> dict:
    """
    Load corpus_stats.json when available.
    """

    if not CORPUS_STATS_PATH.exists():
        return {}

    try:
        with CORPUS_STATS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception:
        return {}


# ============================================================
# Version creation
# ============================================================

def create_corpus_version(
    version: str,
    force: bool = False,
) -> None:

    if not version.strip():
        raise ValueError(
            "Version name cannot be empty."
        )

    version = version.strip()

    version_dir = VERSIONS_DIR / version

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    if version_dir.exists():
        if not force:
            raise FileExistsError(
                f"Corpus version already exists: "
                f"{version_dir}\n"
                f"Use --force only if you intentionally "
                f"want to replace it."
            )

        shutil.rmtree(version_dir)

    version_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    required_files = [
        REGISTRY_PATH,
        DEDUPED_MANIFEST_PATH,
    ]

    missing_required = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_required:
        shutil.rmtree(
            version_dir,
            ignore_errors=True,
        )

        raise FileNotFoundError(
            "Required corpus snapshot files are missing:\n"
            + "\n".join(missing_required)
        )

    # --------------------------------------------------------
    # Snapshot files
    # --------------------------------------------------------

    snapshot_sources = [
        REGISTRY_PATH,
        DEDUPED_MANIFEST_PATH,
        SEED_URLS_PATH,
        CORPUS_STATS_PATH,
        METADATA_REPORT_PATH,
        DUPLICATE_REPORT_PATH,
        HASH_INDEX_PATH,
    ]

    snapshot_files = []

    for source in snapshot_sources:
        metadata = copy_snapshot_file(
            source,
            version_dir,
        )

        if metadata:
            snapshot_files.append(metadata)

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    registry_records = count_csv_rows(
        REGISTRY_PATH
    )

    canonical_judgments = count_csv_rows(
        DEDUPED_MANIFEST_PATH
    )

    seed_records = count_csv_rows(
        SEED_URLS_PATH
    )

    duplicate_records = count_csv_rows(
        DUPLICATE_REPORT_PATH
    )

    corpus_stats = load_corpus_stats()

    # --------------------------------------------------------
    # Version metadata
    # --------------------------------------------------------

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    version_metadata = {
        "version": version,
        "created_at_utc": created_at,
        "project_root": str(PROJECT_ROOT),

        "corpus": {
            "registry_records": registry_records,
            "seed_records": seed_records,
            "canonical_judgments": canonical_judgments,
            "duplicate_records": duplicate_records,
        },

        "statistics": corpus_stats,

        "snapshot_files": snapshot_files,

        "notes": (
            "Corpus version snapshot. "
            "Judgment PDFs themselves are not duplicated. "
            "This snapshot records manifests, metadata, "
            "statistics, hashes, and deduplication state."
        ),
    }

    version_json_path = (
        version_dir
        / "version.json"
    )

    with version_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            version_metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    summary_path = (
        version_dir
        / "README.txt"
    )

    summary = f"""
Legal Research Engine Corpus Version
====================================

Version:
{version}

Created:
{created_at}

Registry records:
{registry_records}

Seed records:
{seed_records}

Canonical unique judgments:
{canonical_judgments}

Duplicate/common-judgment records:
{duplicate_records}

Snapshot directory:
{version_dir}

Important:
The actual PDF files are not copied into this snapshot.
The version records the corpus manifests, metadata,
deduplication state, hashes, and statistics needed to
identify the corpus state reproducibly.
""".strip()

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print(
        "Corpus Version Snapshot Report"
    )
    print("=" * 70)

    print(
        f"Version: {version}"
    )

    print(
        f"Registry records: "
        f"{registry_records}"
    )

    print(
        f"Seed records: "
        f"{seed_records}"
    )

    print(
        f"Canonical judgments: "
        f"{canonical_judgments}"
    )

    print(
        f"Duplicate records: "
        f"{duplicate_records}"
    )

    print(
        f"Files snapshotted: "
        f"{len(snapshot_files)}"
    )

    print(
        f"Snapshot directory: "
        f"{version_dir}"
    )

    print(
        f"Version metadata: "
        f"{version_json_path}"
    )

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible corpus metadata snapshot."
        )
    )

    parser.add_argument(
        "--version",
        required=True,
        help=(
            "Corpus version name, "
            "for example v0.2-56cases"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace an existing snapshot "
            "with the same version name."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    create_corpus_version(
        version=args.version,
        force=args.force,
    )


if __name__ == "__main__":
    main()