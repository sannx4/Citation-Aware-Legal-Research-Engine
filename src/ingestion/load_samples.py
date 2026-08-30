from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import RAW_CASES_DIR, CASE_METADATA_PATH


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "case_id",
        "file_name",
        "title",
        "court",
        "year",
        "source_type",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {missing_columns}")

    return metadata


def verify_case_files(metadata: pd.DataFrame, sample_cases_dir: Path) -> list[dict]:
    verification_results = []

    for _, row in metadata.iterrows():
        file_path = sample_cases_dir / row["file_name"]

        verification_results.append(
            {
                "case_id": row["case_id"],
                "title": row["title"],
                "file_name": row["file_name"],
                "file_path": str(file_path),
                "exists": file_path.exists(),
            }
        )

    return verification_results


def print_report(results: list[dict]) -> None:
    print("\nDay 2 Sample Corpus Verification Report")
    print("=" * 70)

    missing_files = 0

    for item in results:
        status = "FOUND" if item["exists"] else "MISSING"

        if not item["exists"]:
            missing_files += 1

        print(
            f"{item['case_id']} | {status} | "
            f"{item['file_name']} | {item['title']}"
        )

    print("=" * 70)
    print(f"Total cases checked: {len(results)}")
    print(f"Missing files: {missing_files}")

    if missing_files == 0:
        print("All sample case files verified successfully.")
    else:
        print("Some files are missing. Fix them before Day 3.")


def main() -> None:
    metadata = load_metadata(CASE_METADATA_PATH)
    results = verify_case_files(metadata, RAW_CASES_DIR)
    print_report(results)


if __name__ == "__main__":
    main()