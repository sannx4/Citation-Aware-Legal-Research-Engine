import csv
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import NORMALIZED_CITATIONS_PATH, REPORTS_DIR


GOLD_CITATIONS_PATH = ROOT_DIR / "data" / "eval" / "gold_citations.csv"
EVAL_REPORT_PATH = REPORTS_DIR / "citation_evaluation_report.csv"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "citation_error_analysis.md"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def build_key_set(
    dataframe: pd.DataFrame,
    case_column: str = "case_id",
) -> set[tuple[str, str, str]]:
    keys = set()

    for _, row in dataframe.iterrows():
        case_id = str(row[case_column]).strip()
        canonical_id = str(row["canonical_id"]).strip()
        mention_type = str(row["mention_type"]).strip()

        keys.add(
            (
                case_id,
                canonical_id,
                mention_type,
            )
        )

    return keys


def compute_metrics(
    gold_keys: set[tuple[str, str, str]],
    predicted_keys: set[tuple[str, str, str]],
) -> dict:

    true_positives = gold_keys & predicted_keys
    false_positives = predicted_keys - gold_keys
    false_negatives = gold_keys - predicted_keys

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def write_report(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow(
            {
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            }
        )


def format_key(key: tuple[str, str, str]) -> str:
    case_id, canonical_id, mention_type = key
    return f"{case_id} | {mention_type} | {canonical_id}"


def write_error_analysis(metrics: dict, output_path: Path) -> None:
    lines = [
        "# Day 14 Citation Extraction Error Analysis",
        "",
        "## Summary",
        "",
        f"- True Positives: {metrics['tp']}",
        f"- False Positives: {metrics['fp']}",
        f"- False Negatives: {metrics['fn']}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        "",
        "## False Positives",
        "",
        "These are citations predicted by the system but not present in the gold labels.",
        "",
    ]

    if metrics["false_positives"]:
        for item in sorted(metrics["false_positives"]):
            lines.append(f"- {format_key(item)}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## False Negatives",
            "",
            "These are citations present in gold labels but missed by the system.",
            "",
        ]
    )

    if metrics["false_negatives"]:
        for item in sorted(metrics["false_negatives"]):
            lines.append(f"- {format_key(item)}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- High precision means the extractor avoids fake citations.",
            "- High recall means the extractor finds most real citations.",
            "- False positives usually come from regex over-matching.",
            "- False negatives usually come from missing citation patterns.",
            "",
            "## Limitation",
            "",
            "This evaluation depends on manually created gold labels. If the gold file is incomplete, the metric will be misleading.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    gold = load_csv(GOLD_CITATIONS_PATH)
    predicted = load_csv(NORMALIZED_CITATIONS_PATH)

    gold_keys = build_key_set(gold, case_column="case_id")
    predicted_keys = build_key_set(predicted, case_column="source_case_id")

    metrics = compute_metrics(gold_keys, predicted_keys)

    write_report(metrics, EVAL_REPORT_PATH)
    write_error_analysis(metrics, ERROR_ANALYSIS_PATH)

    print("\nDay 14 Citation Evaluation Report")
    print("=" * 70)
    print(f"Gold citations: {len(gold_keys)}")
    print(f"Predicted citations: {len(predicted_keys)}")
    print(f"True positives: {metrics['tp']}")
    print(f"False positives: {metrics['fp']}")
    print(f"False negatives: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Report saved to: {EVAL_REPORT_PATH}")
    print(f"Error analysis saved to: {ERROR_ANALYSIS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()