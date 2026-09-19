from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.citations import temporal_validity as tv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = PROJECT_ROOT / "data" / "eval" / "temporal_validity_gold.jsonl"


def load_gold() -> list[dict]:
    rows: list[dict] = []

    with GOLD_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Invalid JSON in {GOLD_PATH}:{line_number}"
                ) from exc

            rows.append(row)

    return rows


GOLD_SCENARIOS = load_gold()


def build_synthetic_timeline(
    scenario: dict,
) -> dict:
    authority_id = scenario["authority_id"]
    canonical_name = scenario["canonical_name"]

    registry = [
        {
            "case_id": authority_id,
            "canonical_name": canonical_name,
            "reporter_citations": [],
            "resolution_basis": "GOLD_TEST",
            "resolution_confidence": 1.0,
            "needs_review": False,
        }
    ]

    treatment_rows = []
    judgment_dates = {}

    for index, event in enumerate(
        scenario["events"],
        start=1,
    ):
        document_id = (
            f"GOLD_{scenario['scenario_id']}_{index:02d}"
        )

        treatment_rows.append(
            {
                "treatment_id": (
                    f"TREAT_{scenario['scenario_id']}_{index:02d}"
                ),
                "document_id": document_id,
                "target_case_id": authority_id,
                "target_case_name": canonical_name,
                "target_reporters": [],
                "treatment": event["treatment"],
                "treatment_actor": event["actor"],
                "confidence": event.get(
                    "confidence",
                    1.0,
                ),
                "resolution_basis": "GOLD_TEST",
                "resolution_confidence": 1.0,
                "resolution_needs_review": False,
                "detector_version": "GOLD",
                "resolver_version": "GOLD",
            }
        )

        judgment_dates[document_id] = {
            "document_id": document_id,
            "decision_date": event["date"],
            "raw_date": event["date"],
            "date_precision": "EXACT",
            "date_source": "document_text",
            "date_label": "GOLD_EXACT_DATE",
            "confidence": 1.0,
            "needs_review": False,
            "selection_reason": "gold_fixture",
            "date_extractor_version": "GOLD",
        }

    timelines = tv.build_timelines(
        treatment_rows=treatment_rows,
        registry=registry,
        document_metadata={},
        judgment_dates=judgment_dates,
    )

    assert len(timelines) == 1

    return timelines[0]


@pytest.mark.parametrize(
    "scenario",
    GOLD_SCENARIOS,
    ids=lambda row: row["scenario_id"],
)
def test_temporal_gold_scenarios(
    scenario: dict,
) -> None:
    timeline = build_synthetic_timeline(
        scenario
    )

    assert timeline["authority_id"] == scenario["authority_id"]
    assert timeline["canonical_name"] == scenario["canonical_name"]

    assert timeline["dated_event_count"] == len(
        scenario["events"]
    )

    assert timeline["undated_event_count"] == 0

    # All gold events use exact, trusted judgment dates.
    for event in timeline["dated_events"]:
        assert event["event_date_precision"] == "EXACT"
        assert event["event_date_source"] == "judgment_date_extractor"
        assert event["event_date_needs_review"] is False

    for query in scenario["queries"]:
        result = tv.get_status_as_of(
            timeline,
            query["as_of"],
        )

        assert (
            result["status"]
            == query["expected_status"]
        )

        assert (
            result["positive_judicial_treatment_count"]
            == query["expected_positive_judicial_count"]
        )

        assert (
            result["later_events_excluded"]
            == query["expected_later_events_excluded"]
        )

        assert (
            result["non_judicial_negative_claims_ignored"]
            == query["expected_nonjudicial_negative_ignored"]
        )

        expected_basis = query[
            "expected_basis_treatment"
        ]

        if expected_basis is None:
            assert result["basis_event"] is None
        else:
            assert result["basis_event"] is not None
            assert (
                result["basis_event"]["treatment"]
                == expected_basis
            )


@pytest.mark.parametrize(
    "scenario",
    GOLD_SCENARIOS,
    ids=lambda row: row["scenario_id"],
)
def test_no_future_leakage(
    scenario: dict,
) -> None:
    timeline = build_synthetic_timeline(
        scenario
    )

    assert (
        tv.count_temporal_leakage_violations(
            [timeline]
        )
        == 0
    )


def test_party_overruled_is_not_status_changing() -> None:
    scenario = next(
        row
        for row in GOLD_SCENARIOS
        if row["scenario_id"]
        == "party_overruled_claim_is_ignored"
    )

    timeline = build_synthetic_timeline(
        scenario
    )

    event = timeline["dated_events"][0]

    assert event["treatment"] == "OVERRULED"
    assert event["treatment_actor"] == "PARTY"
    assert event["is_status_changing"] is False
    assert (
        event["status_effect"]
        == "NO_STATUS_CHANGE_NON_JUDICIAL"
    )


def test_court_overruled_is_status_changing() -> None:
    scenario = next(
        row
        for row in GOLD_SCENARIOS
        if row["scenario_id"]
        == "court_overruled_changes_status"
    )

    timeline = build_synthetic_timeline(
        scenario
    )

    event = timeline["dated_events"][0]

    assert event["treatment"] == "OVERRULED"
    assert event["treatment_actor"] == "COURT"
    assert event["is_status_changing"] is True
    assert event["status_effect"] == "OVERRULED"


def test_future_overruling_is_not_visible_early() -> None:
    scenario = next(
        row
        for row in GOLD_SCENARIOS
        if row["scenario_id"]
        == "future_overruling_is_excluded"
    )

    timeline = build_synthetic_timeline(
        scenario
    )

    early = tv.get_status_as_of(
        timeline,
        "2024-12-31",
    )

    late = tv.get_status_as_of(
        timeline,
        "2026-01-01",
    )

    assert (
        early["status"]
        == "NO_OBSERVED_NEGATIVE_TREATMENT"
    )
    assert early["later_events_excluded"] == 1

    assert late["status"] == "OVERRULED"
    assert late["later_events_excluded"] == 0


def test_overruled_is_not_silently_undone_by_later_following() -> None:
    scenario = next(
        row
        for row in GOLD_SCENARIOS
        if row["scenario_id"]
        == "later_followed_does_not_undo_overruled"
    )

    timeline = build_synthetic_timeline(
        scenario
    )

    result = tv.get_status_as_of(
        timeline,
        "2024-01-01",
    )

    assert result["status"] == "OVERRULED"
    assert result["basis_event"] is not None
    assert (
        result["basis_event"]["treatment"]
        == "OVERRULED"
    )


def test_gold_file_covers_required_negative_transitions() -> None:
    observed = {
        event["treatment"]
        for scenario in GOLD_SCENARIOS
        for event in scenario["events"]
        if event["actor"] == "COURT"
    }

    required = {
        "FOLLOWED",
        "DISTINGUISHED",
        "DOUBTED",
        "DISAPPROVED",
        "OVERRULED",
    }

    assert required.issubset(
        observed
    )
