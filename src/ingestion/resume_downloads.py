import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import DOWNLOAD_STATE_PATH


def default_state() -> dict:
    return {
        "last_successful_document": "",
        "completed_documents": [],
        "failed_documents": [],
        "pending_documents": [],
        "updated_at": "",
    }


def load_download_state() -> dict:
    if not DOWNLOAD_STATE_PATH.exists():
        return default_state()

    try:
        with DOWNLOAD_STATE_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default_state()


def save_download_state(state: dict) -> None:
    DOWNLOAD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")

    with DOWNLOAD_STATE_PATH.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def initialize_pending_documents(document_ids: list[str]) -> dict:
    state = load_download_state()

    completed = set(state.get("completed_documents", []))
    failed = set(state.get("failed_documents", []))

    state["pending_documents"] = [
        document_id
        for document_id in document_ids
        if document_id not in completed and document_id not in failed
    ]

    save_download_state(state)
    return state


def should_skip_document(document_id: str) -> bool:
    state = load_download_state()
    completed = set(state.get("completed_documents", []))
    return document_id in completed


def mark_download_success(document_id: str) -> None:
    state = load_download_state()

    completed = set(state.get("completed_documents", []))
    failed = set(state.get("failed_documents", []))
    pending = set(state.get("pending_documents", []))

    completed.add(document_id)
    failed.discard(document_id)
    pending.discard(document_id)

    state["last_successful_document"] = document_id
    state["completed_documents"] = sorted(completed)
    state["failed_documents"] = sorted(failed)
    state["pending_documents"] = sorted(pending)

    save_download_state(state)


def mark_download_failed(document_id: str) -> None:
    state = load_download_state()

    completed = set(state.get("completed_documents", []))
    failed = set(state.get("failed_documents", []))
    pending = set(state.get("pending_documents", []))

    if document_id not in completed:
        failed.add(document_id)

    pending.discard(document_id)

    state["completed_documents"] = sorted(completed)
    state["failed_documents"] = sorted(failed)
    state["pending_documents"] = sorted(pending)

    save_download_state(state)


def print_resume_report() -> None:
    state = load_download_state()

    print("\nGap 41 Resume Download State")
    print("=" * 70)
    print(f"Last successful document: {state.get('last_successful_document', '')}")
    print(f"Completed documents: {len(state.get('completed_documents', []))}")
    print(f"Failed documents: {len(state.get('failed_documents', []))}")
    print(f"Pending documents: {len(state.get('pending_documents', []))}")
    print(f"State saved to: {DOWNLOAD_STATE_PATH}")
    print("=" * 70)
