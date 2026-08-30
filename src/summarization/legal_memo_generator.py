import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.config import (
    LEGAL_PROMPT_PATH,
    LEGAL_MEMO_PATH,
    OLLAMA_EXE_PATH,
    OLLAMA_MODEL_NAME,
)


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    return path.read_text(encoding="utf-8")


def generate_memo(prompt: str) -> str:
    result = subprocess.run(
        [
            OLLAMA_EXE_PATH,
            "run",
            OLLAMA_MODEL_NAME,
        ],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


def save_memo(memo: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memo, encoding="utf-8")


def main() -> None:
    prompt = load_prompt(LEGAL_PROMPT_PATH)

    memo = generate_memo(prompt)

    save_memo(memo, LEGAL_MEMO_PATH)

    print("\nPhase 5.3 LLM Legal Memo Generator Report")
    print("=" * 70)
    print(f"Model: {OLLAMA_MODEL_NAME}")
    print(f"Memo saved to: {LEGAL_MEMO_PATH}")
    print("=" * 70)
    print(memo[:1500])
    print("\n... memo truncated in console ...")


if __name__ == "__main__":
    main()