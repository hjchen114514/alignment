"""Shared config loading and path helpers."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()

DATA_DIR = ROOT / CFG["data_dir"]
RESULT_DIR = ROOT / CFG["result_dir"]
NUMBERS_DIR = RESULT_DIR / "numbers"
DIAGRAM_DIR = RESULT_DIR / "diagram"

HUMAN_PARQUET = DATA_DIR / "human_responses.parquet"
HUMAN_RETEST_PARQUET = DATA_DIR / "human_retest.parquet"
SYNTHETIC_PARQUET = DATA_DIR / "synthetic_responses.parquet"
SYNTHETIC_JSONL = DATA_DIR / "synthetic_responses.jsonl"
QUESTIONS_PARQUET = DATA_DIR / "questions.parquet"
PERSONA_PARQUET = DATA_DIR / "persona.parquet"


def categorize(block_name: str) -> str:
    """Map a raw BlockName to one of the three analysis categories."""
    for rule in CFG["categories"]:
        if rule["match"] == "*":
            return rule["name"]
        if rule["match"].lower() in block_name.lower():
            return rule["name"]
    return "Behavioral economics"


def is_skipped_block(block_name: str) -> bool:
    return any(s.lower() in block_name.lower() for s in CFG["skip_blocks"])


def is_skipped_qid(qid: str) -> bool:
    return qid in set(CFG["skip_qids"])


def ensure_dirs() -> None:
    for d in (DATA_DIR, NUMBERS_DIR, DIAGRAM_DIR):
        d.mkdir(parents=True, exist_ok=True)
