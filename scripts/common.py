# This file contains shared functions and constants used by multiple scripts in the project. It is not intended to be run directly, but rather imported by other scripts.
"""Shared config loading and path helpers."""
from pathlib import Path
import yaml

# load the root directory of the project, which is two levels up from this file
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

# load the config file once and store it in a global variable
CFG = load_config()

# here we define some paths to data and result directories, as well as specific files used in the analysis
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
        # * is a wildcard that matches any block name
        if rule["match"] == "*":
            return rule["name"]
        if rule["match"].lower() in block_name.lower():
            return rule["name"]
    return "Behavioral economics"

# skiopped blocks and qids are defined in the config yaml file, and we provide helper functions to check if a given block or qid should be skipped
def is_skipped_block(block_name: str) -> bool:
    return any(s.lower() in block_name.lower() for s in CFG["skip_blocks"])


def is_skipped_qid(qid: str) -> bool:
    return qid in set(CFG["skip_qids"])

# ensure that the data and result directories exist, creating them if necessary
# why not including the diagram directory? 
# Because it is created by the diagram generation script, which is run after the analysis script. The analysis script only needs to ensure that the data and result directories exist, since it will write its output to the result directory.
def ensure_dirs() -> None:
    for d in (DATA_DIR, NUMBERS_DIR, DIAGRAM_DIR):
        d.mkdir(parents=True, exist_ok=True)
