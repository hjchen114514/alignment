"""Shared fixtures: a tiny hand-built dataset written to a temp dir.

Two personas (pid 1, 2), four questions in two blocks:

    block A: q1, q2        block B: q3, q4

Normalized answers (q4 is only seen by pid 1, and its synthetic call failed):

            human        retest       synthetic
    pid 1   0, 1, .5, 0  0, .5, 1     .5, 1, .5, NULL
    pid 2   1, 0, 0      1, 0, 0      1, 0, 1

Every expected number in the tests is worked out by hand from this table.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
# aiGateway builds its OpenAI client at import; tests never reach the network
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


QUESTIONS = [
    # qid,  block, category
    ("q1", "A", "Behavioral economics"),
    ("q2", "A", "Behavioral economics"),
    ("q3", "B", "Pricing Preferences"),
    ("q4", "B", "Pricing Preferences"),
]
HUMAN = {1: {"q1": 0.0, "q2": 1.0, "q3": 0.5, "q4": 0.0},
         2: {"q1": 1.0, "q2": 0.0, "q3": 0.0}}
RETEST = {1: {"q1": 0.0, "q2": 0.5, "q3": 1.0},
          2: {"q1": 1.0, "q2": 0.0, "q3": 0.0}}
SYNTH = {1: {"q1": 0.5, "q2": 1.0, "q3": 0.5, "q4": None},
         2: {"q1": 1.0, "q2": 0.0, "q3": 1.0}}


def _answers(table: dict, with_status: bool = False) -> pd.DataFrame:
    rows = []
    for pid, answers in table.items():
        for qid, norm in answers.items():
            row = {"pid": pid, "qid": qid, "row_id": 0,
                   "answer": None if norm is None else int(norm * 10),
                   "answertext": None if norm is None else str(norm),
                   "normalized": norm}
            if with_status:
                row["status"] = "pending" if norm is None else "finished"
            rows.append(row)
    df = pd.DataFrame(rows)
    df["answer"] = df["answer"].astype("Int64")
    return df


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    """Write the tiny dataset and point every script module at it."""
    paths = {
        "HUMAN_PARQUET": tmp_path / "human_responses.parquet",
        "HUMAN_RETEST_PARQUET": tmp_path / "human_retest.parquet",
        "SYNTHETIC_PARQUET": tmp_path / "synthetic_responses.parquet",
        "QUESTIONS_PARQUET": tmp_path / "questions.parquet",
        "PERSONA_PARQUET": tmp_path / "persona.parquet",
        "PRICING_TEXT_PARQUET": tmp_path / "humanPricingQuestionText.parquet",
    }
    _answers(HUMAN).to_parquet(paths["HUMAN_PARQUET"], index=False)
    _answers(RETEST).to_parquet(paths["HUMAN_RETEST_PARQUET"], index=False)
    _answers(SYNTH, with_status=True).to_parquet(paths["SYNTHETIC_PARQUET"], index=False)
    pd.DataFrame([
        {"qid": q, "row_id": 0, "category": c, "block": b, "question_type": "Slider",
         "questiontext": f"text {q}", "options": [], "range_min": 0.0, "range_max": 10.0}
        for q, b, c in QUESTIONS
    ]).to_parquet(paths["QUESTIONS_PARQUET"], index=False)
    pd.DataFrame([
        {"pid": 1, "gender": "Male", "age": "18-29", "education": "College",
         "race": "White", "favoredPoliticalParty": "Democrat", "income": "Low",
         "politicalViews": "Liberal", "icl": "persona one"},
        {"pid": 2, "gender": "Female", "age": "18-29", "education": "College",
         "race": "White", "favoredPoliticalParty": "Republican", "income": "Low",
         "politicalViews": "Conservative", "icl": "persona two"},
    ]).to_parquet(paths["PERSONA_PARQUET"], index=False)
    # pricing questions show each persona its own product and price
    pd.DataFrame([
        {"pid": 1, "qid": "q3", "questiontext": "butter $4.99"},
        {"pid": 1, "qid": "q4", "questiontext": "ham $35.99"},
        {"pid": 2, "qid": "q3", "questiontext": "chips $0.00"},
    ]).to_parquet(paths["PRICING_TEXT_PARQUET"], index=False)

    import alignmentResult
    import distances
    import drawDiagram
    import getSyntheticData
    for module in (alignmentResult, distances, drawDiagram, getSyntheticData):
        for name, path in paths.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, path)
    monkeypatch.setattr(alignmentResult, "NUMBERS_DIR", tmp_path)
    return paths
