"""Normalization and question parsing (importData.py, common.py)."""
import pytest

import importData
from common import categorize, is_skipped_block, is_skipped_qid
from importData import normalize, parse_question, to_int


@pytest.mark.parametrize("answer, lo, hi, expected", [
    (1, 1, 5, 0.0),       # bottom of a 5-point scale
    (5, 1, 5, 1.0),       # top
    (3, 1, 5, 0.5),       # middle
    (25, 0, 100, 0.25),   # slider
    (7, 0, 20, 0.35),     # text-entry with a config range
    (3, 3, 3, 0.0),       # degenerate one-option range does not divide by zero
])
def test_normalize(answer, lo, hi, expected):
    assert normalize(answer, lo, hi) == pytest.approx(expected)


@pytest.mark.parametrize("raw, expected", [(3, 3), ("50", 50), ("50.0", 50), (7.9, 7)])
def test_to_int(raw, expected):
    assert to_int(raw) == expected


def test_parse_matrix_yields_one_row_per_answered_row():
    q = {"QuestionID": "Q1", "QuestionType": "Matrix", "QuestionText": "Rate:",
         "Rows": ["apples", "pears", "plums"], "Columns": ["bad", "ok", "good"],
         "Answers": {"SelectedByPosition": [3, None, "1"],
                     "SelectedText": ["good", None, "bad"]}}
    rows = list(parse_question(q, "block"))
    assert [(r["row_id"], r["answer"]) for r in rows] == [(0, 3), (2, 1)]
    assert rows[0]["questiontext"] == "Rate: apples"
    assert (rows[0]["range_min"], rows[0]["range_max"]) == (1.0, 3.0)


def test_parse_mc():
    q = {"QuestionID": "Q2", "QuestionType": "MC", "QuestionText": "Pick",
         "Options": ["a", "b", "c", "d"], "Answers": {"SelectedByPosition": 2}}
    (row,) = parse_question(q, "block")
    assert row["answer"] == 2 and row["answertext"] == "b"
    assert (row["range_min"], row["range_max"]) == (1.0, 4.0)


def test_parse_mc_unanswered_yields_nothing():
    q = {"QuestionID": "Q2", "QuestionType": "MC", "Options": ["a"], "Answers": {}}
    assert list(parse_question(q, "block")) == []


def test_parse_slider_uses_source_range():
    q = {"QuestionID": "Q3", "QuestionType": "Slider", "QuestionText": "How much",
         "Range": {"Min": 10, "Max": 50}, "Answers": {"Values": ["20", None, 50.0]}}
    rows = list(parse_question(q, "block"))
    assert [(r["row_id"], r["answer"]) for r in rows] == [(0, 20), (2, 50)]
    assert (rows[0]["range_min"], rows[0]["range_max"]) == (10.0, 50.0)


def test_parse_slider_appends_each_statement():
    q = {"QuestionID": "Q4", "QuestionType": "Slider", "QuestionText": "Support for:",
         "Statements": ["carbon tax", "paid leave"], "Range": {"Min": 0, "Max": 100},
         "Answers": {"Values": ["30", "60"]}}
    rows = list(parse_question(q, "block"))
    assert [r["questiontext"] for r in rows] == ["Support for: carbon tax", "Support for: paid leave"]


def test_parse_slider_empty_statement_keeps_stem():
    q = {"QuestionID": "Q5", "QuestionType": "Slider", "QuestionText": "How likely?",
         "Statements": [""], "Range": {"Min": 0, "Max": 100}, "Answers": {"Values": ["50"]}}
    (row,) = parse_question(q, "block")
    assert row["questiontext"] == "How likely?"


def test_parse_te_uses_config_range(monkeypatch):
    monkeypatch.setattr(importData, "TE_RANGES", {"QTE": (0, 20)})
    q = {"QuestionID": "QTE", "QuestionType": "TE", "Answers": {"Text": "12"}}
    (row,) = parse_question(q, "block")
    assert row["answer"] == 12
    assert (row["range_min"], row["range_max"]) == (0.0, 20.0)


def test_parse_te_without_range_raises(monkeypatch):
    monkeypatch.setattr(importData, "TE_RANGES", {})
    q = {"QuestionID": "QTE", "QuestionType": "TE", "Answers": {"Text": "12"}}
    with pytest.raises(ValueError, match="te_ranges"):
        list(parse_question(q, "block"))


@pytest.mark.parametrize("block, expected", [
    ("Product Preferences - Pricing", "Pricing Preferences"),
    ("Non-experimental heuristics and biases", "Non-experimental heuristics and biases"),
    ("False consensus", "Behavioral economics"),
])
def test_categorize(block, expected):
    assert categorize(block) == expected


def test_skips():
    assert is_skipped_block("Anchoring - Problem 1")
    assert not is_skipped_block("False consensus")
    assert is_skipped_qid("QID163")
    assert not is_skipped_qid("QID1")
