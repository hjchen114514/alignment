"""Prompt rendering and response parsing (adaptors/aiGateway.py). No network."""
import json
from types import SimpleNamespace

import pytest

from adaptors import aiGateway

QUESTIONS = [
    {"qid": "M", "row_id": 0, "question_type": "MC", "questiontext": "Pick one",
     "options": ["red", "blue"], "range_min": 1.0, "range_max": 2.0},
    {"qid": "S", "row_id": 0, "question_type": "Slider", "questiontext": "How much",
     "options": [], "range_min": 0.0, "range_max": 100.0},
]


def test_render_questions():
    text = aiGateway.render_questions(QUESTIONS)
    assert "Answer all 2 questions below, in order, as a list of exactly 2 integers." in text
    assert "Q1: Pick one\n  1. red\n  2. blue\n  (answer with a number from 1 to 2)" in text
    assert "Q2: How much\n  (answer with a whole number from 0 to 100)" in text


def test_parse_answers():
    assert aiGateway.parse_answers('{"answers": [2, 40]}', 2) == [2, 40]
    with pytest.raises(ValueError):
        aiGateway.parse_answers('{"answers": [2]}', 2)


def test_ask_ai_request(monkeypatch):
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(output_text=json.dumps({"answers": [1, 70]}))

    monkeypatch.setattr(aiGateway, "client",
                        SimpleNamespace(responses=SimpleNamespace(create=create)))

    assert aiGateway.ask_ai("PERSONA TEXT", QUESTIONS, 7) == [1, 70]
    assert sent["model"] == aiGateway.CFG["model"]
    assert sent["prompt_cache_key"] == "persona-7"
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["format"]["strict"] is True
    # persona text first (the cached prefix), questions after it
    assert "PERSONA TEXT" in sent["input"][0]["content"]
    assert "Q1: Pick one" in sent["input"][1]["content"]
    assert "tools" not in sent
