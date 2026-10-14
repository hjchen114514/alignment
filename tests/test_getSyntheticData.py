"""Synthetic answer records, resume logic and JSONL collapse (getSyntheticData.py).

No network: aiGateway.ask_ai is replaced with a fake in every test.
"""
import json

import pandas as pd
import pytest

import getSyntheticData as gs

SLIDER = {"qid": "S", "row_id": 0, "question_type": "Slider", "questiontext": "s",
          "options": [], "range_min": 0.0, "range_max": 100.0}
MC = {"qid": "M", "row_id": 0, "question_type": "MC", "questiontext": "m",
      "options": ["a", "b", "c", "d", "e"], "range_min": 1.0, "range_max": 5.0}


@pytest.mark.parametrize("q, answer, expected_answer, expected_text, expected_norm", [
    (SLIDER, 25, 25, "25", 0.25),
    (SLIDER, 150, 100, "100", 1.0),   # clamped down to the range
    (SLIDER, -3, 0, "0", 0.0),        # clamped up
    (MC, 3, 3, "c", 0.5),             # 1-based option index
    (MC, 1, 1, "a", 0.0),
    (MC, 0, 1, "a", 0.0),             # 0 is out of range for a 1-based scale
    (MC, 9, 5, "e", 1.0),
    (MC, "4", 4, "d", 0.75),          # string from the model still works
])
def test_build_record(q, answer, expected_answer, expected_text, expected_norm):
    rec = gs.build_record(7, q, answer, "finished")
    assert rec["pid"] == 7 and rec["qid"] == q["qid"] and rec["status"] == "finished"
    assert rec["answer"] == expected_answer
    assert rec["answertext"] == expected_text
    assert rec["normalized"] == pytest.approx(expected_norm)


def test_build_record_failed_call():
    rec = gs.build_record(7, MC, None, "pending")
    assert rec["answer"] is None and rec["normalized"] is None
    assert rec["status"] == "pending"


@pytest.fixture
def jsonl(tmp_path, monkeypatch):
    path = tmp_path / "synthetic_responses.jsonl"
    monkeypatch.setattr(gs, "SYNTHETIC_JSONL", path)
    monkeypatch.setattr(gs, "SYNTHETIC_PARQUET", tmp_path / "synthetic_responses.parquet")
    return path


def write_lines(path, records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_already_done_only_counts_finished(jsonl):
    write_lines(jsonl, [gs.build_record(1, MC, 2, "finished"),
                        gs.build_record(1, SLIDER, None, "pending")])
    with open(jsonl, "a") as f:
        f.write('{"truncated by a crash\n')        # a half-written line is skipped
    assert gs.already_done() == {(1, "M", 0)}


def test_already_done_without_file(jsonl):
    assert gs.already_done() == set()


def test_clamp_is_reported(capsys):
    gs.build_record(7, MC, 9, "finished")
    assert "answer 9 outside [1, 5], clamped to 5" in capsys.readouterr().out


def test_jsonl_to_parquet_survives_truncated_line(jsonl):
    write_lines(jsonl, [gs.build_record(1, MC, 4, "finished")])
    with open(jsonl, "a") as f:
        f.write('{"pid": 1, "qid": "S", "row')        # killed mid-write
    gs.jsonl_to_parquet()
    assert len(pd.read_parquet(gs.SYNTHETIC_PARQUET)) == 1


def test_jsonl_to_parquet_finished_beats_pending(jsonl):
    # a failed first attempt, then a successful retry of the same question
    write_lines(jsonl, [gs.build_record(1, MC, None, "pending"),
                        gs.build_record(1, SLIDER, None, "pending"),
                        gs.build_record(1, MC, 4, "finished")])
    gs.jsonl_to_parquet()
    df = pd.read_parquet(gs.SYNTHETIC_PARQUET).set_index("qid")
    assert len(df) == 2
    assert df.loc["M", "status"] == "finished" and df.loc["M", "answer"] == 4
    assert df.loc["S", "status"] == "pending" and pd.isna(df.loc["S", "answer"])


def test_ask_question_success_failure_and_resume(jsonl, monkeypatch):
    blocks = {"good": [MC], "bad": [SLIDER]}
    monkeypatch.setattr(gs, "load_blocks_for_persona", lambda pid: blocks)

    def fake_ask(icl, questions, pid):
        if questions[0]["qid"] == "S":
            raise RuntimeError("api down")
        return [2] * len(questions)

    monkeypatch.setattr(gs.aiGateway, "ask_ai", fake_ask)
    assert gs.askQuestion("icl", 1, set()) == (1, 1)

    # rerun: the finished block is skipped, only the failed one is retried
    calls = []
    monkeypatch.setattr(gs.aiGateway, "ask_ai",
                        lambda icl, qs, pid: calls.append(qs) or [50] * len(qs))
    assert gs.askQuestion("icl", 1, gs.already_done()) == (1, 0)
    assert [q["qid"] for q in calls[0]] == ["S"]


def test_ask_question_wrong_answer_count_is_pending(jsonl, monkeypatch):
    monkeypatch.setattr(gs, "load_blocks_for_persona", lambda pid: {"b": [MC, SLIDER]})
    monkeypatch.setattr(gs.aiGateway, "ask_ai", lambda icl, qs, pid: [1])
    assert gs.askQuestion("icl", 1, set()) == (0, 2)


def test_load_blocks_from_worker_threads(tiny_data):
    # regression: threads sharing duckdb's default connection deadlocked here
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(gs.load_blocks_for_persona, [1, 2] * 8))
    assert {b: [q["qid"] for q in qs] for b, qs in results[0].items()} == \
        {"A": ["q1", "q2"], "B": ["q3", "q4"]}
    assert [q["qid"] for q in results[1]["B"]] == ["q3"]


def test_load_blocks_uses_each_personas_pricing_text(tiny_data):
    """Same pricing qid, different product per persona; other questions share one text."""
    one, two = gs.load_blocks_for_persona(1), gs.load_blocks_for_persona(2)
    assert [q["questiontext"] for q in one["B"]] == ["butter $4.99", "ham $35.99"]
    assert [q["questiontext"] for q in two["B"]] == ["chips $0.00"]
    assert [q["questiontext"] for q in two["A"]] == ["text q1", "text q2"]
