"""Alignment and ceiling (retest) (alignmentResult.py).

Expected values are hand-computed from the table in conftest.py.
"""
import pytest

import alignmentResult as ar


def by_pid(df):
    return df.set_index("pid")


def test_persona_alignment_is_two_stage(tiny_data):
    df = by_pid(ar.persona_alignment())
    # pid 1 errors: block A (.5, 0) -> .25, block B (0) -> 0   => 1 - .125
    # (a flat mean over the 3 answers would give 1 - .5/3 = .8333)
    assert df.loc[1, "alignment"] == pytest.approx(0.875)
    # pid 2 errors: block A (0, 0) -> 0, block B (1) -> 1        => 1 - .5
    assert df.loc[2, "alignment"] == pytest.approx(0.5)


def test_pending_answers_are_excluded(tiny_data):
    df = by_pid(ar.persona_alignment())
    # pid 1 has 4 synthetic rows, but q4 is pending (NULL)
    assert df.loc[1, "n_question"] == 3
    assert df.loc[2, "n_question"] == 3


def test_persona_ceiling_from_retest(tiny_data):
    df = by_pid(ar.persona_alignment())
    # pid 1 retest errors: A (0, .5) -> .25, B (.5) -> .5        => 1 - .375
    assert df.loc[1, "ceiling"] == pytest.approx(0.625)
    # pid 2 answered identically twice
    assert df.loc[2, "ceiling"] == pytest.approx(1.0)


def test_question_alignment(tiny_data):
    df = ar.question_alignment().set_index("qid")
    # q4 has no finished synthetic answer, so it has no row
    assert set(df.index) == {"q1", "q2", "q3"}

    # alignment = 1 - mean |synth - human| across personas
    assert df.loc["q1", "alignment"] == pytest.approx(1 - (0.5 + 0) / 2)
    assert df.loc["q2", "alignment"] == pytest.approx(1.0)
    assert df.loc["q3", "alignment"] == pytest.approx(1 - (0 + 1) / 2)

    # ceiling = 1 - mean |human - retest| across personas
    assert df.loc["q1", "ceiling"] == pytest.approx(1.0)
    assert df.loc["q2", "ceiling"] == pytest.approx(1 - (0.5 + 0) / 2)
    assert df.loc["q3", "ceiling"] == pytest.approx(1 - (0.5 + 0) / 2)

    # sample std of the two personas' answers: |a - b| / sqrt(2)
    assert df.loc["q1", "synthetic_std"] == pytest.approx(0.5 / 2 ** 0.5)
    assert df.loc["q1", "human_std"] == pytest.approx(1 / 2 ** 0.5)
    assert df.loc["q2", "n_persona"] == 2
    assert df.loc["q1", "block"] == "A"


def test_groupings(tiny_data):
    persona_df = ar.persona_alignment()
    question_df = ar.question_alignment()
    groups = {g["grouping"]: g for g in ar.build_groupings(persona_df, question_df)}

    allp = groups["all_persona"]
    assert allp["sampleSize"] == {"all": 2}
    assert allp["avgAlignment"]["all"] == pytest.approx((0.875 + 0.5) / 2)
    assert allp["retestAccuracy"]["all"] == pytest.approx((0.625 + 1.0) / 2)
    # rank is ascending by score
    assert [e["name"] for e in allp["rank"]] == ["2", "1"]

    gender = groups["gender"]
    assert gender["avgAlignment"] == pytest.approx({"Male": 0.875, "Female": 0.5})
    assert gender["sampleSize"] == {"Female": 1, "Male": 1}

    # a demographic shared by everyone is the plain mean
    assert groups["age"]["avgAlignment"]["18-29"] == pytest.approx(0.6875)

    # block A = mean(q1 .75, q2 1), block B = q3 .5
    block = groups["block"]
    assert block["avgAlignment"] == pytest.approx({"A": 0.875, "B": 0.5})
    assert block["sampleSize"] == {"A": 2, "B": 1}
    assert groups["category"]["avgAlignment"]["Pricing Preferences"] == pytest.approx(0.5)


def test_main_writes_three_tables(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "PERSONA_ALIGNMENT", tmp_path / "pa.parquet")
    monkeypatch.setattr(ar, "QUESTION_ALIGNMENT", tmp_path / "qa.parquet")
    monkeypatch.setattr(ar, "CALCULATED_RESULT", tmp_path / "cr.parquet")
    monkeypatch.setattr(ar, "ensure_dirs", lambda: None)
    ar.main()
    assert list(ar.persona_alignment().columns) == ["pid", "alignment", "n_question", "ceiling"]
    for name in ("pa", "qa", "cr"):
        assert (tmp_path / f"{name}.parquet").exists()
