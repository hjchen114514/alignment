"""Per-answer alignment and diagram output (drawDiagram.py)."""
import pytest

import alignmentResult as ar
import drawDiagram as dd


def test_paired_alignment(tiny_data):
    df = dd.paired_alignment().set_index(["pid", "qid"])
    # 1 - |synth - human| per answer; pid 1's pending q4 is left out
    assert df.loc[(1, "q1"), "alignment"] == pytest.approx(0.5)
    assert df.loc[(1, "q2"), "alignment"] == pytest.approx(1.0)
    assert df.loc[(2, "q3"), "alignment"] == pytest.approx(0.0)
    assert (1, "q4") not in df.index
    assert len(df) == 6


def test_main_draws_every_diagram(tiny_data, tmp_path, monkeypatch):
    numbers = tmp_path / "numbers"
    numbers.mkdir()
    for name in ("PERSONA_ALIGNMENT", "QUESTION_ALIGNMENT", "CALCULATED_RESULT"):
        monkeypatch.setattr(ar, name, numbers / getattr(ar, name).name)
        monkeypatch.setattr(dd, name, numbers / getattr(ar, name).name)
    monkeypatch.setattr(ar, "ensure_dirs", lambda: None)
    monkeypatch.setattr(dd, "ensure_dirs", lambda: None)
    diagrams = tmp_path / "diagram"
    monkeypatch.setattr(dd, "DIAGRAM_DIR", diagrams)
    monkeypatch.setattr(dd, "PERSONA_DIR", diagrams / "persona")
    monkeypatch.setattr(dd, "QUESTION_DIR", diagrams / "question")
    monkeypatch.setattr(dd, "MISC_DIR", diagrams / "miscellaneous")

    ar.main()
    dd.main()

    assert sorted(p.name for p in (diagrams / "persona").iterdir()) == \
        ["persona_1.png", "persona_2.png"]
    assert sorted(p.name for p in (diagrams / "question").iterdir()) == \
        ["question_q1_0.png", "question_q2_0.png", "question_q3_0.png"]
    misc = {p.name for p in (diagrams / "miscellaneous").iterdir()}
    assert {"rank_allPersona.png", "rank_allQuestion.png", "rank_race.png",
            "rank_income.png", "rank_block.png", "rank_category.png"} <= misc
