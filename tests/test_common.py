"""Terminal-output summaries (common.record_summary)."""
import pytest

import common


def test_record_summary_writes_printed_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(common, "SUMMARY_DIR", tmp_path)
    with common.record_summary("demo"):
        print("hello from the script")
    assert "hello from the script" in capsys.readouterr().out      # still shown in terminal
    (path,) = tmp_path.glob("*_demo_summary.md")
    assert "hello from the script" in path.read_text()


def test_record_summary_keeps_the_traceback(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "SUMMARY_DIR", tmp_path)
    with pytest.raises(ValueError):
        with common.record_summary("crash"):
            print("before")
            raise ValueError("boom")
    text = next(tmp_path.glob("*_crash_summary.md")).read_text()
    assert "before" in text and "ValueError: boom" in text
