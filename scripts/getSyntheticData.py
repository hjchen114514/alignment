"""Generate synthetic persona responses by prompting a model with each persona's ICL text.

One session per (persona, block). Results append to a JSONL as they arrive so a crash
mid-run loses nothing, then convert once to parquet at the end.
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CFG, HUMAN_PARQUET, PERSONA_PARQUET, QUESTIONS_PARQUET,
    SYNTHETIC_JSONL, SYNTHETIC_PARQUET, ensure_dirs,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptors import aiGateway  # noqa: E402

_write_lock = threading.Lock()


def load_blocks_for_persona(pid: int) -> dict[str, list[dict]]:
    """Every question this persona was actually asked, grouped by block.

    Three-way join: human_responses says which questions this pid saw,
    questions supplies the text and options.
    """
    rows = duckdb.sql(f"""
        SELECT q.block, q.qid, q.row_id, q.question_type,
               q.questiontext, q.options, q.range_min, q.range_max
        FROM '{HUMAN_PARQUET}' h
        JOIN '{QUESTIONS_PARQUET}' q USING (qid, row_id)
        WHERE h.pid = {int(pid)}
        ORDER BY q.block, q.qid, q.row_id
    """).fetchall()

    blocks: dict[str, list[dict]] = {}
    for block, qid, row_id, qtype, text, options, lo, hi in rows:
        blocks.setdefault(block, []).append({
            "qid": qid, "row_id": int(row_id), "question_type": qtype,
            "questiontext": text, "options": list(options) if options is not None else [],
            "range_min": float(lo), "range_max": float(hi),
        })
    return blocks


def build_record(pid: int, q: dict, answer, status: str) -> dict:
    """One synthetic_responses row. answer is None when the call failed."""
    if answer is None:
        return {"pid": int(pid), "qid": q["qid"], "row_id": q["row_id"],
                "answer": None, "answertext": None, "normalized": None,
                "status": status}

    answer = int(answer)
    lo, hi = q["range_min"], q["range_max"]
    answer = min(max(answer, int(lo)), int(hi))          # clamp to the valid range
    options = q["options"]
    if q["question_type"] in ("Matrix", "MC") and options:
        answertext = options[answer - 1]
    else:
        answertext = str(answer)
    normalized = 0.0 if hi == lo else (answer - lo) / (hi - lo)
    return {"pid": int(pid), "qid": q["qid"], "row_id": q["row_id"],
            "answer": answer, "answertext": answertext,
            "normalized": normalized, "status": status}


def append_jsonl(records: list[dict]) -> None:
    with _write_lock:
        with open(SYNTHETIC_JSONL, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


def already_done() -> set[tuple[int, str]]:
    """(pid, qid) pairs already generated, so a rerun resumes instead of repeating."""
    if not SYNTHETIC_JSONL.exists():
        return set()
    done = set()
    with open(SYNTHETIC_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "finished":
                done.add((int(r["pid"]), r["qid"]))
    return done


def askQuestion(icl: str, pid: int, done: set[tuple[int, str]]) -> tuple[int, int]:
    """Ask every block this persona saw. Returns (n_finished, n_pending)."""
    blocks = load_blocks_for_persona(pid)
    n_ok = n_bad = 0

    for block, questions in blocks.items():
        pending = [q for q in questions if (int(pid), q["qid"]) not in done]
        if not pending:
            continue

        try:
            answers = aiGateway.ask_ai(icl, pending)
            if len(answers) != len(pending):
                raise ValueError(
                    f"model returned {len(answers)} answers for {len(pending)} questions"
                )
            records = [build_record(pid, q, a, "finished")
                       for q, a in zip(pending, answers)]
            n_ok += len(records)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  pid {pid} block {block!r} failed: {exc}")
            records = [build_record(pid, q, None, "pending") for q in pending]
            n_bad += len(records)

        append_jsonl(records)

    return n_ok, n_bad


def jsonl_to_parquet() -> None:
    """Collapse the JSONL to one row per (pid, qid, row_id), newest wins."""
    df = pd.read_json(SYNTHETIC_JSONL, lines=True)
    df = (df.sort_values("status", ascending=False)       # 'pending' < 'finished'
            .drop_duplicates(subset=["pid", "qid", "row_id"], keep="first")
            .sort_values(["pid", "qid", "row_id"]))
    df["answer"] = df["answer"].astype("Int64")
    df.to_parquet(SYNTHETIC_PARQUET, index=False)
    print(f"\nwrote {SYNTHETIC_PARQUET.name}  {len(df):,} rows "
          f"({df.pid.nunique()} personas)")
    print("status:", df.status.value_counts().to_dict())


def main():
    ensure_dirs()
    personas = duckdb.sql(
        f"SELECT pid, icl FROM '{PERSONA_PARQUET}' ORDER BY pid"
    ).fetchall()
    done = already_done()
    if done:
        print(f"resuming: {len(done)} (pid, qid) pairs already finished")

    total_ok = total_bad = 0
    with ThreadPoolExecutor(max_workers=CFG["concurrency"]) as pool:
        futures = {
            pool.submit(askQuestion, icl, int(pid), done): int(pid)
            for pid, icl in personas
        }
        for i, fut in enumerate(as_completed(futures), 1):
            pid = futures[fut]
            try:
                ok, bad = fut.result()
            except Exception as exc:                              # noqa: BLE001
                print(f"  pid {pid} crashed: {exc}")
                continue
            total_ok += ok
            total_bad += bad
            if i % 10 == 0:
                print(f"  {i}/{len(personas)} personas done "
                      f"({total_ok} answers, {total_bad} pending)")

    print(f"\ngenerated {total_ok} answers, {total_bad} pending")
    jsonl_to_parquet()


if __name__ == "__main__":
    main()
