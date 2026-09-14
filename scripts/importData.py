"""Build the four source tables from the Twin-2K-500 wave_split config.

Writes:
    data/human_responses.parquet   one row per (pid, qid, row_id)
    data/questions.parquet         one row per (qid, row_id)
    data/persona.parquet           one row per pid, incl. ICL text
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CFG, DATA_DIR, HUMAN_PARQUET, HUMAN_RETEST_PARQUET, PERSONA_PARQUET,
    QUESTIONS_PARQUET, categorize, ensure_dirs, is_skipped_block, is_skipped_qid,
)

TE_RANGES = {k: tuple(v) for k, v in (CFG.get("te_ranges") or {}).items()}


def to_int(value):
    """Answers arrive as int, '50', or 50.0 - int('50.0') raises, so go via float."""
    return int(float(value))


def parse_question(q: dict, block_name: str):
    """Yield one dict per answerable row of a single question.

    Matrix -> one row per Rows entry, Slider -> one per Statements entry,
    MC / TE -> exactly one row.
    """
    qid = q["QuestionID"]
    qtype = q["QuestionType"]
    stem = q.get("QuestionText", "")
    answers = q.get("Answers", {}) or {}

    if qtype == "Matrix":
        options = q.get("Columns") or []
        selected = answers.get("SelectedByPosition") or []
        texts = answers.get("SelectedText") or []
        for i, row_label in enumerate(q.get("Rows") or []):
            if i >= len(selected) or selected[i] is None:
                continue
            answer = to_int(selected[i])
            yield {
                "qid": qid, "row_id": i, "question_type": qtype,
                "questiontext": f"{stem} {row_label}".strip(),
                "options": list(options),
                "range_min": 1.0, "range_max": float(len(options)),
                "answer": answer,
                "answertext": texts[i] if i < len(texts) else options[answer - 1],
            }

    elif qtype == "MC":
        options = q.get("Options") or []
        selected = answers.get("SelectedByPosition")
        if selected is None:
            return
        answer = to_int(selected)
        yield {
            "qid": qid, "row_id": 0, "question_type": qtype,
            "questiontext": stem,
            "options": list(options),
            "range_min": 1.0, "range_max": float(len(options)),
            "answer": answer,
            "answertext": answers.get("SelectedText") or options[answer - 1],
        }

    elif qtype == "Slider":
        rng = q.get("Range") or {}
        lo, hi = float(rng.get("Min", 0)), float(rng.get("Max", 100))
        for i, value in enumerate(answers.get("Values") or []):
            if value is None:
                continue
            answer = to_int(value)
            yield {
                "qid": qid, "row_id": i, "question_type": qtype,
                "questiontext": stem,
                "options": [],
                "range_min": lo, "range_max": hi,
                "answer": answer, "answertext": str(answer),
            }

    elif qtype == "TE":
        text = answers.get("Text")
        if text is None or str(text).strip() == "":
            return
        if qid not in TE_RANGES:
            raise ValueError(
                f"{qid} is a text-entry question with no Range in the source and no "
                f"te_ranges entry in config.yaml - cannot normalize."
            )
        lo, hi = TE_RANGES[qid]
        answer = to_int(text)
        yield {
            "qid": qid, "row_id": 0, "question_type": qtype,
            "questiontext": stem,
            "options": [],
            "range_min": float(lo), "range_max": float(hi),
            "answer": answer, "answertext": str(answer),
        }


def normalize(answer: int, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return (answer - lo) / (hi - lo)


def main():
    ensure_dirs()
    from datasets import load_dataset

    n = CFG["n_personas"]
    print(f"loading {CFG['dataset']} ({CFG['dataset_config']}) ...")
    ds = load_dataset(CFG["dataset"], CFG["dataset_config"])["data"]
    print(f"  {len(ds)} personas available, taking first {n}")

    human_rows, retest_rows, persona_rows = [], [], []
    questions: dict[tuple[str, int], dict] = {}
    dq = CFG["demographic_qids"]
    wanted_demo = {qid: field for field, qid in dq.items()}
    clipped = 0

    for index in range(n):
        row = ds[index]
        pid = int(row["pid"])

        # --- responses, from both administrations of the wave-4 questions ---
        # wave4_Q_wave1_3_A is the benchmark; wave4_Q_wave4_A is the same person
        # answering the same items again, which gives the test-retest ceiling.
        for source_field, target in (("wave4_Q_wave1_3_A", human_rows),
                                     ("wave4_Q_wave4_A", retest_rows)):
            record_metadata = source_field == "wave4_Q_wave1_3_A"
            for element in json.loads(row[source_field]):
                block = (element.get("BlockName") or "").strip()
                if is_skipped_block(block):
                    continue
                category = categorize(block)
                for q in element.get("Questions", []):
                    if q.get("is_descriptive") or q.get("QuestionType") == "DB":
                        continue
                    if is_skipped_qid(q.get("QuestionID", "")):
                        continue
                    for rec in parse_question(q, block):
                        key = (rec["qid"], rec["row_id"])
                        if record_metadata and key not in questions:
                            questions[key] = {
                                "qid": rec["qid"], "row_id": rec["row_id"],
                                "category": category, "block": block,
                                "question_type": rec["question_type"],
                                "questiontext": rec["questiontext"],
                                "options": rec["options"],
                                "range_min": rec["range_min"],
                                "range_max": rec["range_max"],
                            }
                        norm = normalize(rec["answer"], rec["range_min"], rec["range_max"])
                        if not (0.0 <= norm <= 1.0):
                            clipped += 1
                            norm = min(max(norm, 0.0), 1.0)
                        target.append({
                            "pid": pid, "qid": rec["qid"], "row_id": rec["row_id"],
                            "answer": rec["answer"],
                            "answertext": str(rec["answertext"]),
                            "normalized": norm,
                        })

        # --- demographics (waves 1-3) + ICL text ---
        persona = {"pid": pid}
        for element in json.loads(row["wave1_3_persona_json"]):
            for q in element.get("Questions", []):
                field = wanted_demo.get(q.get("QuestionID"))
                if not field:
                    continue
                a = q.get("Answers", {}) or {}
                persona[field] = a.get("SelectedText") or a.get("Text")
        missing = [f for f in dq if f not in persona]
        if missing:
            raise ValueError(f"pid {pid}: demographics not found: {missing}")
        persona["icl"] = row["wave1_3_persona_text"]
        persona_rows.append(persona)

        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{n} personas parsed")

    human = pd.DataFrame(human_rows)
    retest = pd.DataFrame(retest_rows)
    qdf = pd.DataFrame(list(questions.values()))
    pdf = pd.DataFrame(persona_rows)[["pid", *dq.keys(), "icl"]]

    human.to_parquet(HUMAN_PARQUET, index=False)
    retest.to_parquet(HUMAN_RETEST_PARQUET, index=False)
    qdf.to_parquet(QUESTIONS_PARQUET, index=False)
    pdf.to_parquet(PERSONA_PARQUET, index=False)

    print(f"\nwrote {HUMAN_PARQUET.relative_to(DATA_DIR.parent)}  "
          f"{len(human):,} rows ({human.pid.nunique()} personas)")
    print(f"wrote {HUMAN_RETEST_PARQUET.relative_to(DATA_DIR.parent)}  "
          f"{len(retest):,} rows (test-retest ceiling source)")
    print(f"wrote {QUESTIONS_PARQUET.relative_to(DATA_DIR.parent)}  {len(qdf):,} rows "
          f"({qdf.qid.nunique()} distinct qids, {qdf.block.nunique()} blocks)")
    print(f"wrote {PERSONA_PARQUET.relative_to(DATA_DIR.parent)}  {len(pdf):,} rows")
    if clipped:
        print(f"WARNING: {clipped} normalized values fell outside [0,1] and were clipped")

    per_persona = human.groupby("pid").size()
    print(f"\nanswers per persona: min={per_persona.min()} "
          f"median={int(per_persona.median())} max={per_persona.max()}")
    print("question types:", qdf.question_type.value_counts().to_dict())
    print("categories:", qdf.category.value_counts().to_dict())


if __name__ == "__main__":
    main()
