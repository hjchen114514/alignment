"""Compute alignment scores and write the result tables.

Alignment uses the paper's two-stage average: mean absolute error within each
block first, then the mean across blocks. That keeps the 40-question pricing
block from dominating the 15 single-question blocks.

Writes:
    result/numbers/personaAlignment.parquet
    result/numbers/questionAlignment.parquet
    result/numbers/calculatedResult.parquet
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    HUMAN_PARQUET, HUMAN_RETEST_PARQUET, NUMBERS_DIR, PERSONA_PARQUET,
    QUESTIONS_PARQUET, SYNTHETIC_PARQUET, ensure_dirs,
)

DEMOGRAPHICS = ["race", "gender", "age", "education", "income",
                "favoredPoliticalParty", "politicalViews"]

PERSONA_ALIGNMENT = NUMBERS_DIR / "personaAlignment.parquet"
QUESTION_ALIGNMENT = NUMBERS_DIR / "questionAlignment.parquet"
CALCULATED_RESULT = NUMBERS_DIR / "calculatedResult.parquet"


def sql(query: str) -> pd.DataFrame:
    return duckdb.sql(query).df()


# --------------------------------------------------------------------------
# persona level
# --------------------------------------------------------------------------

def persona_alignment() -> pd.DataFrame:
    """Per-persona alignment, ceiling and shuffle floor, two-stage averaged."""
    return sql(f"""
        WITH answer_err AS (
            SELECT s.pid, q.block, ABS(s.normalized - h.normalized) AS err
            FROM '{SYNTHETIC_PARQUET}' s
            JOIN '{HUMAN_PARQUET}'    h USING (pid, qid, row_id)
            JOIN '{QUESTIONS_PARQUET}' q USING (qid, row_id)
            WHERE s.normalized IS NOT NULL
        ),
        block_err AS (
            SELECT pid, block, AVG(err) AS e FROM answer_err GROUP BY pid, block
        ),
        alignment AS (
            SELECT pid, 1 - AVG(e) AS alignment FROM block_err GROUP BY pid
        ),
        counts AS (
            SELECT pid, COUNT(*) AS n_question FROM answer_err GROUP BY pid
        ),

        -- ceiling: the same human answering the same items a second time
        retest_err AS (
            SELECT h.pid, q.block, ABS(h.normalized - r.normalized) AS err
            FROM '{HUMAN_PARQUET}'        h
            JOIN '{HUMAN_RETEST_PARQUET}' r USING (pid, qid, row_id)
            JOIN '{QUESTIONS_PARQUET}'    q USING (qid, row_id)
        ),
        retest_block AS (
            SELECT pid, block, AVG(err) AS e FROM retest_err GROUP BY pid, block
        ),
        ceiling AS (
            SELECT pid, 1 - AVG(e) AS ceiling FROM retest_block GROUP BY pid
        ),

        -- shuffle floor: persona A's synthetic answers scored against every
        -- OTHER human. If this matches the real alignment, the ICL text taught
        -- the model nothing person-specific.
        cross_err AS (
            SELECT s.pid AS synth_pid, h.pid AS human_pid, q.block,
                   AVG(ABS(s.normalized - h.normalized)) AS e
            FROM '{SYNTHETIC_PARQUET}' s
            JOIN '{HUMAN_PARQUET}'     h USING (qid, row_id)
            JOIN '{QUESTIONS_PARQUET}' q USING (qid, row_id)
            WHERE s.pid <> h.pid AND s.normalized IS NOT NULL
            GROUP BY s.pid, h.pid, q.block
        ),
        cross_pair AS (
            SELECT synth_pid, human_pid, AVG(e) AS pair_err
            FROM cross_err GROUP BY synth_pid, human_pid
        ),
        shuffle AS (
            SELECT synth_pid AS pid, 1 - AVG(pair_err) AS shuffle_floor
            FROM cross_pair GROUP BY synth_pid
        )

        SELECT a.pid, a.alignment, c.n_question, ce.ceiling, sh.shuffle_floor
        FROM alignment a
        JOIN counts   c  USING (pid)
        JOIN ceiling  ce USING (pid)
        JOIN shuffle  sh USING (pid)
        ORDER BY a.pid
    """)


# --------------------------------------------------------------------------
# question level
# --------------------------------------------------------------------------

def question_alignment() -> pd.DataFrame:
    """Per-question alignment, plus the human/synthetic answer spread."""
    return sql(f"""
        WITH paired AS (
            SELECT s.qid, s.row_id,
                   s.normalized AS synth_norm,
                   h.normalized AS human_norm
            FROM '{SYNTHETIC_PARQUET}' s
            JOIN '{HUMAN_PARQUET}'     h USING (pid, qid, row_id)
            WHERE s.normalized IS NOT NULL
        ),
        scored AS (
            SELECT qid, row_id,
                   1 - AVG(ABS(synth_norm - human_norm)) AS alignment,
                   COUNT(*)            AS n_persona,
                   STDDEV_SAMP(synth_norm) AS synthetic_std,
                   STDDEV_SAMP(human_norm) AS human_std
            FROM paired GROUP BY qid, row_id
        ),
        retest AS (
            SELECT h.qid, h.row_id, 1 - AVG(ABS(h.normalized - r.normalized)) AS ceiling
            FROM '{HUMAN_PARQUET}'        h
            JOIN '{HUMAN_RETEST_PARQUET}' r USING (pid, qid, row_id)
            GROUP BY h.qid, h.row_id
        )
        SELECT sc.qid, sc.row_id, q.block, q.category,
               sc.alignment, sc.n_persona, rt.ceiling,
               sc.synthetic_std, sc.human_std
        FROM scored sc
        JOIN '{QUESTIONS_PARQUET}' q USING (qid, row_id)
        LEFT JOIN retest rt USING (qid, row_id)
        ORDER BY sc.qid, sc.row_id
    """)


# --------------------------------------------------------------------------
# grouped aggregates
# --------------------------------------------------------------------------

def build_groupings(persona_df: pd.DataFrame,
                    question_df: pd.DataFrame) -> list[dict]:
    """One entry per grouping, each carrying its own dicts of per-value stats."""
    personas = sql(f"SELECT * EXCLUDE (icl) FROM '{PERSONA_PARQUET}'")
    merged = persona_df.merge(personas, on="pid")
    out = []

    def rank(d: dict) -> list[dict]:
        return [{"name": str(k), "score": float(v)}
                for k, v in sorted(d.items(), key=lambda kv: kv[1])]

    # every persona pooled
    out.append({
        "grouping": "all_persona",
        "sampleSize": {"all": len(persona_df)},
        "retestAccuracy": {"all": float(persona_df.ceiling.mean())},
        "avgAlignment": {"all": float(persona_df.alignment.mean())},
        "shuffleFloor": {"all": float(persona_df.shuffle_floor.mean())},
        "rank": rank(dict(zip(persona_df.pid.astype(str), persona_df.alignment))),
    })

    # one entry per demographic field
    for field in DEMOGRAPHICS:
        g = merged.groupby(field)
        align = g.alignment.mean().to_dict()
        out.append({
            "grouping": field,
            "sampleSize": {str(k): int(v) for k, v in g.size().to_dict().items()},
            "retestAccuracy": {str(k): float(v) for k, v in g.ceiling.mean().to_dict().items()},
            "avgAlignment": {str(k): float(v) for k, v in align.items()},
            "shuffleFloor": {str(k): float(v) for k, v in g.shuffle_floor.mean().to_dict().items()},
            "rank": rank(align),
        })

    # every question pooled
    qkey = question_df.qid + "_" + question_df.row_id.astype(str)
    out.append({
        "grouping": "all_question",
        "sampleSize": {"all": len(question_df)},
        "retestAccuracy": {"all": float(question_df.ceiling.mean())},
        "avgAlignment": {"all": float(question_df.alignment.mean())},
        "shuffleFloor": {},
        "rank": rank(dict(zip(qkey, question_df.alignment))),
    })

    # questions grouped by block, then by category
    for field in ("block", "category"):
        g = question_df.groupby(field)
        align = g.alignment.mean().to_dict()
        out.append({
            "grouping": field,
            "sampleSize": {str(k): int(v) for k, v in g.size().to_dict().items()},
            "retestAccuracy": {str(k): float(v) for k, v in g.ceiling.mean().to_dict().items()},
            "avgAlignment": {str(k): float(v) for k, v in align.items()},
            "shuffleFloor": {},
            "rank": rank(align),
        })

    return out


CALCULATED_SCHEMA = pa.schema([
    ("grouping", pa.string()),
    ("sampleSize", pa.map_(pa.string(), pa.int64())),
    ("retestAccuracy", pa.map_(pa.string(), pa.float64())),
    ("avgAlignment", pa.map_(pa.string(), pa.float64())),
    ("shuffleFloor", pa.map_(pa.string(), pa.float64())),
    ("rank", pa.list_(pa.struct([("name", pa.string()), ("score", pa.float64())]))),
])


def write_calculated(groupings: list[dict], schema: pa.Schema, path: Path) -> None:
    table = pa.Table.from_pylist(groupings, schema=schema)
    pq.write_table(table, path)


# --------------------------------------------------------------------------

def report(persona_df: pd.DataFrame, question_df: pd.DataFrame,
           groupings: list[dict]) -> None:
    align = persona_df.alignment.mean()
    ceiling = persona_df.ceiling.mean()
    floor = persona_df.shuffle_floor.mean()

    print("\n" + "=" * 62)
    print("OVERALL")
    print("=" * 62)
    print(f"  ceiling  (human vs himself)   {ceiling:>8.2%}")
    print(f"  alignment(synthetic vs human) {align:>8.2%}")
    print(f"  shuffle floor (vs strangers)  {floor:>8.2%}")
    print(f"  lift over shuffle             {align - floor:>+8.2%}")
    print(f"  pct of ceiling                {align / ceiling:>8.2%}")
    print(f"  personas={len(persona_df)}  questions={len(question_df)}")

    best = persona_df.nlargest(3, "alignment")
    worst = persona_df.nsmallest(3, "alignment")
    print("\nBEST personas :", ", ".join(
        f"pid {int(r.pid)} {r.alignment:.1%}" for r in best.itertuples()))
    print("WORST personas:", ", ".join(
        f"pid {int(r.pid)} {r.alignment:.1%}" for r in worst.itertuples()))

    qb = question_df.nlargest(3, "alignment")
    qw = question_df.nsmallest(3, "alignment")
    print("\nBEST questions :", ", ".join(
        f"{r.qid}_{r.row_id} {r.alignment:.1%}" for r in qb.itertuples()))
    print("WORST questions:", ", ".join(
        f"{r.qid}_{r.row_id} {r.alignment:.1%}" for r in qw.itertuples()))

    # where the model collapsed to one answer while humans disagreed
    collapse = question_df.assign(gap=question_df.human_std - question_df.synthetic_std)
    top = collapse.nlargest(3, "gap")
    print("\nLARGEST spread gap (model less varied than humans):")
    for r in top.itertuples():
        print(f"  {r.qid}_{r.row_id:<3} human std {r.human_std:.3f} "
              f"vs synthetic {r.synthetic_std:.3f}  ({r.block})")

    for g in groupings:
        if g["grouping"] in ("all_persona", "all_question"):
            continue
        print(f"\n{g['grouping'].upper()}")
        for entry in reversed(g["rank"]):
            n = g["sampleSize"].get(entry["name"], 0)
            print(f"  {entry['score']:.2%}  n={n:<4} {entry['name']}")


def main():
    ensure_dirs()
    for required in (SYNTHETIC_PARQUET, HUMAN_PARQUET, HUMAN_RETEST_PARQUET):
        if not required.exists():
            raise SystemExit(
                f"missing {required} - run importData.py and getSyntheticData.py first"
            )

    persona_df = persona_alignment()
    question_df = question_alignment()
    groupings = build_groupings(persona_df, question_df)

    persona_df.to_parquet(PERSONA_ALIGNMENT, index=False)
    question_df.to_parquet(QUESTION_ALIGNMENT, index=False)
    write_calculated(groupings, CALCULATED_SCHEMA, CALCULATED_RESULT)

    print(f"wrote {PERSONA_ALIGNMENT.name}   {len(persona_df):,} rows")
    print(f"wrote {QUESTION_ALIGNMENT.name}  {len(question_df):,} rows")
    print(f"wrote {CALCULATED_RESULT.name}   {len(groupings)} groupings")
    report(persona_df, question_df, groupings)


if __name__ == "__main__":
    main()
