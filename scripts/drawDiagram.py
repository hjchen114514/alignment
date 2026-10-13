"""Draw the alignment diagrams. Run after alignmentResult.py.

    result/diagram/persona/persona_<pid>.png          alignment per question, one per persona
    result/diagram/question/question_<qid>_<row>.png  alignment per persona, one per question
    result/diagram/miscellaneous/rank_<grouping>.png  ranked bar chart per grouping

Bars come from the paired responses (1 - |synthetic - human| per answer). The
dashed reference lines and the rank charts are read from the tables
alignmentResult.py wrote.
"""
import sys
from pathlib import Path

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DIAGRAM_DIR, HUMAN_PARQUET, NUMBERS_DIR, SYNTHETIC_PARQUET, ensure_dirs,
    record_summary,
)

PERSONA_DIR = DIAGRAM_DIR / "persona"
QUESTION_DIR = DIAGRAM_DIR / "question"
MISC_DIR = DIAGRAM_DIR / "miscellaneous"

PERSONA_ALIGNMENT = NUMBERS_DIR / "personaAlignment.parquet"
QUESTION_ALIGNMENT = NUMBERS_DIR / "questionAlignment.parquet"
CALCULATED_RESULT = NUMBERS_DIR / "calculatedResult.parquet"

BAR = "#4C72B0"
AVG_LINE = "#C44E52"       # dashed line 1: average alignment
CEILING_LINE = "#55A868"   # dashed line 2: retest ceiling
OWN_LINE = "#E1B400"       # dashed line 3: this persona's / question's own alignment

# grouping name in calculatedResult -> diagram file name
FILE_NAMES = {"all_persona": "allPersona", "all_question": "allQuestion"}


def paired_alignment() -> pd.DataFrame:
    """Alignment of every answer that has both a human and a synthetic value."""
    return duckdb.sql(f"""
        SELECT s.pid, s.qid, s.row_id,
               1 - ABS(s.normalized - h.normalized) AS alignment
        FROM '{SYNTHETIC_PARQUET}' s
        JOIN '{HUMAN_PARQUET}'     h USING (pid, qid, row_id)
        WHERE s.normalized IS NOT NULL
        ORDER BY s.pid, s.qid, s.row_id
    """).df()


def draw_bars(labels, scores, avg: float, avg_label: str, ceiling: float,
              own: float, own_label: str, xlabel: str, title: str, path: Path) -> None:
    """Bars of alignment with the three dashed reference lines."""
    fig, ax = plt.subplots(figsize=(max(8, len(scores) * 0.16), 4))
    ax.bar(range(len(scores)), scores, color=BAR, width=0.8)
    ax.axhline(avg, ls="--", color=AVG_LINE, lw=1.4, label=f"{avg_label} ({avg:.1%})")
    ax.axhline(ceiling, ls="--", color=CEILING_LINE, lw=1.4,
               label=f"retest ceiling ({ceiling:.1%})")
    ax.axhline(own, ls="--", color=OWN_LINE, lw=1.4, label=f"{own_label} ({own:.1%})")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, rotation=90, fontsize=4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("alignment")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_rank(grouping: str, entries: list[dict], sizes: dict,
              out_dir: Path, value_label: str = "alignment") -> None:
    """Ranked bar chart, lowest score on the left, n under each named bar."""
    if not entries:
        return
    names = [f"{e['name']}\nn={sizes[e['name']]}" if e["name"] in sizes else e["name"]
             for e in entries]
    scores = [e["score"] for e in entries]
    many = len(entries) > 40          # all personas / all questions: thin bars
    fig, ax = plt.subplots(figsize=(max(6, len(entries) * (0.16 if many else 0.5)), 4.5))
    ax.bar(range(len(entries)), scores, color=BAR)
    ax.set_xticks(range(len(entries)))
    if many:
        ax.set_xticklabels(names, rotation=90, fontsize=4)
    else:
        ax.set_xticklabels(names, rotation=45, ha="right",
                           fontsize=7 if len(entries) > 20 else 8)
    ax.set_ylabel(value_label)
    total = sizes.get("all", len(entries))
    ax.set_title(f"{value_label} by {grouping}, ranked ascending (n={total})")
    fig.tight_layout()
    fig.savefig(out_dir / f"rank_{FILE_NAMES.get(grouping, grouping)}.png", dpi=130)
    plt.close(fig)


def main():
    ensure_dirs()
    for d in (PERSONA_DIR, QUESTION_DIR, MISC_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not CALCULATED_RESULT.exists():
        raise SystemExit("no alignment results - run alignmentResult.py first")

    pairs = paired_alignment()
    persona_df = pd.read_parquet(PERSONA_ALIGNMENT)
    question_df = pd.read_parquet(QUESTION_ALIGNMENT)
    persona_avg = persona_df.alignment.mean()
    question_avg = question_df.alignment.mean()

    print(f"drawing {len(persona_df)} persona diagrams ...")
    for p in persona_df.itertuples():
        rows = pairs[pairs.pid == p.pid]
        draw_bars(rows.qid + "_" + rows.row_id.astype(str), rows.alignment,
                  persona_avg, "average alignment, all personas", p.ceiling,
                  p.alignment, "this persona's alignment",
                  "question", f"Persona {p.pid} - alignment per question (n={len(rows)} questions)",
                  PERSONA_DIR / f"persona_{p.pid}.png")

    print(f"drawing {len(question_df)} question diagrams ...")
    for q in question_df.itertuples():
        rows = pairs[(pairs.qid == q.qid) & (pairs.row_id == q.row_id)]
        draw_bars(rows.pid.astype(str), rows.alignment,
                  question_avg, "average alignment, all questions", q.ceiling,
                  q.alignment, "this question's alignment",
                  "persona", f"{q.qid} row {q.row_id} - alignment per persona (n={len(rows)} personas)",
                  QUESTION_DIR / f"question_{q.qid}_{q.row_id}.png")

    print("drawing ranked summaries ...")
    for g in pd.read_parquet(CALCULATED_RESULT).itertuples():
        draw_rank(g.grouping, [dict(e) for e in g.rank], dict(g.sampleSize), MISC_DIR)

    n_png = len(list(DIAGRAM_DIR.rglob("*.png")))
    print(f"\nwrote {n_png} diagrams under {DIAGRAM_DIR}")


if __name__ == "__main__":
    with record_summary("drawDiagram"):
        main()
