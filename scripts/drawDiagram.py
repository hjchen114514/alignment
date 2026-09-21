"""Draw the alignment diagrams.

    result/diagram/persona/persona_<pid>.png        MAE per question, one per persona
    result/diagram/question/question_<qid>_<row>.png MAE per persona, one per question
    result/diagram/miscellaneous/rank_*.png          ranked bar charts per grouping
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
    DIAGRAM_DIR, HUMAN_PARQUET, NUMBERS_DIR, QUESTIONS_PARQUET,
    SYNTHETIC_PARQUET, ensure_dirs,
)

PERSONA_DIR = DIAGRAM_DIR / "persona"
QUESTION_DIR = DIAGRAM_DIR / "question"
MISC_DIR = DIAGRAM_DIR / "miscellaneous"

BAR = "#4C72B0"
LINE = "#C44E52"


def paired_errors() -> pd.DataFrame:
    """Absolute error for every (persona, question) pair that has both answers."""
    return duckdb.sql(f"""
        SELECT s.pid, s.qid, s.row_id, q.block, q.category,
               ABS(s.normalized - h.normalized) AS mae
        FROM '{SYNTHETIC_PARQUET}' s
        JOIN '{HUMAN_PARQUET}'     h USING (pid, qid, row_id)
        JOIN '{QUESTIONS_PARQUET}' q USING (qid, row_id)
        WHERE s.normalized IS NOT NULL
        ORDER BY s.pid, s.qid, s.row_id
    """).df()


def draw_persona(pid: int, rows: pd.DataFrame, reference: float) -> None:
    labels = rows.qid + "_" + rows.row_id.astype(str)
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.16), 4))
    ax.bar(range(len(rows)), rows.mae, color=BAR, width=0.8)
    ax.axhline(reference, ls="--", color=LINE, lw=1.4,
               label=f"mean MAE, all personas ({reference:.3f})")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("question")
    ax.set_ylabel("MAE")
    ax.set_title(f"Persona {pid} - MAE per question (n={len(rows)} questions)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PERSONA_DIR / f"persona_{pid}.png", dpi=130)
    plt.close(fig)


def draw_question(qid: str, row_id: int, rows: pd.DataFrame,
                  reference: float) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.16), 4))
    ax.bar(range(len(rows)), rows.mae, color=BAR, width=0.8)
    ax.axhline(reference, ls="--", color=LINE, lw=1.4,
               label=f"mean MAE, all questions ({reference:.3f})")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(rows.pid.astype(str), rotation=90, fontsize=4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("persona")
    ax.set_ylabel("MAE")
    ax.set_title(f"{qid} row {row_id} - MAE per persona (n={len(rows)} personas)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(QUESTION_DIR / f"question_{qid}_{row_id}.png", dpi=130)
    plt.close(fig)


def draw_rank(grouping: str, entries: list[dict], sizes: dict,
              out_dir: Path, value_label: str = "alignment") -> None:
    """Ranked bar chart, lowest score on the left."""
    if not entries:
        return
    names = [f"{e['name']}\nn={sizes.get(e['name'], '')}" for e in entries]
    scores = [e["score"] for e in entries]
    width = max(6, len(entries) * 0.5)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.bar(range(len(entries)), scores, color=BAR)
    ax.set_xticks(range(len(entries)))
    fontsize = 7 if len(entries) > 20 else 8
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=fontsize)
    ax.set_ylabel(value_label)
    ax.set_title(f"{value_label} by {grouping} (ranked ascending)")
    fig.tight_layout()
    fig.savefig(out_dir / f"rank_{grouping}.png", dpi=130)
    plt.close(fig)


def draw_persona_ranking(persona_df: pd.DataFrame) -> None:
    """All personas on one chart, with ceiling and shuffle floor for reference."""
    ranked = persona_df.sort_values("alignment")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(range(len(ranked)), ranked.alignment, color=BAR, width=0.9)
    ax.axhline(ranked.ceiling.mean(), ls="--", color="#55A868", lw=1.5,
               label=f"ceiling ({ranked.ceiling.mean():.1%})")
    ax.axhline(ranked.shuffle_floor.mean(), ls="--", color=LINE, lw=1.5,
               label=f"shuffle floor ({ranked.shuffle_floor.mean():.1%})")
    ax.set_xticks(range(len(ranked)))
    ax.set_xticklabels(ranked.pid.astype(str), rotation=90, fontsize=4)
    ax.set_xlabel("persona")
    ax.set_ylabel("alignment")
    ax.set_title(f"Alignment by persona, ranked (n={len(ranked)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MISC_DIR / "rank_allPersona.png", dpi=130)
    plt.close(fig)


def main():
    ensure_dirs()
    for d in (PERSONA_DIR, QUESTION_DIR, MISC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    errors = paired_errors() #get the MAE scores of each personas
    if errors.empty:
        raise SystemExit("no paired responses - run getSyntheticData.py first")

    persona_mean = errors.groupby("pid").mae.mean().mean()
    question_mean = errors.groupby(["qid", "row_id"]).mae.mean().mean()

    print(f"drawing {errors.pid.nunique()} persona diagrams ...")
    for pid, rows in errors.groupby("pid"):
        draw_persona(int(pid), rows.reset_index(drop=True), persona_mean)

    groups = list(errors.groupby(["qid", "row_id"]))
    print(f"drawing {len(groups)} question diagrams ...")
    for (qid, row_id), rows in groups:
        draw_question(qid, int(row_id), rows.reset_index(drop=True), question_mean)

    print("drawing ranked summaries ...")
    calc = duckdb.sql(
        f"SELECT * FROM '{NUMBERS_DIR / 'calculatedResult.parquet'}'"
    ).df()
    for row in calc.itertuples():
        if row.grouping in ("all_persona", "all_question"):
            continue
        entries = [dict(e) for e in row.rank]
        draw_rank(row.grouping, entries, dict(row.sampleSize), MISC_DIR)

    persona_df = duckdb.sql(
        f"SELECT * FROM '{NUMBERS_DIR / 'personaAlignment.parquet'}'"
    ).df()
    draw_persona_ranking(persona_df)

    n_png = len(list(DIAGRAM_DIR.rglob("*.png")))
    print(f"\nwrote {n_png} diagrams under {DIAGRAM_DIR}")


if __name__ == "__main__":
    main()
