"""Jensen-Shannon distance between each persona's human and synthetic answer
distributions.

This is a different question from alignmentResult.py. Alignment is paired: it
asks whether the twin gave *this* answer to *this* question. JSD is unpaired:
it only compares the shape of the two answer distributions, so a twin could
score well here by giving the right answers to the wrong questions. Divergence
between the two measures is informative, not a bug.

Writes:
    result/numbers/personaDistances.parquet
    result/numbers/calculatedResult_distances.parquet
    result/diagram/distances/persona_<pid>.png
    result/diagram/distances/miscellaneous/rank_*.png
"""
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.distance import jensenshannon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CFG, DIAGRAM_DIR, HUMAN_PARQUET, NUMBERS_DIR, PERSONA_PARQUET,
    SYNTHETIC_PARQUET, ensure_dirs,
)
from drawDiagram import draw_rank  # noqa: E402

BINS = int(CFG.get("jsd_bins", 10))
DEMOGRAPHICS = ["race", "gender", "age", "education", "income",
                "favoredPoliticalParty", "politicalViews"]

DIST_DIR = DIAGRAM_DIR / "distances"
DIST_MISC_DIR = DIST_DIR / "miscellaneous"
PERSONA_DISTANCES = NUMBERS_DIR / "personaDistances.parquet"
CALCULATED_DISTANCES = NUMBERS_DIR / "calculatedResult_distances.parquet"

EDGES = np.linspace(0.0, 1.0, BINS + 1)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


def histogram(values: np.ndarray) -> np.ndarray:
    """Normalized histogram over [0,1]; sums to 1."""
    counts, _ = np.histogram(values, bins=EDGES)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2, in [0,1]."""
    return float(jensenshannon(p, q, base=2) ** 2)


def load_by_persona() -> tuple[dict, dict]:
    human = duckdb.sql(
        f"SELECT pid, normalized FROM '{HUMAN_PARQUET}'"
    ).df()
    synth = duckdb.sql(
        f"SELECT pid, normalized FROM '{SYNTHETIC_PARQUET}' WHERE normalized IS NOT NULL"
    ).df()
    h = {int(p): g.normalized.to_numpy() for p, g in human.groupby("pid")}
    s = {int(p): g.normalized.to_numpy() for p, g in synth.groupby("pid")}
    return h, s


def compute(human: dict, synth: dict) -> tuple[pd.DataFrame, dict]:
    """Per-persona JSD plus a shuffle floor against every other human."""
    human_hist = {pid: histogram(v) for pid, v in human.items()}
    synth_hist = {pid: histogram(v) for pid, v in synth.items()}

    rows = []
    for pid, sh in synth_hist.items():
        if pid not in human_hist:
            continue
        others = [jsd(sh, hh) for other, hh in human_hist.items() if other != pid]
        rows.append({
            "pid": pid,
            "JSD": jsd(sh, human_hist[pid]),
            "n_question": int(len(synth[pid])),
            "shuffle_floor": float(np.mean(others)) if others else float("nan"),
        })

    df = pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)
    return df, {"human": human_hist, "synthetic": synth_hist}


def build_groupings(dist_df: pd.DataFrame) -> list[dict]:
    personas = duckdb.sql(f"SELECT * EXCLUDE (icl) FROM '{PERSONA_PARQUET}'").df()
    merged = dist_df.merge(personas, on="pid")
    out = []

    def rank(d: dict) -> list[dict]:
        return [{"name": str(k), "score": float(v)}
                for k, v in sorted(d.items(), key=lambda kv: kv[1])]

    out.append({
        "grouping": "all_persona",
        "sampleSize": {"all": len(dist_df)},
        "avgJSD": {"all": float(dist_df.JSD.mean())},
        "shuffleFloor": {"all": float(dist_df.shuffle_floor.mean())},
        "rank": rank(dict(zip(dist_df.pid.astype(str), dist_df.JSD))),
    })

    for field in DEMOGRAPHICS:
        g = merged.groupby(field)
        means = g.JSD.mean().to_dict()
        out.append({
            "grouping": field,
            "sampleSize": {str(k): int(v) for k, v in g.size().to_dict().items()},
            "avgJSD": {str(k): float(v) for k, v in means.items()},
            "shuffleFloor": {str(k): float(v)
                             for k, v in g.shuffle_floor.mean().to_dict().items()},
            "rank": rank(means),
        })
    return out


DISTANCE_SCHEMA = pa.schema([
    ("grouping", pa.string()),
    ("sampleSize", pa.map_(pa.string(), pa.int64())),
    ("avgJSD", pa.map_(pa.string(), pa.float64())),
    ("shuffleFloor", pa.map_(pa.string(), pa.float64())),
    ("rank", pa.list_(pa.struct([("name", pa.string()), ("score", pa.float64())]))),
])


def draw_persona_distribution(pid: int, human_hist: np.ndarray,
                              synth_hist: np.ndarray, score: float) -> None:
    """Two densities over the normalized answer scale, JSD in the title."""
    width = 1.0 / BINS
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(CENTERS, human_hist / width, color="#4C72B0", lw=2,
            marker="o", ms=4, label="human")
    ax.plot(CENTERS, synth_hist / width, color="#C44E52", lw=2,
            marker="s", ms=4, label="synthetic")
    ax.fill_between(CENTERS, human_hist / width, color="#4C72B0", alpha=0.15)
    ax.fill_between(CENTERS, synth_hist / width, color="#C44E52", alpha=0.15)
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalized answer")
    ax.set_ylabel("probability density")
    ax.set_title(f"Persona {pid} - answer distribution (JSD = {score:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIST_DIR / f"persona_{pid}.png", dpi=130)
    plt.close(fig)


def report(dist_df: pd.DataFrame, groupings: list[dict]) -> None:
    print("\n" + "=" * 62)
    print("JENSEN-SHANNON DIVERGENCE  (0 = identical, lower is better)")
    print("=" * 62)
    print(f"  mean JSD                 {dist_df.JSD.mean():>8.4f}")
    print(f"  median JSD               {dist_df.JSD.median():>8.4f}")
    print(f"  shuffle floor (vs others){dist_df.shuffle_floor.mean():>8.4f}")
    print(f"  bins={BINS}  personas={len(dist_df)}")

    best = dist_df.nsmallest(3, "JSD")
    worst = dist_df.nlargest(3, "JSD")
    print("\nCLOSEST personas:", ", ".join(
        f"pid {int(r.pid)} {r.JSD:.4f}" for r in best.itertuples()))
    print("FURTHEST personas:", ", ".join(
        f"pid {int(r.pid)} {r.JSD:.4f}" for r in worst.itertuples()))

    # does JSD say the same thing as the paired metric?
    align_path = NUMBERS_DIR / "personaAlignment.parquet"
    if align_path.exists():
        a = duckdb.sql(f"SELECT pid, alignment FROM '{align_path}'").df()
        merged = dist_df.merge(a, on="pid")
        corr = merged.JSD.corr(merged.alignment)
        print(f"\ncorrelation(JSD, alignment) = {corr:+.3f}")
        print("  strongly negative -> the two metrics agree")
        print("  near zero         -> the twin matches the population but not the person")

    for g in groupings:
        if g["grouping"] == "all_persona":
            continue
        print(f"\n{g['grouping'].upper()}")
        for entry in g["rank"]:
            n = g["sampleSize"].get(entry["name"], 0)
            print(f"  {entry['score']:.4f}  n={n:<4} {entry['name']}")


def main():
    ensure_dirs()
    for d in (DIST_DIR, DIST_MISC_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not SYNTHETIC_PARQUET.exists():
        raise SystemExit("no synthetic responses - run getSyntheticData.py first")

    human, synth = load_by_persona()
    dist_df, hists = compute(human, synth)
    groupings = build_groupings(dist_df)

    dist_df.to_parquet(PERSONA_DISTANCES, index=False)
    pq.write_table(
        pa.Table.from_pylist(groupings, schema=DISTANCE_SCHEMA),
        CALCULATED_DISTANCES,
    )
    print(f"wrote {PERSONA_DISTANCES.name}            {len(dist_df):,} rows")
    print(f"wrote {CALCULATED_DISTANCES.name}  {len(groupings)} groupings")

    print(f"drawing {len(dist_df)} distribution diagrams ...")
    for row in dist_df.itertuples():
        pid = int(row.pid)
        draw_persona_distribution(pid, hists["human"][pid],
                                  hists["synthetic"][pid], row.JSD)

    for g in groupings:
        if g["grouping"] == "all_persona":
            continue
        draw_rank(g["grouping"], g["rank"], g["sampleSize"],
                  DIST_MISC_DIR, value_label="JSD")

    report(dist_df, groupings)


if __name__ == "__main__":
    main()
