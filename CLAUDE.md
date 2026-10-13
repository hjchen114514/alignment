# Twin-2K-500 ICL Alignment

Research pipeline: condition an LLM in-context on a real person's wave 1-3
survey answers (Twin-2K-500, HF `LLM-Digital-Twin/Twin-2K-500`, config
`wave_split`), ask it the wave 4 questions that person actually answered, and
score how closely the synthetic answers match the human's. A secondary JSD
metric (`distances.py`) was requested by a collaborator's lab; keep it, but it
is not the headline number.

## Commands

Python 3.14, venv at `.venv/`. Use the venv interpreter (`.venv/bin/python`).

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests -q            # fast, no network, no real data needed
.venv/bin/python scripts/importData.py         # HF dataset -> data/*.parquet
.venv/bin/python scripts/getSyntheticData.py   # calls the model; resumable
.venv/bin/python scripts/alignmentResult.py    # -> result/numbers/*.parquet + terminal report
.venv/bin/python scripts/drawDiagram.py        # -> result/diagram/{persona,question,miscellaneous}
.venv/bin/python scripts/distances.py          # -> JSD numbers + result/diagram/distances
```

The OpenAI key goes in `.env` at the project root (`OPENAI_API_KEY=...`, gitignored).

## Pipeline

1. **importData.py**: reads the HF dataset (not the local `raw_data/` files,
   which are only for checking by hand; the repo must stay shareable) for the
   first `n_personas`, and writes four tables to `data/`. Skips descriptive/DB
   questions, any block containing "Anchoring", and QID163-170 (probing).
   After skipping, each persona has 16 blocks, 59 questions, 94 answers.
2. **getSyntheticData.py**: for each persona (first `n_synthetic` by pid; null
   = all), for each block it saw, calls `aiGateway.ask_ai(icl, questions, pid)`
   once with every question of that block, then writes the answers.
3. **alignmentResult.py**: computes persona-, question- and group-level scores,
   writes three parquets, and prints highlights (best/worst, spread gaps, each
   grouping ranked).
4. **drawDiagram.py**: bars from the paired responses (SQL on synthetic +
   human), reference lines and rank charts from alignmentResult's parquets.
5. **distances.py**: per-persona JSD between the normalized human and
   synthetic answer distributions, plus group averages and diagrams.

`adaptors/aiGateway.py` is the ONLY file that knows the model provider
(OpenAI Responses API, model from `config.yaml`, currently `gpt-6-luna`).
Its contract is in its docstring: one int per question, same order, raise on
failure. No tools are used: just one structured-output call per block.

## Data model

Question key is `(qid, row_id)`. Matrix expands to one row per `Rows` entry,
Slider to one per value; MC/TE always have `row_id = 0`.

| question type | text | options | answer |
|---|---|---|---|
| Matrix | QuestionText + row label | Columns | SelectedByPosition (1-based) |
| MC | QuestionText | Options | SelectedByPosition (1-based) |
| Slider | QuestionText | none, uses Range | Answers.Values |
| TE | QuestionText | none, range from `te_ranges` | int(Answers.Text) |

- `data/human_responses.parquet` / `synthetic_responses.parquet`: pid, qid,
  row_id, answer (int), answertext, normalized (0-1), and `status`
  (finished | pending) for synthetic only.
- `data/human_retest.parquet`: same schema; wave 4 re-answers (`wave4_Q_wave4_A`),
  the source of the retest ceiling. `human_responses` comes from `wave4_Q_wave1_3_A`.
- `data/questions.parquet`: qid, row_id, category, block, question_type,
  questiontext, options, range_min, range_max.
- `data/persona.parquet`: pid, gender, age, education, race,
  favoredPoliticalParty, income, politicalViews (from wave 1-3 by QID12-22 in
  `demographic_qids`), icl (`wave1_3_persona_text`).
- `normalized = (answer - range_min) / (range_max - range_min)`.
- Categories by block name: "Product Preferences" -> Pricing Preferences,
  "Non-experimental" -> Non-experimental heuristics and biases, else
  Behavioral economics.

## Metrics

- Answer alignment = `1 - |synthetic_norm - human_norm|`.
- **Persona alignment is two-stage**: average within each block, then across
  blocks (paper method, so the 40-question pricing block does not dominate).
  Never flatten to a plain mean over answers.
- **Ceiling (retest accuracy)** = the same two-stage score between a human's
  `human_responses` and `human_retest` answers.
- Question alignment = mean answer alignment across personas, with
  `synthetic_std` / `human_std` of the answers to show whether the model's
  answers collapse while humans disagree.
- There is no shuffle floor (removed on purpose; do not reintroduce).

## Outputs

`result/numbers/`
- `personaAlignment.parquet`: pid, alignment, n_question, ceiling
- `questionAlignment.parquet`: qid, row_id, block, category, alignment,
  n_persona, ceiling, synthetic_std, human_std
- `calculatedResult.parquet`: one row per grouping (all_persona, race, gender,
  age, education, income, favoredPoliticalParty, politicalViews, all_question,
  block, category) with map columns sampleSize, retestAccuracy, avgAlignment
  and a `rank` list ascending by score. Parquet, not JSON, so every output has
  one file type.
- `personaDistances.parquet` (pid, JSD, n_question) and
  `calculatedResult_distances.parquet` (persona groupings only, no question level).

`result/diagram/`
- `persona/persona_<pid>.png`: x = question, y = alignment; dashed lines for
  the average alignment of all personas, this persona's retest ceiling and
  this persona's own alignment (red, green, yellow); title shows n questions.
- `question/question_<qid>_<row>.png`: x = persona, y = alignment; dashed
  lines for the average alignment of all questions, this question's retest
  ceiling and this question's own alignment; title shows n personas.
- `miscellaneous/rank_<grouping>.png` (rank_allPersona, rank_allQuestion,
  rank_race, ...): bars ranked ascending left to right, n under each bar.
- `distances/persona_<pid>.png` (human vs synthetic density, JSD in title)
  and `distances/miscellaneous/rank_*.png` (persona groupings only).

## Conventions and invariants

- Generation results append to `data/synthetic_responses.jsonl` as they
  arrive, then collapse to parquet; a finished row beats a pending row for the
  same key. Reruns resume from the JSONL (finished `(pid, qid)` are skipped), so
  never delete it unless you intend to pay to regenerate everything.
- Prompt caching: instructions + persona icl form a byte-identical prefix for
  all 16 of a persona's blocks; the questions must come after it. gpt-5.6+
  caches only at breakpoints, so the icl block carries an explicit
  `prompt_cache_breakpoint` and `prompt_cache_options.mode` is `explicit`
  (the automatic breakpoint lands after the questions and never hits).
  `prompt_cache_key` is `persona-<pid>`.
- `getSyntheticData` runs personas in a thread pool (`concurrency`), blocks of
  one persona sequentially. DuckDB inside worker threads needs its own
  `duckdb.connect()`: threads sharing the default `duckdb.sql` connection deadlock.
- SQL is DuckDB directly over parquet files.
- All tunables live in `config.yaml`; `scripts/common.py` loads it once as
  `CFG` and defines every path. Code outside `scripts/` imports it as
  `scripts.common`.
- Names are camelCase in places (`getSyntheticData.py`, `askQuestion`,
  `avgAlignment`): match the surrounding file rather than renaming.
- Tests live in `tests/`; `conftest.py` builds a tiny hand-computed dataset
  and monkeypatches module path constants to point at it. When changing a
  calculation, update the hand-worked expected values there.
- Generated files (`data/*.parquet`, `data/*.jsonl`, `result/**`), `.env`
  and `raw_data/` are gitignored.
