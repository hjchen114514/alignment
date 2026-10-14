# Twin-2K-500 ICL Alignment

How closely can an LLM stand in for a specific real person on a survey? This
repo measures the **alignment** between real human survey answers and
synthetic answers from an LLM "digital twin" of each person.

Each twin is conditioned in-context on one person's wave 1-3 answers from
[Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
(~24k tokens of their own past answers). The twin then answers the wave 4
questions that person actually answered. Every synthetic answer is scored
against the human's real answer. The same people also re-took those wave 4
questions, which gives a **retest ceiling**: how consistent humans are with
themselves.

**Result with `gpt-6-luna` on the first 100 personas:**

| | score |
|---|---|
| retest accuracy (human vs themself) | **82.78%** (the paper reports 81.72% on ~2,000 personas) |
| alignment (twin vs human) | **72.82%** |
| alignment ÷ retest accuracy | **88.0%** |

The twins get 88.0% of the way to the human consistency ceiling.

> **Replication will be close, not exact.** The model runs at default
> temperature, and the OpenAI Responses API has no `seed` parameter, so a
> rerun produces somewhat different answers. Expect the overall numbers to land
> near the ones above, with individual answers changing. The retest ceiling
> uses no model, so it reproduces exactly.

## What you can do with it

- Test how well **any OpenAI model** stands in for real people: change one
  line in `config.yaml`.
- Swap in **another provider**: only `adaptors/aiGateway.py` knows about the
  model.
- See **where** a model fails: per persona, per question, per block, per
  category, and per demographic group (race, gender, age, education, income,
  party, political views). Every score sits next to the matching human retest
  ceiling.
- Spot **answer collapse**: for each question, how spread out the twins'
  answers are compared with the humans'.
- Scale from a quick 5-persona smoke test to the full dataset by editing
  `n_personas` / `n_synthetic`.

## What I found interesting

All numbers are from the 100-persona run above.

1. **Alignment moves with retest accuracy.** Personas whose own answers are
   less consistent also tend to have lower alignment (r = +0.39 per persona,
   +0.44 per question). *Why it's interesting:* a score means most next to
   its ceiling. It is not a rule, though: the best-aligned persona (88.5%) has
   a below-average ceiling of 69.3%.
2. **Twins are more extreme than people, in both directions.**
   - *More rational:* all 100 twins pick the likelier outcome on all 16
     probability-matching trials; humans do 56-93% of the time on each trial. 82% of twins
     would drive 10 minutes to save $10 on a $250 jacket; 35% of humans would.
     The jacket block scores 35.1% alignment against an 87.7% ceiling.
   - *More biased:* the twins show the Allais certainty effect (61 vs 38
     points), willingness-to-accept vs willingness-to-pay (0.58 vs 0.32) and
     base-rate neglect more strongly than humans do.
   - *Why it's interesting:* the twins seem to answer the way a textbook
     *describes* people, not the way these people answered. A study run on
     these twins would overstate classic effects.
3. **Pricing: right on average, weaker per person.** On 40 "would you buy
   this product at this price" questions (each person saw their own prices),
   alignment is 72.45% vs a ceiling of 84.5% and chance agreement of 50.7%.
   Per product, twin and human buy rates correlate at r = 0.76. *Why it's
   interesting:* the twins know which products are popular but are more
   price-sensitive than people: below half a product's median price, 84% of
   twins buy vs 72% of humans.
4. **Twins are far less varied than people.** The mean std per question is
   0.250 for twins and 0.366 for humans. On 17 of 118 questions, every twin
   gave the identical answer. *Why it's interesting:* conditioning on a
   persona moves the model less than real differences between people do. A
   collapsed answer can still score high when most humans agree with it.
5. **No race gap at n = 100.** Using `mean ± 1.96 × s / √n`, White is
   73.5% (71.9% – 75.2%, n = 61) and Black 70.3% (66.3% – 74.3%, n = 16).
   *Why it's interesting:* the gap looks like "LLMs favor the majority",
   but the intervals overlap, and Black respondents' ceiling is also lower.
   Relative to ceiling the two groups are nearly equal (87.7% vs 86.7%).
   Party, views, gender, age and income intervals overlap too. Education is
   the only demographic with two non-overlapping groups, and only by 0.4
   points.

## How to use this repo

1. **Set up** the environment and API key (below).
2. **Edit `config.yaml`** to choose the model, the number of personas,
   concurrency and your rate limit. Start small (e.g. `n_personas: 5`) to
   check everything works before paying for a full run.
3. **Run the five scripts in order.** Each one prints a report and saves a
   copy to `result/summary/`.
4. **Read the results:** scores are in `result/numbers/*.parquet`, charts in
   `result/diagram/`.
5. **Change something and compare:** a different model, prompt (in
   `aiGateway.py`) or persona count.

## Setup

Python 3.14.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key" > .env      # gitignored
```

## Run

```bash
python scripts/importData.py         # build the source tables
python scripts/getSyntheticData.py   # generate synthetic responses (resumable)
python scripts/alignmentResult.py    # compute alignment scores
python scripts/drawDiagram.py        # draw alignment diagrams
python scripts/distances.py          # JSD distributions (secondary metric)
```

- **Only `getSyntheticData.py` calls the model and costs money.** It makes one
  request per (persona, block), so 100 personas = 1,600 requests.
- **Interrupting is safe.** Each answer is appended to
  `data/synthetic_responses.jsonl` as it arrives. A rerun skips everything
  already finished and retries only failed blocks. Don't delete the JSONL
  unless you want to pay to regenerate everything.
- **Run time:** at 2M tokens/min a 100-persona run takes at least ~23 minutes.
  Every request sends the full ~27k-token persona, and cached tokens still
  count toward the per-minute limit.

### config.yaml

Every setting lives here; no script has hardcoded settings. This file is what
you change to replicate the run or to run something else.

| key | current | meaning |
|---|---|---|
| `dataset` | `LLM-Digital-Twin/Twin-2K-500` | Hugging Face dataset id. Must have the Twin-2K-500 layout (`wave1_3_persona_text`, `wave4_Q_wave1_3_A`, `wave4_Q_wave4_A`) |
| `dataset_config` | `wave_split` | dataset config (subset) to load |
| `model` | `gpt-6-luna` | OpenAI model id used for every synthetic answer |
| `max_retries` | `6` | OpenAI SDK retries on 429 / 5xx errors; a backstop behind the rate limiter |
| `tpm_limit` | `1900000` | tokens per minute the rate limiter allows. Set it a little **under** your account's TPM limit (here 2M, Build tier) |
| `n_personas` | `100` | how many personas `importData.py` loads, in dataset order. Change it, then rerun `importData.py` |
| `n_synthetic` | `null` | how many of those personas `getSyntheticData.py` generates for, first n by pid. `null` = all. Raising it later only generates the new personas |
| `concurrency` | `8` | personas generated in parallel. A persona's 16 blocks always run one after another so they share its prompt cache |
| `data_dir` | `data/` | where the source tables and synthetic answers go |
| `result_dir` | `result/` | where numbers, diagrams and summaries go |
| `skip_blocks` | `["Anchoring"]` | blocks to drop (substring match on the block name) |
| `skip_qids` | `QID163` … `QID170` | questions to drop (the anchoring probes) |
| `demographic_qids` | `gender: QID12` … | which wave 1-3 question supplies each demographic column of `persona.parquet` |
| `categories` | 3 rules | maps block names to categories. First match wins; `"*"` catches the rest |
| `te_ranges` | `QID181/182: [0, 20]` | answer range for text-entry questions, which have none in the source |
| `jsd_bins` | `10` | histogram bins for the JSD metric in `distances.py` |

With the defaults, each persona gets 16 blocks, 59 questions and 94 answers.

## Metric

Alignment follows the Twin-2K-500 paper: for each answer,
`1 - |human - synthetic| / range`, averaged **within each block first**, then
across blocks. The two-stage average stops the 40-question pricing block from
dominating the 15 single-question blocks.

| | meaning |
|---|---|
| **ceiling** (retest accuracy) | the same human answering the same questions twice (~82.8% on 100 personas; the paper reports 81.72%) |
| **alignment** | the synthetic twin vs the real human |

`distances.py` adds Jensen-Shannon divergence as a secondary, *unpaired* view.
It compares the shape of the two answer distributions without regard to which
answer went with which question. The two metrics disagreeing is a finding, not
a bug.

## Model

`adaptors/aiGateway.py` calls the OpenAI Responses API with the model named in
`config.yaml` (`gpt-6-luna`), reading the key from `.env`. It is the only file
that knows the provider. To use another one, rewrite `ask_ai()` and keep its
contract: one int per question, same order, raise on failure.

- **Prompt caching with an explicit breakpoint.** The system prompt + the
  persona's ~24k-token history form an identical prefix for all 16 of a
  persona's blocks. The questions come after it. The history block carries a
  `prompt_cache_breakpoint`, with `prompt_cache_options` set to `explicit` and
  `prompt_cache_key = persona-<pid>`. Block 1 writes the cache and blocks 2-16
  read it. (gpt-5.6+ caches only at breakpoints, and the automatic one lands
  after the questions, where it never hits.) The 100-persona run had an 86.6%
  cache hit rate.
- **Structured outputs.** Responses use a strict JSON schema,
  `{"answers": [int, ...]}`, so there is no free text to parse. The answer
  count is checked, and out-of-range answers are clamped and reported.
- **Rate limiting.** `TokenRateLimiter` paces requests under `tpm_limit`, so
  they wait instead of failing with 429.
- **No tools.** One structured-output call per block.

If a model rejects the cache-breakpoint parameters, remove them from `ask_ai()`.

## Repository layout

```
config.yaml                 every setting (see the table above)
requirements.txt
.env                        your OPENAI_API_KEY (create it; gitignored)
adaptors/
  aiGateway.py              the only model-specific code: prompt, caching, structured output, rate limiter
scripts/
  common.py                 loads config.yaml once, defines every path, writes result/summary files
  importData.py             1. Hugging Face dataset -> data/*.parquet
  getSyntheticData.py       2. asks the model, one request per (persona, block); resumable
  alignmentResult.py        3. alignment + ceiling per persona, question and group -> result/numbers
  drawDiagram.py            4. alignment charts -> result/diagram/{persona,question,miscellaneous}
  distances.py              5. JSD metric -> result/numbers + result/diagram/distances
tests/                      unit tests on a tiny hand-computed dataset (no network)
data/                       generated source tables and synthetic answers (gitignored)
result/
  numbers/                  result tables (parquet)
  diagram/                  charts (png)
  summary/                  each script's terminal output, <YYYY-MM-DD_HHMM>_<script>_summary.md
```

Everything under `data/` and `result/` is generated and gitignored; rerun the
scripts to rebuild it.

## Data

Written to `data/` by `importData.py`, except the synthetic tables, which come
from `getSyntheticData.py`. A question is identified by `(qid, row_id)`:
matrix questions have one row per matrix row, sliders one per statement,
and MC and text-entry questions always have `row_id = 0`.

**`human_responses.parquet`**: the real wave 1-3 answers to the wave 4
questions (`wave4_Q_wave1_3_A`). The twin is scored against these. One row
per (pid, qid, row_id).

| column | type | content |
|---|---|---|
| `pid` | BIGINT | persona id |
| `qid` | VARCHAR | question id, e.g. `QID287` |
| `row_id` | BIGINT | matrix row / slider statement index, else 0 |
| `answer` | BIGINT | 1-based option number (Matrix, MC) or the number entered (Slider, text entry) |
| `answertext` | VARCHAR | the option's text, or the number as text |
| `normalized` | DOUBLE | `(answer - range_min) / (range_max - range_min)`, 0-1 |

**`human_retest.parquet`**: same schema. The same people's wave 4 answers to
the same questions (`wave4_Q_wave4_A`), used only for the retest ceiling.

**`synthetic_responses.parquet`**: same schema plus one column.

| column | type | content |
|---|---|---|
| `status` | VARCHAR | `finished`, or `pending` if the block's request failed (a rerun retries it); `answer` / `normalized` are null when pending |

`synthetic_responses.jsonl` is the append-only log behind it. It is the resume
point; the parquet is rebuilt from it.

**`questions.parquet`**: one row per (qid, row_id).

| column | type | content |
|---|---|---|
| `qid`, `row_id` | VARCHAR, BIGINT | question key |
| `category` | VARCHAR | Behavioral economics / Non-experimental heuristics and biases / Pricing Preferences |
| `block` | VARCHAR | survey block name, e.g. `False consensus` |
| `question_type` | VARCHAR | `Matrix`, `MC`, `Slider` or `TE` (text entry) |
| `questiontext` | VARCHAR | question text, including the matrix row label or slider statement. For pricing questions this is only the first persona's text; see below |
| `options` | VARCHAR[] | answer options for Matrix/MC; empty for Slider/TE |
| `range_min`, `range_max` | DOUBLE | valid answer range, used for normalization |

**`humanPricingQuestionText.parquet`**: one row per (pid, pricing qid). The
survey showed each person a different product and price under the same pricing
qid (e.g. QID9_3 is Goya ham at $35.99 for pid 574 but Doritos at $0.00 for pid
2001), and the same one again in wave 4. `questions.parquet` can hold only one
text per qid, so `getSyntheticData.py` takes each twin's pricing text from here.

| column | type | content |
|---|---|---|
| `pid` | BIGINT | persona id |
| `qid` | VARCHAR | pricing question id, `QID9_1` … `QID9_40` |
| `questiontext` | VARCHAR | the product and price this persona was shown |

**`persona.parquet`**: one row per persona.

| column | type | content |
|---|---|---|
| `pid` | BIGINT | persona id |
| `gender`, `age`, `education`, `race`, `favoredPoliticalParty`, `income`, `politicalViews` | VARCHAR | demographic answer text, e.g. `18-29`, `Some college, no degree` |
| `icl` | VARCHAR | the persona's full wave 1-3 answer history; this is what the twin is conditioned on |

## Results

### Numbers (`result/numbers/`)

**`personaAlignment.parquet`**: one row per persona.

| column | type | content |
|---|---|---|
| `pid` | BIGINT | persona id |
| `alignment` | DOUBLE | two-stage alignment, twin vs human (0-1) |
| `n_question` | BIGINT | answers scored |
| `ceiling` | DOUBLE | two-stage retest accuracy of the same person |

**`questionAlignment.parquet`**: one row per (qid, row_id).

| column | type | content |
|---|---|---|
| `qid`, `row_id` | VARCHAR, BIGINT | question key |
| `block`, `category` | VARCHAR | where the question belongs |
| `alignment` | DOUBLE | mean answer alignment across personas |
| `n_persona` | BIGINT | personas who answered it |
| `ceiling` | DOUBLE | mean retest accuracy on this question |
| `synthetic_std`, `human_std` | DOUBLE | spread of the normalized answers across personas. A much smaller `synthetic_std` means the twins collapse to one answer |

**`calculatedResult.parquet`**: one row per grouping: `all_persona`,
`race`, `gender`, `age`, `education`, `income`, `favoredPoliticalParty`,
`politicalViews`, `all_question`, `block`, `category`.

| column | type | content |
|---|---|---|
| `grouping` | VARCHAR | grouping name |
| `sampleSize` | MAP(VARCHAR, BIGINT) | group value → number of personas (or questions) |
| `retestAccuracy` | MAP(VARCHAR, DOUBLE) | group value → mean ceiling |
| `avgAlignment` | MAP(VARCHAR, DOUBLE) | group value → mean alignment |
| `rank` | STRUCT(name VARCHAR, score DOUBLE)[] | members ranked by alignment, lowest first |

**`personaDistances.parquet`**: `pid` BIGINT, `JSD` DOUBLE (0 = same answer
distribution, 1 = disjoint), `n_question` BIGINT.

**`calculatedResult_distances.parquet`**: like `calculatedResult`, persona
groupings only, with `avgJSD` MAP(VARCHAR, DOUBLE) in place of the alignment
and ceiling columns.

**Viewing parquet:** install a VS Code extension such as **Parquet Explorer**
(query a file with SQL) or **Parquet Viewer** (open it as a table). You can
also query any file directly with DuckDB:
`duckdb -c "select * from 'result/numbers/personaAlignment.parquet' order by alignment"`.

### Diagrams (`result/diagram/`)

Every alignment chart is on a 0-1 scale. Most have three dashed lines:
**red** = average alignment across all personas (or all questions),
**green** = the retest ceiling, **yellow** = this chart's own alignment.

| path | x axis | y axis | notes |
|---|---|---|---|
| `persona/persona_<pid>.png` | each question the persona answered (`qid_row`) | alignment of that answer | red/green/yellow lines; title shows n questions |
| `question/question_<qid>_<row>.png` | each persona who answered (pid) | alignment of that persona's answer | red/green/yellow lines; title shows n personas |
| `miscellaneous/rank_<grouping>.png` | group members, ranked lowest to highest | mean alignment | n under each bar. Includes `rank_allPersona`, `rank_allQuestion`, `rank_block`, `rank_category` and one per demographic |
| `distances/persona_<pid>.png` | normalized answer (0-1, `jsd_bins` bins) | probability density | human (blue) vs synthetic (red) answer distribution; JSD in the title |
| `distances/miscellaneous/rank_<grouping>.png` | group members, ranked | mean JSD | persona groupings only |

### Summaries (`result/summary/`)

`<YYYY-MM-DD_HHMM>_<script>_summary.md`: everything a script printed,
including a crash traceback. `alignmentResult`'s summary is the quickest
overview: best and worst personas and questions, the spread gaps, and every
grouping ranked.

## Tests

```bash
python -m pytest tests -q
```

The tests run without network access, an API key or the real dataset.
`conftest.py` builds a tiny dataset (2 personas, 4 questions) whose alignment,
ceiling and JSD were worked out by hand. The tests check:

- **Data parsing:** normalization and question parsing per question type.
- **Scoring:** the two-stage alignment and ceiling math, and the JSD.
- **Generation:** resume logic and JSONL → parquet collapse, with the model
  faked.
- **Model adapter:** prompt rendering, response parsing and the rate limiter.
- **Outputs:** diagram output and the summary files.

Run them after changing any calculation.
