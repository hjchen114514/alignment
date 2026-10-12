# Twin-2K-500 ICL Alignment

Measures how closely LLM-generated synthetic survey responses match real human
responses, using the [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
dataset. Personas are conditioned in-context on their wave 1-3 responses, then
asked the wave 4 questions they actually answered.

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

Settings live in `config.yaml`. `n_synthetic` limits generation to the first n
personas (`null` = all); raising it later skips personas already finished.

## Metric

Alignment follows the Twin-2K-500 paper: for each answer,
`1 - |human - synthetic| / range`, averaged **within each block first**, then
across blocks. The two-stage average stops the 40-question pricing block from
dominating the 15 single-question blocks.

| | meaning |
|---|---|
| **ceiling** | the same human answering the same questions twice (~82.8% on 100 personas; the paper reports 81.72%) |
| **alignment** | the synthetic twin vs the real human |

`distances.py` adds Jensen-Shannon divergence as a secondary, *unpaired* view:
it compares the shape of the two answer distributions without regard to which
answer went with which question. The two metrics disagreeing is a finding, not
a bug.

## Model

`adaptors/aiGateway.py` calls the OpenAI Responses API with the model named in
`config.yaml` (`gpt-6-luna`), reading the key from `.env`.

## Data

| file | contents |
|---|---|
| `data/human_responses.parquet` | real answers, one row per (pid, qid, row_id) |
| `data/human_retest.parquet` | the same people answering again - the ceiling |
| `data/synthetic_responses.parquet` | model answers, same schema plus `status` |
| `data/questions.parquet` | question text, options, range, block, category |
| `data/persona.parquet` | demographics and the ICL conditioning text |

## Results

| path | contents |
|---|---|
| `result/numbers/personaAlignment.parquet` | alignment and ceiling per persona |
| `result/numbers/questionAlignment.parquet` | alignment, ceiling and answer spread per question |
| `result/numbers/calculatedResult.parquet` | averages and rankings by demographic, block and category |
| `result/numbers/personaDistances.parquet` | JSD per persona |
| `result/numbers/calculatedResult_distances.parquet` | JSD averages and rankings by demographic |
| `result/diagram/persona/` | alignment per question, one chart per persona |
| `result/diagram/question/` | alignment per persona, one chart per question |
| `result/diagram/miscellaneous/` | ranked bar charts per grouping |
| `result/diagram/distances/` | human vs synthetic answer distributions, plus ranked JSD charts |

Generated files are gitignored; rerun the scripts to rebuild them.

## Tests

```bash
python -m pytest tests -q
```

No network or real data needed - the tests build a tiny hand-computed dataset.
