# Twin-2K-500 ICL Alignment

Measures how closely LLM-generated synthetic survey responses match real human
responses, using the [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
dataset. Personas are conditioned in-context on their wave 1-3 responses, then
asked the wave 4 questions they actually answered.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python scripts/importData.py        # build the source tables
.venv/bin/python scripts/getSyntheticData.py  # generate synthetic responses (needs aiGateway)
.venv/bin/python scripts/alignmentResult.py   # compute alignment scores
.venv/bin/python scripts/drawDiagram.py       # draw alignment diagrams
.venv/bin/python scripts/distances.py         # JSD distributions (secondary metric)
```

Settings live in `config.yaml`.

## Metric

Alignment follows the Twin-2K-500 paper: for each answer,
`1 - |human - synthetic| / range`, averaged **within each block first**, then
across blocks. The two-stage average stops the 40-question pricing block from
dominating the 15 single-question blocks.

Three reference points make the number interpretable:

| | meaning |
|---|---|
| **ceiling** | the same human answering the same questions twice (~82.8% on 100 personas; the paper reports 81.72%) |
| **alignment** | the synthetic twin vs the real human |
| **shuffle floor** | the synthetic twin vs 99 *other* humans - what you would score by ignoring the persona entirely |

If alignment sits near the shuffle floor, the in-context conditioning taught the
model nothing person-specific.

`distances.py` adds Jensen-Shannon divergence as a secondary, *unpaired* view:
it compares the shape of the two answer distributions without regard to which
answer went with which question. The two metrics disagreeing is a finding, not
a bug.

## Data

| file | contents |
|---|---|
| `data/human_responses.parquet` | real answers, one row per (pid, qid, row_id) |
| `data/human_retest.parquet` | the same people answering again - the ceiling |
| `data/synthetic_responses.parquet` | model answers, same schema plus `status` |
| `data/questions.parquet` | question text, options, range, block, category |
| `data/persona.parquet` | demographics and the ICL conditioning text |

Generated files are gitignored; rerun the scripts to rebuild them.

## Status

`adaptors/aiGateway.py` is a stub - the model has not been chosen yet. Its
docstring holds the interface contract and implementation pseudocode.
