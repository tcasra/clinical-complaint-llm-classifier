# MPHY Final Project: Clinical Complaint Classification Field Guide Artifact

This repository packages a reproducible evaluation pipeline for a clinical voice-intake classifier. The system maps a short spoken or transcribed patient complaint into one of 25 closed-set intake labels and recommends human review for uncertain or high-risk predictions.

The project compares two LLM-based classification modes:

1. `zero_shot`: transcript plus the allowed label set
2. `few_shot`: transcript plus retrieved labeled examples from the training split

It supports two input pathways:

1. `gold`: use the reference transcript from the metadata CSV
2. `asr`: run audio through speech-to-text first, then classify the transcript

## Repository Structure

```text
MPHY-FINAL/
├── README.md
├── requirements.txt
├── field_guide.md
├── field_guide.pdf
├── data/
│   ├── README.md
│   ├── overview-of-recordings.csv
│   └── recordings/
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── model.py
│   ├── evaluate.py
│   ├── train.py
│   └── visualize.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── figures/
│   └── templates/
├── pipeline.py
├── workflow.py
├── plot_workflow_results.py
└── deployment_runbook_template.md
```

Notes:

- The real implementation now lives under `src/`.
- The root scripts are thin compatibility wrappers so existing commands still work.
- `src/train.py` is the experiment-orchestration entry point. This project does not perform gradient-based model fitting, so “training” here means configuring and running the evaluation workflow for prompt-based methods.
- The final submission should include both `field_guide.md` and an exported `field_guide.pdf` at the repo root.
- `deployment_runbook_template.md` and `outputs/templates/intake_audit_log_template.csv` are included so the governance section points to concrete operational artifacts instead of placeholders.

## Quickstart

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Populate the dataset:

1. Download the **Medical Speech, Transcription, and Intent** dataset from Kaggle:
   `https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent`
2. Extract the metadata CSV plus the three audio archives for train, validate, and test.
3. Arrange the files so this repo has the following local layout:

```text
data/
├── overview-of-recordings.csv
└── recordings/
    ├── train/
    ├── validate/
    └── test/
```

Notes:

- Kaggle distributes the metadata file as `recordings-overview.csv`; rename it to
  `overview-of-recordings.csv` so it matches the default code path used by this repo.
- The raw audio is several GB, so it is not tracked in git here. A fresh clone needs the
  manual dataset step above before any evaluation commands will run.
- See [`data/README.md`](data/README.md) for the expected counts, important columns, and
  reproducibility notes.

Run the unit tests:

```bash
python3 -m unittest discover -s tests
```

Run the recommended evaluation workflow:

```bash
python3 workflow.py \
  --plan recommended \
  --development-limit 75 \
  --final-gold-limit 500 \
  --concurrency 8
```

Add end-to-end ASR evaluation:

```bash
python3 workflow.py \
  --plan full \
  --development-limit 75 \
  --final-gold-limit 500 \
  --final-asr-limit 500 \
  --concurrency 8
```

Render summary figures from the most complete workflow run:

```bash
python3 plot_workflow_results.py
```

## Source Modules

- `src/data_loading.py`: dataset and split access helpers
- `src/model.py`: OpenRouter-facing model helpers and client exports
- `src/evaluate.py`: core evaluation engine, metrics, confidence intervals, artifact writing
- `src/train.py`: workflow staging, stratified sampling, resumable experiment plans
- `src/visualize.py`: figure generation and summary markdown

## CLI Entry Points

- `python3 pipeline.py ...`: run a single stage directly
- `python3 workflow.py ...`: run a staged workflow (`development`, `final_gold`, `final_asr`)
- `python3 plot_workflow_results.py`: render figure assets from workflow outputs

Use `workflow.py` for the built-in benchmark stages on the repo's `validate` and `test` splits.
Use `pipeline.py` when you need to evaluate a site-specific dataset via custom `--csv-path`, `--recordings-dir`, and `--split` arguments.

## Expected Inputs and Outputs

Input:

- audio from `data/recordings/<split>` for `--transcript-source asr`
- transcript text from `data/overview-of-recordings.csv` for `--transcript-source gold`

Output per prediction row:

- one closed-set complaint label from `prompt`
- confidence score
- top-3 label list
- brief extracted clinical details

Output artifacts per run:

- `results.csv` and `results.json`
- `summary.json`
- `stt_records.json`
- `errors.json`
- `confusion_matrix_<method>.csv`
- `per_class_metrics_<method>.json`
- `subgroup_metrics_<method>.json`
- `top_confusions_<method>.json`

## Reproducibility Notes

- Python 3.9+ is supported.
- Set an OpenRouter API key in one of:
  - `OPENROUTER_API_KEY`
  - `OPEN_ROUTER_API_KEY`
  - `MPHY_OPENROUTER_API_KEY`
  - `MPHY_API_KEY`
- A local `.env` file is loaded automatically if present.
- Workflow runs are resumable via `outputs/workflows/...`.

## Current Best Run

The figure set and summary markdown in `outputs/figures/` reflect the strongest completed workflow currently in the repo:

- `final_gold`: held-out transcript evaluation
- `final_asr`: end-to-end audio to ASR to classifier evaluation

Use `outputs/figures/evaluation_summary.md` for the current headline numbers and known failure modes.
Use `deployment_runbook_template.md` for the site-specific operational handoff template and `outputs/templates/intake_audit_log_template.csv` for the minimum live-monitoring schema.
