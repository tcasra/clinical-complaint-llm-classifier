# Clinical Voice-Intake Complaint Classifier — Field Guide

**Track 2 (Clinical NLP) · MPHY Final Project**

---

## 1. Executive Summary

This field guide describes a **closed-set complaint classifier** for a clinical voice-intake (chief-complaint) workflow. Given a short spoken patient complaint, the system either (a) consumes a typed transcript or (b) runs the audio through speech-to-text (STT) first, and then routes it to one of **25 complaint labels** (e.g. *Cough*, *Heart hurts*, *Knee pain*). Every prediction carries a confidence, a top-3 candidate list, and a recommended action — *auto-accept* or *send to human review*.

**Who should use it and when.** Front-desk or telephone-triage staff at a primary-care clinic, urgent-care, or community health setting can use this tool to pre-tag a patient's chief complaint at intake, so it can populate a structured EHR field, drive a triage queue, or pre-load a templated note. It is intended to **assist** an intake clinician or nurse — not replace them.

**Headline performance (held-out test, n = 500, stratified ~20 / class across 25 labels):**

| Pathway | Method | Accuracy (95% CI) | Macro-F1 (95% CI) | Top-3 Acc | Human-review rate |
|---|---|---|---|---|---|
| Gold transcript → classifier | Few-shot | **0.924 (0.902–0.946)** | **0.923 (0.899–0.944)** | 0.982 | 4.2% |
| Audio → Whisper STT → classifier | Few-shot | **0.890 (0.862–0.916)** | **0.890 (0.860–0.914)** | 0.946 | 6.2% |
| Gold transcript → classifier | Zero-shot | 0.896 (0.870–0.918) | 0.892 (0.863–0.913) | 0.968 | 6.4% |
| Audio → Whisper STT → classifier | Zero-shot | 0.836 (0.798–0.866) | 0.834 (0.797–0.861) | 0.890 | 10.4% |

The **few-shot prompting** strategy is the recommended deployment configuration: it adds +2.8 pp accuracy / +0.031 macro-F1 over zero-shot on gold transcripts, and the gain is larger on noisy ASR input (+5.4 pp accuracy). End-to-end inference (STT + classification) costs **≈ $0.0033 per encounter** at current OpenRouter list prices.

---

## 2. Clinical Context

### 2.1 Problem

Front-desk intake staff spend a non-trivial share of every visit translating a patient's free-form complaint ("my back has been killing me for a week") into a structured complaint label that the EHR, triage queue, and downstream order sets can act on. In a high-volume community clinic, this is a low-margin task that:

- **Slows throughput.** Even 30–60 seconds of typing per patient adds up across a busy waiting room.
- **Loses information.** When intake staff are rushed, the structured field is often left blank, set to a generic default ("Other"), or filled with a single keyword that loses nuance.
- **Varies by staff.** Two nurses asked to map the same complaint to the same closed list will not always agree, especially across the spectrum of musculoskeletal complaints (*Joint pain* vs *Knee pain* vs *Muscle pain*).

### 2.2 Current standard of care

Today, intake staff manually choose a complaint from a closed list in the EHR after asking the patient. Some sites use templated phone scripts; others rely on free-text "reason for visit" boxes that are not later structured. Many EHRs offer keyword-based autocomplete, but those systems do not use the patient's own phrasing as input — staff still have to translate.

### 2.3 How this tool changes the workflow

The classifier is wired in **upstream** of the EHR field, not as a replacement for it:

1. The patient (or intake staff) speaks the complaint into a tablet/phone or types it.
2. If audio: STT produces a transcript.
3. The classifier returns a **single label**, a **top-3 ranked list**, a **confidence score**, and a **brief extracted clinical summary** (chief complaint sentence, symptoms, body parts, coarse acuity guess).
4. The intake screen pre-fills the structured complaint field with the predicted label; staff can accept or override with one click.
5. **High-risk labels** (*Blurry vision*, *Hard to breath*, *Heart hurts*, *Infected wound*, *Internal pain*) and **low-confidence cases** are flagged for explicit human review before the value is committed.

The system **does not** make any triage acuity decisions, route the patient to a clinician, or modify the chart. It is a structured-data assist for an intake task that humans already do.

---

## 3. Technical Description

### 3.1 Data requirements

- **Inputs:** either a UTF-8 transcript (≤ ~1 sentence of patient speech, English) or a `.wav` audio file (mono, ≥ 8 kHz, ≤ ~30 s per encounter). The pipeline will resolve which mode to run from a `--transcript-source {gold,asr}` flag.
- **Label space:** a **fixed list of 25 complaint labels** (`Acne`, `Back pain`, `Blurry vision`, `Body feels weak`, `Cough`, `Ear ache`, `Emotional pain`, `Feeling cold`, `Feeling dizzy`, `Foot ache`, `Hair falling out`, `Hard to breath`, `Head ache`, `Heart hurts`, `Infected wound`, `Injury from sports`, `Internal pain`, `Joint pain`, `Knee pain`, `Muscle pain`, `Neck pain`, `Open wound`, `Shoulder pain`, `Skin issue`, `Stomach ache`). The label set is closed at deploy time; new labels require revalidation.
- **Few-shot retrieval pool:** a small labeled "train" split (~380 utterances) used as the source of in-context examples for few-shot mode. No gradient updates.

The development corpus is the open-source *Medical Speech, Transcription, and Intent* spoken-complaint dataset (6,661 short utterances across the 25 labels, with annotated audio-quality fields: `audio_clipping`, `background_noise_audible`, `quiet_speaker`, `overall_quality_of_the_audio`). See [`data/README.md`](data/README.md) for splits.

### 3.2 Architecture (high-level)

The system is a **prompt-based pipeline**, not a fine-tuned model:

```
audio.wav  ─► Whisper large-v3-turbo (STT) ─► transcript ─┐
                                                          ├─► Claude Haiku 4.5 (classifier) ─► JSON
typed transcript ─────────────────────────────────────────┘
```

- **STT:** `openai/whisper-large-v3-turbo` (via OpenRouter). Returns plain transcript.
- **Classifier:** `anthropic/claude-haiku-4.5` (via OpenRouter), called with a structured-output JSON schema that constrains `clinical_label` to the 25-label enum and forces a `top_3_labels` ordered list, a self-reported `confidence ∈ [0,1]`, and lightweight extraction fields (`chief_complaint`, `symptoms`, `body_parts`, `acuity`).
- **Two prompting modes are compared:**
  - `zero_shot`: system prompt + transcript + the allowed label list.
  - `few_shot`: zero-shot prompt + **k = 4** labeled examples drawn from the train split (round-robin coverage so different test items see different examples).
- **Decision policy:** for non-high-risk labels, auto-accept if `confidence ≥ 0.6`; otherwise route to human review. For predicted labels in the high-risk list, apply the stricter threshold `confidence ≥ 0.8`; predictions below that threshold are routed to human review.

### 3.3 Input / output specification

**Input (per encounter):** one of
- `transcript`: free-text string, or
- `audio_path`: path to `.wav`.

**Output (per encounter, JSON):**

```json
{
  "clinical_label": "Knee pain",
  "confidence": 0.91,
  "top_3_labels": ["Knee pain", "Joint pain", "Injury from sports"],
  "chief_complaint": "Patient reports right knee pain after running.",
  "symptoms": ["pain", "swelling"],
  "body_parts": ["knee"],
  "acuity": "low",
  "review_required": false,
  "review_reason": null
}
```

`review_required` is set by the decision policy in §3.2 and is the value the EHR should respect.

### 3.4 Computational requirements

- **No on-prem GPU required.** Both STT and classification are remote API calls.
- **Latency:** ~1–3 s per encounter end-to-end with concurrency of 8 in batch mode (workflow runs cleared 500 ASR encounters in ~4 minutes).
- **Cost (OpenRouter list prices, May 2026):** **≈ $0.0033 / encounter end-to-end** (≈ $0.0001 STT + ≈ $0.0032 classification). At 200 encounters/day, that is < $200/year per clinic.
- **Storage:** transcripts and predictions are written as JSON / CSV to `outputs/workflows/<timestamp>/`. Audio files are not retained by the pipeline beyond the call.
- **Network:** outbound HTTPS to `openrouter.ai`. Pipeline is resumable from a `run_state.json` checkpoint.

Code: see [`src/evaluate.py`](src/evaluate.py) (engine), [`src/train.py`](src/train.py) (workflow staging), [`src/model.py`](src/model.py) (API client), [`src/data_loading.py`](src/data_loading.py) (split access), [`src/visualize.py`](src/visualize.py) (figures), [`workflow.py`](workflow.py) (CLI entry point).

---

## 4. Performance Evaluation

### 4.1 Evaluation protocol

- **Test set:** stratified subsample of `n = 500` encounters from a held-out *test* split, with ~20 examples per label across all 25 labels.
- **Bootstrap:** 95% CIs are reported from 300 bootstrap resamples on the test set.
- **Comparison:** zero-shot vs few-shot (k = 4) is the required two-method comparison. Both modes share the same model, prompt skeleton, schema, and decoding settings (temperature = 0); only the in-context labeled examples differ.
- **End-to-end:** an additional matched stratified sample (`n = 500`) was run through Whisper → classifier so the operational pathway is evaluated, not only the text-only pathway.

### 4.2 Headline numbers (held-out test, n = 500)

| Pathway | Method | Accuracy | 95% CI | Macro-F1 | Top-3 Acc | Auto-accept rate | Auto-accept accuracy |
|---|---|---|---|---|---|---|---|
| Gold | Zero-shot | 0.896 | 0.870–0.918 | 0.892 | 0.968 | 93.6% | 91.7% |
| Gold | **Few-shot** | **0.924** | **0.902–0.946** | **0.923** | **0.982** | **95.8%** | **93.7%** |
| ASR | Zero-shot | 0.836 | 0.798–0.866 | 0.834 | 0.934 | 89.6% | 88.0% |
| ASR | **Few-shot** | **0.890** | **0.862–0.916** | **0.890** | **0.946** | **93.8%** | **92.8%** |

**Few-shot beats zero-shot at every level**: +2.8 pp accuracy / +0.031 macro-F1 on gold, +5.4 pp / +0.056 on ASR. The lift is bigger on noisier ASR input, suggesting in-context examples help anchor the model when the transcript is imperfect. Few-shot is therefore the recommended deployment configuration. STT alone introduces a 3.4 pp accuracy drop versus the gold transcript path (*92.4% → 89.0%*, separately stratified samples).

See figures: [`outputs/figures/01_headline_performance.png`](outputs/figures/01_headline_performance.png), [`05_gold_vs_asr.png`](outputs/figures/05_gold_vs_asr.png).

### 4.3 Subgroup performance

We slice on the dataset's three audio-quality annotations: `background_noise_audible`, `audio_clipping`, `quiet_speaker`. Few-shot on the ASR pathway:

| Subgroup field | Bucket | n | Accuracy | Macro-F1 | Review rate |
|---|---|---|---|---|---|
| background noise | no_noise | 272 | 0.908 | 0.899 | 3.7% |
| background noise | light_noise | 228 | 0.868 | 0.868 | 9.2% |
| audio clipping | no_clipping | 492 | 0.892 | 0.892 | 6.1% |
| audio clipping | light_clipping | 8 | 0.750 | 0.200 | 12.5% |
| quiet speaker | audible_speaker | 500 | 0.890 | 0.890 | 6.2% |

Two findings worth flagging at deployment: (1) **light background noise costs ~4 pp accuracy** (reflected in a higher review rate, which is the system behaving as designed); (2) **audio clipping is rare but degrades sharply** — only 8/500 calls were clipped, but accuracy dropped to 0.75 and macro-F1 collapsed (the support is too small for that F1 to be reliable, but it is consistent with the dropout being driven by a few hard cases). The dataset has no `quiet_speaker` examples in the test slice, so that slice cannot be evaluated. See [`04_subgroup_performance.png`](outputs/figures/04_subgroup_performance.png).

### 4.4 Comparison to baseline / alternative approaches

Two methods were evaluated on identical splits:

- **Baseline (zero-shot):** the LLM is given only the label list. Gold accuracy 0.896.
- **Recommended (few-shot, k = 4):** the LLM is given 4 labeled examples drawn from the train split. Gold accuracy 0.924.

A trivial majority-class baseline would score 4% (1/25) and is reported only for orientation. A keyword-rule baseline was not run; the few-shot/zero-shot delta on this label set already shows that providing in-context exemplars drives most of the lift, and a hand-crafted rule set on 25 overlapping musculoskeletal labels would not generalize to phrasing variation.

### 4.5 Confidence calibration & review economics

The model's self-reported confidence is **moderately calibrated** (see [`06_confidence_calibration.png`](outputs/figures/06_confidence_calibration.png)). The operating policy uses confidence as a routing signal, not a probability:

- Gold few-shot at the deployed thresholds (auto-accept ≥ 0.6, high-risk ≥ 0.8) routes **95.8%** of calls to auto-accept with **93.7%** accuracy on those routed cases — i.e. the routing improves precision-of-the-accepted slice modestly above the unconditional accuracy.
- The remaining **4.2%** are sent to human review, of which ~3.8% are high-risk-label hits that get a stricter threshold. That is the design intent: high-risk labels eat a slightly larger review budget.
- ASR few-shot routes 93.8% to auto-accept at 92.8% accuracy. The review queue grows slightly to 6.2% — operationally cheap, and the increase is driven mostly by genuinely harder noisy inputs.

See [`07_review_economics.png`](outputs/figures/07_review_economics.png) for the full sweep.

### 4.6 Known failure modes

These are observed in the held-out gold few-shot run and are stable across the ASR run.

**Worst labels by F1 (gold few-shot, support = 20 each):**

| Label | Precision | Recall | F1 |
|---|---|---|---|
| Internal pain | 0.71 | 0.60 | **0.649** |
| Shoulder pain | 0.76 | 0.95 | 0.844 |
| Joint pain | 1.00 | 0.75 | 0.857 |
| Muscle pain | 1.00 | 0.75 | 0.857 |
| Hard to breath | 0.79 | 0.95 | 0.864 |

**Top confusions (gold few-shot):** *Internal pain → Stomach ache* (5 cases), *Body feels weak → Internal pain* (3), *Cough → Hard to breath* (3), *Injury from sports → Shoulder pain* (3), *Infected wound → Open wound* (2). The same *Internal pain ↔ Stomach ache* and *Cough ↔ Hard to breath* pairs are the dominant confusions on the ASR pathway as well, so they reflect label-semantic overlap, not STT artifacts.

The dominant pattern: **abdominal/visceral pain** (*Internal pain*) overlaps semantically with *Stomach ache*, and **musculoskeletal labels** (*Joint*, *Muscle*, *Shoulder*) overlap with each other. Two of the worst three are **high-risk** (*Internal pain*, *Hard to breath*), which is *exactly* why the system enforces a 0.8 confidence threshold and human review on those labels.

---

## 5. Governance Framework

This framework is the section a clinic should rely on when deciding to adopt and operate this tool. It assumes a small clinic with no dedicated ML/MLOps team.

### 5.1 Acceptance testing protocol for new sites

Before turning the tool on for live encounters at a new site, run this protocol. A site **must pass all four gates** before a 2-week shadow-mode pilot, and the pilot must pass before any auto-accept routing.

**Gate 1 — Label coverage check.**
Confirm that the clinic's intake complaint vocabulary is covered by the 25-label set. If > 5% of the clinic's prior 30 days of intake complaints fall outside the set, the tool **must not** be deployed without first re-scoping or updating the label set (which requires revalidation, see §5.2).

**Gate 2 — Site acceptance test (n ≥ 100 encounters).**
Collect 100+ stratified encounters from the site, transcribed and labeled by the site's intake staff. Run them through the evaluation engine with site-specific metadata and recordings, for example:

```bash
python3 pipeline.py \
  --csv-path /path/to/site/overview.csv \
  --recordings-dir /path/to/site/recordings \
  --split acceptance \
  --transcript-source gold \
  --methods zero_shot,few_shot \
  --output-dir outputs/site_acceptance
```

Use `--transcript-source asr` instead of `gold` when you want the end-to-end audio pathway. Pass criteria:

- Site-level accuracy ≥ **0.85** (lower 95% CI ≥ 0.78).
- Macro-F1 ≥ **0.80**.
- No high-risk label (*Blurry vision*, *Hard to breath*, *Heart hurts*, *Infected wound*, *Internal pain*) has recall < 0.70.
- Auto-accept accuracy on those that routed to auto-accept ≥ **0.90**.

**Gate 3 — Subgroup audit.**
Repeat Gate 2 within each of the dataset's audio-quality buckets (`background_noise_audible`, `audio_clipping`, `quiet_speaker`) plus any clinic-relevant demographic slice (age band, language). No subgroup may be > 10 pp below the overall site accuracy. If a slice fails, the tool may still launch but **only in transcript-text mode** (no ASR) for that slice.

**Gate 4 — Workflow rehearsal.**
Run a 1-day rehearsal where the prediction is shown to intake staff but **not auto-applied**. Track override rate. If staff override > 25% of predictions, do not proceed.

**Pilot (2 weeks shadow + 2 weeks supervised).**
Shadow mode: predictions are logged but the EHR field is filled by staff as usual. Supervised mode: predictions auto-fill but every prediction is human-verified before commit. Promote to auto-accept only after both phases meet the Gate 2 thresholds at site-level.

### 5.2 Monitoring plan

Track the following continuously and review on the cadence below. Targets are derived from §4 numbers.

| Metric | Target | Soft alarm | Hard alarm | Cadence |
|---|---|---|---|---|
| Daily volume | — | — | — | daily |
| Daily auto-accept rate (ASR) | ≥ 90% | < 88% | < 82% | daily |
| Daily auto-accept accuracy on audited subset | ≥ 90% | < 88% | < 82% | weekly (audit ≥ 30 calls/wk) |
| Daily macro-F1 (audited subset) | ≥ 0.85 | < 0.82 | < 0.75 | monthly |
| High-risk recall on audited subset | ≥ 0.85 | < 0.80 | < 0.70 | monthly |
| Override rate (staff overrides predicted label) | < 15% | > 20% | > 30% | weekly |
| Out-of-vocabulary rate (transcripts where the model returns a low-confidence Internal-pain-ish prediction or staff types in a label not in the 25) | < 5% | > 7% | > 10% | weekly |
| STT word-overlap (vs an audited reference) | ≥ 0.85 | < 0.80 | < 0.70 | monthly |
| Subgroup accuracy gap (worst slice vs overall) | ≤ 10 pp | > 12 pp | > 15 pp | monthly |
| Cost / encounter | ≤ $0.005 | > $0.008 | > $0.012 | monthly |
| API error rate | < 1% | > 2% | > 5% | daily |

**Soft alarm:** open an investigation; the tool stays on. **Hard alarm:** automatic pause of auto-accept routing — predictions are still shown, but no longer auto-fill the EHR field — and an escalation per §5.3.

The audit subset should be a **random 5% of encounters per week**, re-labeled by an intake nurse without seeing the prediction, to estimate ground truth independently. Audit labels feed both the alarm metrics and the model-drift dashboard.

### 5.3 Escalation protocol

**Tier 1 — Soft alarm or staff complaint.**
*Owner:* clinic informatics lead. *SLA:* 5 business days.
- Inspect the audit subset for the affected metric, especially per-label and per-subgroup slices.
- Pull the most-recent week of low-confidence and overridden predictions; sample 30 for case review.
- Decide: continue, raise threshold (e.g. lift the 0.6 to 0.7 globally for 2 weeks), or escalate to Tier 2.

**Tier 2 — Hard alarm, sustained metric breach, or any safety-relevant incident.**
*Owner:* clinic informatics lead **and** the ML maintainer. *SLA:* 24 hours.
- **Auto-pause** auto-accept routing; revert to advisory mode.
- Notify clinic leadership and any institutional AI governance committee.
- Root-cause review: STT change, model change, prompt change, data drift, label-set drift?
- Re-acceptance test (§5.1, Gates 2 + 3) before un-pausing.

**Tier 3 — Patient-harm event or near-miss.**
*Owner:* clinic medical director. *SLA:* immediate.
- **Stop the tool entirely** at the affected site.
- Standard institutional incident-review process applies.
- Tool may not be re-enabled at the site without medical-director sign-off and a documented mitigation.

A single point of contact (the **ML maintainer**) is named at deployment and is reachable for Tier 2/3 issues. Contact details should live in the site's completed runbook. This repo ships a template at [`deployment_runbook_template.md`](deployment_runbook_template.md); the benchmark summary in [`outputs/figures/evaluation_summary.md`](outputs/figures/evaluation_summary.md) is performance-only and should not be treated as an operational contact sheet.

### 5.4 Human-in-the-loop requirements

The system is designed to be **always-supervised** by intake staff. The following are non-negotiable:

1. **No silent auto-fill of low-confidence high-risk labels.** Any prediction in {*Blurry vision*, *Hard to breath*, *Heart hurts*, *Infected wound*, *Internal pain*} with `confidence < 0.8` must be confirmed by a human before commit.
2. **Staff can always override.** The intake screen must show the predicted label, the top-3 list, the confidence, and an "edit" affordance. The tool's response to an override is to log it, not to argue.
3. **Low-confidence cases (< 0.6) never auto-fill** — they show as "needs review" with the top-3 list.
4. **The tool does not make triage acuity decisions.** The `acuity` field in the output JSON is *informational only* and must not drive routing without a clinician-signed-off rule layer on top.
5. **Audit trail.** Every prediction must log: input mode (gold vs asr), model versions (STT + classifier), prompt mode (zero / few), confidence, top-3, and whether staff accepted, edited, or rejected. The offline evaluation pipeline in this repo already emits the model-side fields plus review routing; a live deployment must append staff-action fields using the template at [`outputs/templates/intake_audit_log_template.csv`](outputs/templates/intake_audit_log_template.csv). That audit table is the single source of truth for §5.2 metrics.
6. **Annual model re-validation.** Even if monitoring is green, re-run the §5.1 Gate 2 acceptance test annually, since model providers update underlying weights.

### 5.5 Change management

Any change to the **classifier model**, **STT model**, **prompt template**, **few-shot pool**, or **label set** is a "model change" and triggers a fresh §5.1 Gate 2 + Gate 3 acceptance test before rollout. Do not silently swap models.

---

## 6. Limitations & Risks

### 6.1 Known limitations

- **Closed label set.** 25 labels only. Anything outside that set is forced into the closest available label, which is the wrong behavior for, e.g., obstetric, mental-health-acute, or pediatric-specific complaints not in the list.
- **Not a triage tool.** The `acuity` field is a coarse text-only guess and should not be used to route patients.
- **English-only and short-utterance.** Both Whisper and the classifier were evaluated on short single-sentence English utterances. Longer multi-symptom narratives, code-switched speech, and non-English inputs are out of scope.
- **Single-utterance only.** No memory across encounters, no ability to use prior chart context.
- **Prompt-based, not fine-tuned.** Performance depends on the underlying frontier-model provider's behavior. A silent provider update can shift behavior — covered by §5.5.
- **Confidence is self-reported.** Calibration is reasonable but not strict (see [`06_confidence_calibration.png`](outputs/figures/06_confidence_calibration.png)). Treat confidence as a *routing signal*, not a probability.
- **Subgroup evaluation is shallow.** The dataset's audio-quality fields are the only available subgroup labels — there are no patient-level demographics to slice on. A site evaluation must add demographic slices in Gate 3.

### 6.2 Populations where performance may degrade

- **Patients with strong accents or limited English proficiency** — Whisper word-overlap drops on the ASR pathway by an unknown amount on this dataset (no language tags), and the classifier inherits that error.
- **Patients in noisy waiting rooms** — measurable ~4 pp accuracy drop at "light noise"; expected larger drop at higher noise.
- **Patients reporting multi-system or vague complaints** (*Body feels weak*, *Internal pain*) — these are the labels the model already confuses (§4.6). A patient saying "my whole body just feels off" will be force-mapped.
- **Pediatric patients** — the dataset is adult-style phrasing. Pediatric speech and parent-report phrasing were not represented.
- **Clinics with very different complaint distributions** — if the site sees mostly OB/GYN, pediatrics, or mental health, the label coverage gate (§5.1, Gate 1) will fail and the tool must not be deployed.

### 6.3 Potential for misuse

- **Triage substitution.** Using the model's `acuity` field or `Heart hurts`/`Hard to breath` predictions to silently de-prioritize a patient is unsafe. The system has not been validated for triage.
- **Replacement of intake clinicians.** The economics will tempt clinics to remove a person from the loop. The §5.4 human-in-the-loop requirements must be contractual at deployment.
- **Pseudo-diagnostic claims.** The output JSON includes `chief_complaint`, `symptoms`, `body_parts`. These are extracted from a single utterance and are not a clinical assessment.
- **Re-purposing the audit log for performance reviews.** Override rate is a **system-quality** metric, not a measure of staff competence; using it as one will distort overrides.

### 6.4 What this tool should NOT be used for

1. **Triage** or any acuity-based routing.
2. **Diagnostic decision support.**
3. **Charting** beyond pre-filling a single structured complaint field that a human will accept or override.
4. **Billing / coding** — the labels are intake bins, not ICD/CPT.
5. **Pediatrics, obstetrics, behavioral-health crisis intake** without label-set re-scoping and a fresh acceptance test.
6. **Languages other than English** without re-validation.
7. **Any setting where a missed or wrong label could not be caught by the next human in the workflow within ~1 minute.** The whole governance framework presumes a fast human override loop.

---

## Appendix A — Supporting Repo Artifacts

This appendix is intentionally brief so the exported field guide can stay within the
course's 6-10 page target. The full supporting tables and operational artifacts live in the
repository.

### A.1 Detailed results in the repo

- Per-label metrics: `outputs/workflows/<timestamp>/<stage>/per_class_metrics_<method>.json`
  and `.csv`
- Confusion tables: `outputs/workflows/<timestamp>/<stage>/confusion_matrix_<method>.csv`
  and `.html`
- Top confusions: `outputs/workflows/<timestamp>/<stage>/top_confusions_<method>.json`
- Subgroup performance: `outputs/workflows/<timestamp>/<stage>/subgroup_metrics_<method>.json`
  and `.csv`
- Headline summary used for the figures in this guide:
  [`outputs/figures/evaluation_summary.md`](outputs/figures/evaluation_summary.md)

### A.2 Reproduction and operations

- Reproduction commands and dataset setup:
  [`README.md`](README.md) and [`data/README.md`](data/README.md)
- Site handoff / acceptance-testing template:
  [`deployment_runbook_template.md`](deployment_runbook_template.md)
- Minimum audit-log schema for live monitoring:
  [`outputs/templates/intake_audit_log_template.csv`](outputs/templates/intake_audit_log_template.csv)
