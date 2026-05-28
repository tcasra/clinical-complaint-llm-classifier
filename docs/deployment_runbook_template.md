# Deployment Runbook Template

This template is the site-specific operational companion to the field guide. Fill it out before any pilot or deployment.

## System Identity

- Tool name: Clinical Voice-Intake Complaint Classifier
- Repository version / commit:
- Deployment date:
- Site name:
- EHR / intake system:

## Named Owners

- Clinic informatics lead:
- ML maintainer:
- Medical director:
- IT / integration owner:
- Escalation email list:
- After-hours contact:

## Deployed Configuration

- Transcript mode enabled: `gold` / `asr` / both
- Classifier model:
- STT model:
- Prompt mode: `zero_shot` / `few_shot`
- Few-shot example count:
- Confidence threshold:
- High-risk confidence threshold:
- High-risk labels:

## Acceptance Test Record

- Gate 2 command used:
- Gate 2 artifact directory:
- Gate 1 label coverage completed on:
- Gate 2 site acceptance sample size:
- Gate 2 accuracy:
- Gate 2 macro-F1:
- Gate 2 high-risk recall:
- Gate 2 auto-accept accuracy:
- Gate 3 subgroup audit completed on:
- Gate 4 workflow rehearsal override rate:
- Pilot start date:
- Pilot end date:
- Approval to enable auto-accept granted by:

## Operational Thresholds

- Soft alarm trigger:
  Use the thresholds defined in the field guide monitoring table.
- Hard alarm trigger:
  Use the thresholds defined in the field guide monitoring table.
- Auto-pause behavior:
  Disable auto-fill and revert to advisory mode only.

## Required Dashboards / Reports

- Daily volume
- Auto-accept rate
- Audited auto-accept accuracy
- Audited macro-F1
- High-risk recall
- Override rate
- Out-of-vocabulary rate
- STT word overlap on audited subset
- Worst subgroup gap
- API error rate
- Cost per encounter

## Audit Log Location

- Production audit log path or table:
- Weekly audit sample owner:
- Monthly monitoring review owner:

Use the provided template at `outputs/templates/intake_audit_log_template.csv` as the minimum schema.

## Pause / Escalation Procedure

1. If a hard alarm fires, pause auto-accept routing immediately.
2. Notify the clinic informatics lead and ML maintainer within 24 hours.
3. Pull the last 7 days of overridden, low-confidence, and high-risk cases.
4. Review at least 30 recent cases and identify whether the issue is STT drift, classifier drift, workflow drift, or label-set mismatch.
5. Re-run acceptance testing before re-enabling auto-accept.

## Change Log

| Date | Change | Approved by | Notes |
|---|---|---|---|
| | | | |
