# Model Performance Evaluation

## Final Gold (held-out test transcripts, gold text input)
- Sample: **n = 500** stratified across 25 complaint labels (~20 / class)
- **Zero-shot**: accuracy 89.6% (95% CI 87.0%–91.8%), macro-F1 0.892 (95% CI 0.863–0.913), top-3 acc 96.8%
- **Few-shot**: accuracy 92.4% (95% CI 90.2%–94.6%), macro-F1 0.923 (95% CI 0.899–0.944), top-3 acc 98.2%
- **Few-shot lift over zero-shot**: +2.8 pp accuracy, +0.031 macro-F1
- **Cost**: ~$1.612 (0.771 Zero-shot + 0.840 Few-shot)

## Final ASR (end-to-end audio → Whisper STT → classifier)
- Sample: **n = 500** stratified across the 25 labels
- **STT fidelity** (Whisper large-v3-turbo): exact-match 61.8%, avg similarity 0.970, avg word overlap 0.886
- **Zero-shot** (over ASR): accuracy 83.6% (95% CI 79.8%–86.6%), macro-F1 0.834
- **Few-shot** (over ASR): accuracy 89.0% (95% CI 86.2%–91.6%), macro-F1 0.890
- **Gold vs ASR stage-level gap (few-shot)**: 92.4% vs 89.0% (difference 3.4 pp; separate stratified samples, overlap 43/500)
- **Cost**: ~$1.665 total (STT $0.056 + chat $1.609)

## Weakest labels (final_gold Few-shot, by F1)
- **Internal pain** — F1 0.649 (precision 0.71, recall 0.60, support 20)
- **Shoulder pain** — F1 0.844 (precision 0.76, recall 0.95, support 20)
- **Joint pain** — F1 0.857 (precision 1.00, recall 0.75, support 20)
- **Muscle pain** — F1 0.857 (precision 1.00, recall 0.75, support 20)
- **Hard to breath** — F1 0.864 (precision 0.79, recall 0.95, support 20)

## Top confusions (final_gold Few-shot)
- Internal pain → Stomach ache (5 cases)
- Body feels weak → Internal pain (3 cases)
- Cough → Hard to breath (3 cases)
- Injury from sports → Shoulder pain (3 cases)
- Infected wound → Open wound (2 cases)

## Subgroup performance (final_asr Few-shot)
- **background noise audible**:
  - light_noise (n=228): acc 86.8%, F1 0.868
  - no_noise (n=272): acc 90.8%, F1 0.899
- **audio clipping**:
  - light_clipping (n=8): acc 75.0%, F1 0.200
  - no_clipping (n=492): acc 89.2%, F1 0.892
- **quiet speaker**:
  - audible_speaker (n=500): acc 89.0%, F1 0.890

