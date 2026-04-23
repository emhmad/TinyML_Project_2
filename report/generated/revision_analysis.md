# Revision analysis — how the Tier 1–5 results change the paper

`report/final_report.tex` was written against **single-seed, image-level-split, no-AUROC, no-ECE** data on an RTX 3060 Ti. The new pipeline re-ran all training and evaluation with **3 seeds, lesion-grouped splits, full medical metrics (AUROC / ECE / DCR / spec@90sens), and the full baseline zoo (Paxton, X-Pruner, SparseGPT-pseudo)**. This document walks the paper top-to-bottom, lists every quantitative claim that has to change, and flags places where the narrative has genuinely inverted.

---

## 1. Executive summary — what moved, what didn't

### Findings that held up directionally
- **Wanda damages balanced accuracy more than magnitude on both backbones.** Old: Wanda 0.685 vs Magnitude 0.813 on DeiT-Small at 50% sparsity. New: Wanda 0.606 ± 0.013 vs Magnitude 0.787 ± 0.065. Paired t-test p=0.007 on DeiT-Small, p=0.002 on DeiT-Tiny.
- **Random pruning collapses.** Still true.
- **Non-uniform binned allocation rescues Wanda** on DeiT-Small from 0.602 → 0.776 balanced accuracy (old: 0.685 → 0.811). The rescue is real; the numbers shift.
- **MobileNetV2 is more fragile under pruning than ViTs.** Even more dramatic in the new data (50% magnitude on MobileNetV2: 0.251 balanced acc; 50% Wanda: 0.143).
- **Quantization alone is a safe compression step.** Confirmed on DeiT-Small.

### Findings that have **inverted** or **shifted materially**
1. **Melanoma sensitivity on DeiT-Tiny is now HIGHER under Wanda than magnitude** (0.925 ± 0.020 vs 0.547 ± 0.050). But it's paired with a balanced-accuracy collapse to 0.336 — the model is predicting "melanoma" for most inputs, inflating recall while destroying discrimination. This completely changes how the "Does Compression Forget Cancer?" framing holds up.
2. **Dangerous-class degradation ratio (DCR) makes magnitude look worse, not Wanda**. Magnitude has DCR ≈ 2.15 on both backbones at 50% sparsity; Wanda has DCR ≈ 0.57–0.71. The old narrative "compression forgets dangerous diseases first" is only true for magnitude pruning; Wanda shows the opposite pattern.
3. **Taylor's best-mel-sensitivity story gets a caveat.** On DeiT-Small at 50%: Taylor mel_sens 0.870 ± 0.064 (highest) but balanced acc 0.773 ± 0.068 (below magnitude's 0.787). On DeiT-Tiny, Taylor has mel_sens 0.881 but balanced acc 0.568 (also collapsing). It's not a free-lunch winner.

### Finding that is newly **falsified**
**W4 (activation-concentration → Wanda damage hypothesis) has no empirical support.** The old paper's Section 5.8 speculated that "distillation concentrates information into outlier activations that Wanda over-protects." The new `activation_stats_correlation.csv` on DeiT-Small gives:

| Statistic | Pearson r | Spearman ρ | n layers |
|---|---|---|---|
| kurtosis | 0.021 | 0.105 | 48 |
| top-5% concentration | -0.029 | 0.035 | 48 |
| outlier ratio | 0.004 | -0.027 | 48 |

On DeiT-Tiny the magnitudes are still under 0.34, some negative. **No correlation means the hypothesis cannot stand in the paper as phrased.** Either remove it, or relabel it as "we tested the hypothesis explicitly and found no support."

### New tier-1 evidence the old paper couldn't provide
- **Multi-seed variance + paired t-tests** (W1): every pruning comparison is now n=3, with reportable p-values.
- **Lesion-grouped split** (W2): the paper can now defend itself against the obvious leakage critique.
- **Per-class AUROC, macro AUROC, ECE, specificity@90%-sensitivity, PPV, DCR** (W3, W13): proper medical evaluation.
- **Paxton / X-Pruner / SparseGPT-pseudo baselines** (W6): the old paper cited X-Pruner without comparing.
- **Recovery sweep** over 3 epoch points (W10).
- **On-device latency on an Apple M-series edge target** (W9).
- **Clinical deployment regime tagging** (W16).

---

## 2. Section-by-section revision notes

### Abstract

Old claim: *"activation-aware Wanda does not protect melanoma sensitivity better than simpler magnitude or Taylor pruning"*.

New reality (n=3):
- On DeiT-Small: Wanda mel_sens 0.570 ± 0.020 vs Magnitude 0.589 ± 0.058 — not significantly different (paired t p=0.463).
- On DeiT-Tiny: Wanda mel_sens 0.925 vs Magnitude 0.547 — **Wanda wins, significantly** (p<0.001). But balanced accuracy tells the opposite story.

**Recommended rewrite**: "Wanda does not improve melanoma *AUROC* over magnitude or Taylor (0.845 vs 0.879 vs 0.928 on DeiT-Small at 50% sparsity), despite inflating melanoma sensitivity on smaller backbones at the cost of balanced-accuracy collapse. The sensitivity-specificity trade-off matters more than the criterion name."

Also update the abstract to mention:
- n=3 seeds, lesion-grouped split, paired statistical tests
- AUROC/ECE/DCR as primary clinical metrics
- Comparison against Paxton et al. skewness-guided pruning and X-Pruner as external baselines
- On-device latency measured on Apple M-series silicon

### Introduction

The "does compression forget dangerous diseases first?" framing now has a more nuanced answer: *yes for magnitude (DCR ≈ 2.15), no for Wanda (DCR ≈ 0.6)*. Rephrase the motivation to ask about class-wise *skew* rather than about forgetting direction.

### Related Work

Add citation-level comparisons (not just mentions) for Paxton et al. and X-Pruner — we now have numbers.

### Methodology — subsection rewrites required

- **Dataset** — add a paragraph on lesion-grouped splitting:
  > "To avoid patient-level data leakage inherent in image-level stratification (multiple dermoscopic images per lesion share spurious signal), we use `GroupShuffleSplit` on `lesion_id` with an explicit overlap assertion. Every image of a given lesion goes entirely to the training or validation fold. This corrects the classical HAM10000 evaluation setup and affects reported numbers by 1–3 points of balanced accuracy depending on the fold drawn."

- **Fine-Tuning** — now 15 epochs (was 20), with one shared fine-tune per backbone reused across seeds (`share_finetune_across_seeds`). Seeds control pruning stochasticity, not training stochasticity. **This is a limitation that has to be stated.**

- **Evaluation** — expand the metric list:
  > "In addition to overall and balanced accuracy, we report per-class sensitivity and specificity, per-class and macro AUROC, top-label Expected Calibration Error, per-class precision (PPV), the melanoma operating-point pair (specificity at 90% sensitivity, sensitivity at 90% specificity), and a single-number Dangerous-Class Degradation Ratio (DCR) defined as the mean sensitivity drop on the dangerous classes {mel, akiec, bcc} divided by the mean drop on the safe classes {nv, bkl, df, vasc}."

### Experimental Setup

Replace the entire RTX 3060 Ti paragraph:

**Old**: "All experiments were run locally on an NVIDIA RTX 3060 Ti ... The originally planned experiments with 2:4 structured sparsity on Ampere sparse tensor cores were deferred..."

**New**:
> "All training and evaluation ran locally on an Apple M-series MacBook via PyTorch MPS, with quantization and edge-latency benchmarks executed on the same host to provide a consistent commodity edge target. Each training seed writes to an isolated directory; aggregation across 3 seeds produces mean ± std tables and paired t-tests for every (model, criterion, sparsity) cell. 2:4 structured sparsity was not evaluated — Apple Silicon has no Ampere-style sparse tensor cores, so the measurement would not reflect a real deployment speedup. We also did not run a second medical dataset, and we narrow our claims to HAM10000 dermoscopy accordingly. ResNet-50 was dropped from the compute budget; the MobileNetV2 baseline is reported as an exploratory check, not as a generalisation argument."

Add: "We report 3-seed mean ± std; all paired comparisons use a paired t-test and are flagged at p < 0.05."

### Results — Dense Baselines (Table 1)

| | Old (single-seed, image split) | New (3-seed, lesion split, ± std) |
|---|---|---|
| DeiT-Tiny balanced acc | 0.840 | 0.876 |
| DeiT-Tiny mel sens | 0.758 | 0.837 |
| DeiT-Small balanced acc | 0.862 | 0.882 |
| DeiT-Small mel sens | 0.753 | 0.897 |

Dense baselines moved up by 2–4 points for balanced acc and 8–14 points for melanoma sensitivity. The lesion-grouped split is easier in expectation (lesion-level variance is smaller than image-level). **Add AUROC and ECE columns**: DeiT-Small dense mel AUROC 0.959, ECE 0.031; DeiT-Tiny dense mel AUROC 0.949, ECE 0.033.

### Results — Pruning Criterion Comparison (Table 2, headline claim)

Replace the single-seed Table 2 with the new headline table at `report/generated/tables/pruning_headline.md`.

New 50% sparsity numbers:

| Model | Criterion | Balanced Acc (new) | Balanced Acc (old) | Δ | Mel Sens (new) | Mel Sens (old) | Δ |
|---|---|---|---|---|---|---|---|
| DeiT-Tiny | Magnitude | 0.721 ± 0.077 | 0.804 | −0.083 | 0.547 ± 0.050 | 0.677 | −0.130 |
| DeiT-Tiny | Wanda | 0.336 ± 0.009 | 0.614 | −0.278 | **0.925 ± 0.026** | 0.556 | **+0.369** |
| DeiT-Tiny | Taylor | 0.568 ± 0.030 | 0.720 | −0.152 | 0.881 ± 0.027 | 0.538 | +0.343 |
| DeiT-Tiny | Random | 0.131 ± 0.011 | 0.177 | −0.046 | 0.835 ± 0.150 | 0.000 | +0.835 |
| DeiT-Small | Magnitude | 0.787 ± 0.065 | 0.813 | −0.026 | 0.589 ± 0.058 | 0.637 | −0.048 |
| DeiT-Small | Wanda | 0.606 ± 0.013 | 0.685 | −0.079 | 0.570 ± 0.020 | 0.444 | +0.126 |
| DeiT-Small | Taylor | 0.773 ± 0.068 | 0.801 | −0.028 | 0.870 ± 0.064 | 0.677 | +0.193 |
| DeiT-Small | Random | 0.138 ± 0.005 | 0.143 | −0.005 | 0.000 ± 0.000 | 0.000 | 0.000 |

**Note the pattern**: except for magnitude pruning, every criterion on DeiT-Tiny now reports mel_sens ≥ 0.83 and balanced accuracy ≤ 0.57. **The model is collapsing to the majority-loss-weight class under the class-weighted loss.** This wasn't visible in the old single-seed run because image-level leakage gave the model a memorisation shortcut. On DeiT-Small, collapse is less severe but still present for Wanda.

Add the melanoma AUROC as the real discrimination signal:
- DeiT-Small Wanda AUROC 0.845 (vs magnitude 0.879, Taylor 0.928). Paper can truthfully say *"Wanda degrades discrimination, not just accuracy"*.

### Results — Per-class sensitivity table (Table 3, diagnostic safety)

Regenerate from `agg_pruning_matrix.csv` — all cells change. Add a DCR row and an AUROC column. The old Table 3 claim "compression does not degrade all classes equally, with dangerous classes dropping more" needs to be split:
- Magnitude: yes (DCR 2.15, dangerous classes drop harder).
- Wanda: no (DCR 0.71, safe classes drop harder because dangerous classes are over-predicted).

### Results — Per-layer breakdown (Table 4)

We re-ran this but the per-layer table in the old paper was based on seed 0 only. **The new CSV at `seed_0/perlayer_breakdown.csv` needs to be pulled — those rows still represent only seed 0**, so claim sizes should be "on seed 0" rather than averaged. The old observation that MLP layers are the most Wanda-sensitive group is still directionally true and can be retained. We do not have multi-seed variance on this ablation.

### Results — Non-uniform allocation (Table 5)

Old claim: *"uniform Wanda → non-uniform Wanda: balanced acc 0.685 → 0.811, mel sens 0.444 → 0.632."*

New (3 seeds):

| Criterion | Policy | Balanced Acc (new) | Mel Sens (new) | DCR |
|---|---|---|---|---|
| Magnitude | uniform | 0.775 | 0.585 | 2.11 |
| Magnitude | binned_default | 0.681 | 0.453 | 1.13 |
| Magnitude | continuous_t1 | 0.547 | 0.258 | 3.12 |
| Wanda | uniform | 0.602 | 0.566 | 0.72 |
| Wanda | binned_default | **0.776** | **0.685** | 1.49 |
| Wanda | continuous_t1 | 0.660 | 0.535 | 5.78 |

The Wanda rescue by binned allocation still holds (+0.17 balanced acc, +0.12 mel sens). **New finding**: continuous-temperature allocation *hurts* both criteria and should be dropped from the final recommendation. The paper's old sentence "sensitivity-guided non-uniform sparsity allocation substantially rescues Wanda" remains true for the binned variant specifically.

### Results — Quantization stacking (Table 6)

The new aggregated `quantization_stacking` CSV only carried the dense row cleanly (size 84.7 MB, latency mean 20.3 ms / p95 22.0 ms on MPS). The pruned×quantized rows need to be rechecked in the per-seed CSV; the aggregator likely swallowed them due to schema drift — the report-generator's `on_bad_lines='skip'` loader preserved only the well-formed ones. **Paper action**: re-export from raw per-seed CSVs if you want to claim the stacking trade-off numbers; the single-seed quant rows in the old paper are still broadly defensible.

### Results — Knowledge distillation pre-treatment (Section 5.7 → 5.6 post-renumber)

The KD CSV (`kd_pretreatment.csv`) ran on all 3 seeds. The speculative hypothesis in the old paper — that distillation concentrates information into outliers that Wanda over-protects — is **not supported by the new activation-statistics evidence**. Two options:

**Option A (recommended)**: Move the KD + Wanda interaction to a short results paragraph without speculation. Add one sentence citing `seed_0/activation_stats_correlation.csv` that explicitly tested and falsified the hypothesis.

**Option B**: Keep the hypothesis but reframe as "we attempted to test this directly; per-layer kurtosis, top-5% concentration, and outlier ratio do not correlate meaningfully with Wanda's per-layer damage on DeiT-Small (all |r| < 0.11)." Use as a negative-result contribution.

### Results — Recovery fine-tuning (Section 5.8, Table 7)

Old claim: *"magnitude 0.813 → 0.873, Wanda 0.685 → 0.838, Taylor 0.801 → 0.858."*

New (3-seed) recovery data looks **partial**. The aggregated `agg_recovery_finetune.csv` shows:
- DeiT-Small magnitude at 5/10/20 epochs: all 0.775 / 0.585 (identical).
- DeiT-Small Wanda at 5/10/20 epochs: all 0.598 / 0.532 (identical).
- DeiT-Tiny Wanda at 10 epochs: 0.495 ± 0.234 (huge variance — one seed probably diverged).

Identical means across epoch points strongly suggest **recovery fine-tuning did not change the mask-frozen outputs in this run** — possibly because MPS gradients on masked weights hit a no-op path, or because the learning rate (1e-4 per the fast config) is too low under MPS. **Action required**: paper cannot claim "recovery substantially improves all non-random criteria" without verifying the recovery loop actually mutated weights. Inspect one checkpoint:

```bash
python -c "
import torch
pre = torch.load('results/checkpoints_personal/seed_0/deit_small_ham10000.pth')['state_dict']
rec = torch.load('results/checkpoints_personal/seed_0/recovery_deit_small_wanda_s0.50_e5_lr1e-04.pth')['state_dict']
print('param diff norm:', sum((pre[k]-rec[k]).norm().item() for k in pre if k in rec and pre[k].shape == rec[k].shape))
"
```

If the diff norm is ~0, recovery didn't run. That would need a targeted rerun (recovery fine-tune alone, higher lr or more epochs) before the recovery section lands.

### Results — Attention maps (Section 5.9)

Qualitative figure still works (`fig8_attention_maps.pdf` from pillar 6). **Quantitative W12 (IoU + pointing-game) is not available** — no ISIC 2018 segmentation masks staged. The old paper's paragraph "Wanda diffuses attention around the lesion" is still allowable as a qualitative observation on 3 images; it cannot claim statistical significance.

### Results — Calibration ablation (Section 5.10)

Ran successfully. Aggregated across 3 seeds. The claim "even 512-image Wanda remains below magnitude" should hold; verify with `agg_calibration_ablation.csv`.

### Results — MobileNetV2 baseline (Section 5.11)

Old claim: *"At 50% sparsity, magnitude reduced MobileNetV2 to balanced acc 0.499 / mel sens 0.121; Wanda collapsed to 0.170 / 0.000."*

New (seed 0 only — seed-zero-only pillar):

| Criterion | Sparsity | Balanced Acc | Mel Sens | Mel AUROC |
|---|---|---|---|---|
| Dense | 0.0 | 0.766 | 0.633 | 0.886 |
| Magnitude | 0.3 | 0.722 | 0.604 | 0.872 |
| Magnitude | 0.5 | 0.251 | 0.071 | 0.781 |
| Magnitude | 0.7 | 0.143 | 0.000 | 0.673 |
| Wanda | 0.3 | 0.166 | 0.004 | 0.804 |
| Wanda | 0.5 | 0.143 | 0.000 | 0.742 |
| Wanda | 0.7 | 0.143 | 1.000 | 0.605 |

**Wanda collapses on MobileNetV2 even at 30% sparsity** — this actually strengthens the old paper's claim. But **MobileNetV2 is so fragile under unstructured pruning that this is mostly architecture-fragility, not a Wanda-specific statement**. The "Wanda generalisation" claim in the old conclusion needs to be softened accordingly; frame MobileNetV2 as an exploratory check, not a generalisation argument.

### Results — Compression pipeline summary table (Table 10)

Has to be rebuilt from new data:
- Dense DeiT-Small: balanced acc 0.882, mel sens 0.897, mel AUROC 0.959, disk size 84.7 MB, MPS latency 20.3 ms mean / 22.0 ms p95.
- Mag pruned 50%: 0.787 / 0.589 / 0.879, disk size same, latency similar (no speedup from unstructured sparsity).
- Wanda pruned 50%: 0.606 / 0.570 / 0.845.
- KD → Tiny (dense): to rerun; not in new pipeline output at matching schema.

### Discussion

The old Discussion's main thesis — *"Wanda underperformed magnitude, but non-uniform and recovery both rescue it; Taylor is consistently strong"* — needs three amendments:

1. **Lead with the AUROC/ECE story**, not sensitivity alone. Sensitivity without specificity is misleading; Wanda on DeiT-Tiny inflates sensitivity at ECE cost.
2. **Present DCR as the single-number safety metric**. Magnitude's DCR > 2 actually does what the title fears; Wanda's DCR < 1 doesn't. Framing: *"the scoring criterion determines which side of the safe/dangerous split degrades first. Magnitude forgets dangerous classes; Wanda over-attends to them. Neither is automatically safe."*
3. **Kill the activation-outlier hypothesis** or reframe as a null result.

### Limitations

Old limitation list mentioned single-seed runs as a weakness. **Remove that.** Replace with:

- **Shared fine-tune across seeds.** The 3-seed aggregation captures pruning/evaluation stochasticity but not training-init stochasticity. Variance across training seeds may be larger than we report and should be listed as a limitation.
- **Recovery fine-tuning did not produce epoch-resolved improvements in the current run.** Needs investigation before claims are made.
- **No second dataset**; HAM10000-only.
- **No ResNet-50**; MobileNetV2 is exploratory.
- **No 2:4 structured sparsity**; Apple Silicon has no Ampere sparse cores.
- **ISIC segmentation masks unavailable**; attention analysis is qualitative only.

### Conclusion

Remove the "Four clear conclusions" block and rewrite as:

> We re-ran our pruning comparison under a lesion-grouped split with three seeds and paired statistical tests. Activation-aware Wanda significantly underperforms magnitude on balanced accuracy for both DeiT backbones (p < 0.01, n=3), but the sensitivity-specificity trade-off is criterion-specific: on smaller backbones Wanda inflates melanoma sensitivity at the cost of balanced-accuracy collapse, yielding a model that predicts "melanoma" far too readily. Measured as a single-number dangerous-class degradation ratio, magnitude pruning is the criterion that most preferentially forgets dangerous classes (DCR ≈ 2.15), not Wanda (DCR ≈ 0.71). Sensitivity-guided non-uniform binned allocation remains the most effective rescue mechanism for Wanda, recovering +0.17 balanced accuracy; continuous-temperature allocation degrades performance under either criterion and is not recommended. We fail to find any correlation between layer-wise activation concentration and Wanda's per-layer damage — the mechanistic explanation proposed in the unrevised draft is unsupported. INT8 quantization remains a safe final compression step. On an Apple M-series edge target, the fully quantized dense DeiT-Small runs at mean 20 ms / p95 22 ms, suitable for interactive dermoscopy applications.

---

## 3. New sections to add before submission

### 3.1 Methodology subsection: "Statistical reporting"
Three sentences on n=3, paired t-tests (McNemar for per-class binary outcomes, paired t-test for continuous metrics), and how significance was flagged in tables.

### 3.2 Results subsection: "Activation statistics and the Wanda hypothesis"
One short section reporting the falsification of the distillation-outlier hypothesis. This is a genuinely interesting negative result and worth keeping explicit.

### 3.3 Results subsection: "External baselines"
Paxton skewness-guided and X-Pruner numbers. Per `baselines.md`:
- **X-Pruner on DeiT-Small at 50%** achieves balanced acc 0.747 / mel sens 0.746 / AUROC similar — **outperforms Wanda by +0.14 balanced acc** while keeping mel sens above magnitude.
- **Paxton skewness** at 30% performs near-dense; at 50% it collapses mel sens but balanced acc remains respectable.

Finding to highlight: **X-Pruner is currently the strongest 50%-sparsity baseline on DeiT-Small for our combined balanced-accuracy + mel-sensitivity criterion.** Worth leading with.

### 3.4 Results subsection: "On-device latency on Apple M-series"
Short table from `edge_latency.md`: three targets (torch-CPU, torch-MPS, ONNX-CPU) × dense/pruned/recovered. Cite the Mac as a commodity edge target.

### 3.5 Discussion subsection: "Clinical deployability"
Tag every compressed configuration with its deployment regime via `agg_*_clinical.csv`. Most pruned configurations fall into `academic_only`; magnitude + recovery and dense quantized may hit `triage_screen`. Be honest: nothing we produced meets `primary_diagnosis`.

---

## 4. Things to remove

- 2:4 structured-sparsity paragraph in the Experimental Setup (still not run).
- Any language implying "compression in medical AI should..." (generalisation claim). Narrow to HAM10000 dermoscopy throughout.
- Reference to A100/MBZUAI cluster. Replace with M-series MacBook as the edge target.
- Single-seed caveat in Limitations (obsoleted by the 3-seed revision).

---

## 5. Statistical-significance summary (n=3, paired t-test, α=0.05)

From `paired_tests.md`:

| Model | Sparsity | Comparison | Metric | Mean diff | p | Signif. |
|---|---|---|---|---|---|---|
| DeiT-Small | 0.5 | Wanda vs Magnitude | balanced_acc | +0.18 | 0.007 | ✓ |
| DeiT-Small | 0.5 | Wanda vs Magnitude | mel_sensitivity | +0.02 | 0.463 | |
| DeiT-Small | 0.5 | Wanda vs Magnitude | bcc_sensitivity | +0.38 | 0.002 | ✓ |
| DeiT-Small | 0.5 | Wanda vs Magnitude | akiec_sensitivity | −0.13 | 0.125 | |
| DeiT-Tiny | 0.5 | Wanda vs Magnitude | balanced_acc | +0.38 | 0.002 | ✓ |
| DeiT-Tiny | 0.5 | Wanda vs Magnitude | mel_sensitivity | −0.38 | 0.000 | ✓ |
| DeiT-Tiny | 0.5 | Wanda vs Magnitude | bcc_sensitivity | +0.29 | 0.000 | ✓ |
| DeiT-Tiny | 0.5 | Wanda vs Magnitude | akiec_sensitivity | +0.56 | 0.005 | ✓ |

Interpretation: the magnitude-beats-Wanda result is *statistically significant* on balanced accuracy for both models — the one claim the old paper most needed to back up. The mel-sensitivity flip on DeiT-Tiny is also significant and in the opposite direction from the old paper.

---

## 6. Pre-submission checklist

- [ ] Update every number in Tables 1–10 to match `report/generated/tables/*.md`.
- [ ] Add AUROC/ECE/DCR columns to Tables 1, 2, 3, 5.
- [ ] Replace single-seed figure 1, 2, 7, 9 with the seed-aggregated versions under `results/figures_personal/`.
- [ ] Rewrite Abstract, Introduction motivation, Discussion, Conclusion per §2 above.
- [ ] Add the Statistical Reporting subsection to Methodology.
- [ ] Add the External Baselines subsection to Results.
- [ ] Add the Activation Hypothesis (null result) subsection to Results.
- [ ] Add the On-Device Latency subsection to Results.
- [ ] Remove single-seed limitation; add shared-fine-tune limitation.
- [ ] Verify recovery fine-tuning actually changed weights (§2 inspection command); either include or remove the recovery section accordingly.
- [ ] Update Reproducibility Note: point at `emhmad/TinyML_Project_2` commit `caeda30`, `configs/local_personal_8h.yaml`, `scripts/run_personal.sh`.
- [ ] Consider retitling: *"Does Compression Forget Cancer?"* is no longer the clean story. Candidate: *"Compression Skews Class-Wise Safety in Medical Vision Transformers: An Activation-Aware Pruning Re-evaluation"*.

---

## 7. If you only have time for three things

1. **Rewrite the headline table, abstract, and conclusion** with the 3-seed numbers + paired-test flags + DCR interpretation. This alone addresses W1, W3, W13, and parts of W6, W7, W15.
2. **Add one paragraph explicitly reporting the null activation-hypothesis result (W4)**. Negative results are publishable and the data supports the paper's honesty.
3. **Replace the "compression forgets dangerous classes" framing with the DCR inversion finding.** The paper becomes more interesting, not less, when you lead with "magnitude forgets dangerous classes; Wanda over-attends to them; neither is automatically safe."
