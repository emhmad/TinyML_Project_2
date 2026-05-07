# Master Context - Does Compression Forget Cancer?

## Final Status — Bundle M++ Complete, Paper Submission-Ready (2026-05-07)

**Submission deadline:** 2026-05-08. Paper PDF compiled cleanly — `report/does_compression_forget_cancer_revised.pdf` (12 pages, 205 KB), 0 unresolved references, 0 missing citations.

### What Bundle M++ actually delivered

The cluster's `Max GPU cards per account: 4` and 8h `--time` limit on `cscc-gpu-qos` capped what could finish in the window. Final effective coverage:

| Component | Status | n |
|---|---|---|
| **S0** (pruning matrix DeiT-Tiny + DeiT-Small) | ✅ done | n=4 cell-level (seeds 0,1,2 rsync'd + seed 3 fresh + seed 4 mostly-fresh); n=3 paired tests (seeds 0,1,2 with full pillar coverage) |
| **L4** (quantitative attention–mask IoU using Tschandl segmentations) | ✅ done | n=5 |
| **P1.1** (per-class gradient flow, mechanism probe) | ✅ done | n=5 |
| **P1.3** (per-head class-conditional activation) | ✅ done | n=5 |
| **Recovery sweep** at corrected `lr=1e-4` | partial | n=2 (seeds 3,4); seeds 0–2 had a no-op recovery from prior compute (verified, deleted) |
| **L2** (ISIC 2019 cross-cohort retrained sweep) | ❌ cancelled | wall-time + QoS cap; remains future work |
| **L3** (MobileNetV2 + ResNet-50 baselines at n=10) | ❌ cancelled | same reason; MobileNetV2 stays as n=1 exploratory check |

### Final paper findings (vs original n=3 paper)

1. **DCR inversion confirmed at higher power.** Magnitude $\DCR = 2.15$ on DeiT-Small, $2.18$ on DeiT-Tiny. Wanda $\DCR < 1$. All Wanda-vs-magnitude paired tests significant at $p < 0.005$.
2. **Mechanism identified (gradient-coupling).** Magnitude's per-layer pruning rate is positively correlated with per-class gradient mass: Pearson $r \approx +0.62$ (dangerous), $+0.74$ (safer), $p < 10^{-5}$ across all 5 seeds. Wanda is uncorrelated ($|r| < 0.20$). Taylor is *negatively* correlated ($r \approx -0.41$, $p < 0.012$). The DCR inversion now has a falsifiable structural account.
3. **Three distinct failure modes** (paper now claims, supported by data):
   - Magnitude → gradient-coupling (high-gradient layers prune more, dangerous classes hit harder per sample)
   - Wanda → classifier-head collapse (constant prediction, high mel sens but low AUROC and balanced acc)
   - Taylor → deliberate gradient-preservation (negatively correlated, best clinical metrics)
4. **Wanda's attention is *better* localised**, not worse. L4 IoU: Wanda 0.434 vs magnitude 0.408; pointing-game 91.4% vs 84.7%. Reframes the original qualitative "Wanda diffuses attention" claim — failure is downstream of attention, at the classifier head.
5. **Activation-outlier hypothesis remains disconfirmed** (preserved in §5.4.1 for transparency).

### Paper edits applied (commit-ready in `report/does_compression_forget_cancer_revised.tex`)

| Patch | Section | Change |
|---|---|---|
| 1 | Abstract | n=4/n=3 phrasing; three-mechanism story; attention IoU finding |
| 2 | Contributions list | New items 5–6 (gradient-coupling mechanism; quantitative attention IoU) |
| 3 | §5.4 Mechanism | Wrapped existing disconfirmation as §5.4.1; added §5.4.2 with new gradient-coupling table (Pearson/Spearman across criteria) |
| 4 | §5.5 (NEW) Quantitative Attention–Mask Overlap | New table with IoU + pointing-game across n=5 seeds; reframes Wanda failure as head-collapse |
| 5 | Limitations | Retired L#1, L#4, L#5 with cross-references to the new sections; L#2 and L#3 documented as deferred |
| 6 | Bibliography | Added `tschandl2018ham10000-seg` (Harvard Dataverse segmentations) |

### Cluster execution lessons (for future runs)

- Repo MUST live at `$HOME/TinyML_Project_2` on CSCC. `/scratch/$USER/...` is login-node-only; compute nodes can't see it and jobs fail silently with no logs. Lustre `/l/users/$USER` is currently down.
- Conda env at `~/.local/share/mamba/envs/tinyml-cancer-forget`. `conda activate <name>` succeeds silently on compute nodes but doesn't update PATH when env was created by mamba/micromamba — every `job_*.sbatch` carries a defensive PATH-prepend after `conda activate` (commit `fe7b9ae`). Always `export ENV_NAME=/full/path/...` before sbatch.
- `cscc-gpu-qos` caps at MaxSubmitJobsPerUser=4 with PreemptMode=REQUEUE; every sbatch carries `--requeue`.
- `processed_metadata.csv` MUST be rebuilt on the cluster (image_path absolute paths are user-machine-specific).
- BasicTeX is bare-bones — to compile the paper locally, install: `multirow`, `tabularx`, `subcaption`, `makecell`, `colortbl`, `fancyhdr`, `titlesec`, `enumitem`, `microtype`, `hyperref`, `caption`, plus PSNFSS + URW base35 fonts. User-mode `tlmgr --usermode install ...` works without sudo.
- See `MBZUAI_HPC_Docs.md` in repo root for the cluster's official rules.

### Repo layout (post-cleanup, 2026-05-07)

- `report/does_compression_forget_cancer_revised.{tex,pdf}` — final paper
- `report/references.bib`, `report/related_work.md` — paper assets
- `report/generated/` — auto-generated tables, statistics.json, this file, revision_analysis, weakness_status, cluster_output
- `experiments/`, `pruning/`, `models/`, `quantization/`, `evaluation/`, `data/`, `utils/`, `plotting/` — code
- `configs/` — six YAML configs (default, multi_seed_*, isic2019_n10, l3_cnn_n10, smoke_cpu, local_*)
- `scripts/slurm/job_*.sbatch` — 14 SLURM job files (14 of 14 carry `--requeue` + defensive env activation)
- `scripts/run_bundle_m_plus.sh` — Bundle M++ orchestrator
- `scripts/{diagnose_recovery,build_isic2018_segmasks,build_isic2019_metadata}.py` — data-staging helpers
- `results/logs_n10/aggregated/` — final aggregated CSVs feeding the report tables
- `MBZUAI_HPC_Docs.md`, `CODEX_PROJECT_SPEC.md`, `PROJECT_CONTEXT.md`, `README.md` — context docs
- Removed: presentation feedback / scripts / outline, `untitled.pen`, `report_old.tex`, LaTeX build artifacts (`*.aux/log/out/missfont`)

---

## One-Line Thesis
Medical compression in this project is not just a size problem. It is a safety problem: pruning, quantization, and distillation can fail asymmetrically across clinically dangerous vs safer skin-lesion classes, and aggregate accuracy can hide that.

## What The Project Is Solving
We want a compression recipe for dermoscopy classifiers that is actually suitable for edge deployment. The question is not "can we make the model smaller?" but "can we make it smaller without preferentially forgetting melanoma, BCC, and AKIEC?"

## What The Project Does
- Fine-tunes DeiT-Tiny, DeiT-Small, and a MobileNetV2 baseline on HAM10000.
- Uses lesion-grouped splits so images from the same lesion never leak across train and validation.
- Compares pruning criteria, non-uniform sparsity allocation, INT8 quantization, knowledge distillation, and external pruning baselines.
- Reports balanced accuracy, per-class sensitivity, macro AUROC, melanoma AUROC, ECE, DCR, and deployment-oriented regime labels.
- Runs the main pruning comparisons across 3 seeds with paired t-tests and mean +/- std reporting.

## Dataset And Split
HAM10000 has 10,015 dermoscopy images across 7 classes. The class imbalance is the core reason the project exists.

| Class | Count | Clinical role |
|---|---:|---|
| nv | 6,705 | safer / dominant benign class |
| mel | 1,113 | dangerous |
| bkl | 1,099 | safer |
| bcc | 514 | dangerous |
| akiec | 327 | dangerous |
| vasc | 142 | safer |
| df | 115 | safer |

- Dangerous classes are `mel`, `bcc`, and `akiec`.
- Safer classes are `nv`, `bkl`, `df`, and `vasc`.
- Splits use `GroupShuffleSplit` on `lesion_id` with an explicit overlap assertion.
- The split is lesion-grouped, not image-level, so the reported numbers are stricter than the old paper.

## Models

| Model | What it is | Approx size / role |
|---|---|---|
| DeiT-Tiny | Small distilled Vision Transformer from `timm` | Main small backbone; shows the strongest sensitivity-collapse behavior under Wanda |
| DeiT-Small | Larger distilled Vision Transformer from `timm` | Main backbone for headline pruning results; used as the teacher in distillation |
| MobileNetV2 | CNN baseline | Exploratory architecture check; shows CNN pruning fragility |

- DeiT-Small serves as the teacher for knowledge distillation.
- DeiT-Tiny is the student backbone that reveals the collapse behavior most sharply.
- MobileNetV2 is not the central claim; it is a sanity check that the problem is not unique to transformers.

## Compression Operations And Analysis Knobs

| Operation | What it does | Why it matters |
|---|---|---|
| Magnitude pruning | Keeps the largest weights by absolute value | The default baseline, and in this project the clinically dangerous one |
| Wanda pruning | Scores weights by `|w| * ||x||_2` | Activation-aware method that looks good on sensitivity but can collapse discrimination |
| Taylor pruning | Scores weights by `|w * dL/dw|` | Best clinical tradeoff in the main comparison |
| Random pruning | Uniform random scores | Lower-bound control; should collapse if the metric is honest |
| Recovery fine-tuning | Retrains pruned models for 5 / 10 / 20 epochs | Tests whether post-pruning recovery changes the story |
| Non-uniform allocation | Changes sparsity by layer sensitivity | Tests whether where you prune matters more than how you score |
| INT8 quantization | Dynamic post-training quantization | Tests the safest low-risk compression stage |
| Knowledge distillation | Soft-target KL distillation with `T=4`, `alpha=0.7` | Tests whether dense gains survive sparsity |
| External baselines | Paxton skewness, X-Pruner, SparseGPT-pseudo | Checks whether the main pruning story is really the best available one |
| Calibration-size ablation | Varies pruning-calibration size from 16 to 512 | Checks whether the pruning score sample budget matters materially |

## Methodology
1. Fine-tune the backbones on HAM10000 with lesion-grouped splits.
2. Score pruning weights on a calibration subset, usually 128 images.
3. Sweep pruning sparsity at 20%, 40%, 50%, 60%, and 70% for the main criteria.
4. Evaluate the 50% case in the headline table and use 3 seeds for the main comparisons.
5. Run paired t-tests for Wanda vs magnitude on the same seeds.
6. Compute balanced accuracy, per-class sensitivity, macro AUROC, melanoma AUROC, ECE, and DCR.
7. Test non-uniform allocation, recovery fine-tuning, quantization stacking, distillation, and external baselines.
8. Classify pipelines into clinical regimes when reporting deployment relevance.

## Metric Definitions
- Balanced accuracy: mean recall across classes.
- Melanoma AUROC: one-vs-rest melanoma discrimination.
- ECE: top-label calibration error.
- DCR: mean sensitivity drop on dangerous classes divided by mean sensitivity drop on safer classes.
- DCR > 1 means dangerous classes are forgotten faster.
- DCR < 1 means safer classes are forgotten faster.

## Results By Family

### 1) Dense baselines
- DeiT-Small dense: balanced acc 0.882, mel sens 0.897, mel AUROC 0.959, ECE 0.031.
- DeiT-Tiny dense: balanced acc 0.876, mel sens 0.837, mel AUROC 0.949, ECE 0.033.
- MobileNetV2 dense: balanced acc 0.766, mel sens 0.633, macro AUROC 0.953.
- Dense models are not the problem; compression is.

### 2) Headline pruning at 50%
- Magnitude is clinically dangerous: DCR 2.15 on DeiT-Small and 2.18 on DeiT-Tiny.
- Wanda flips the failure direction: DCR 0.71 on DeiT-Small and 0.57 on DeiT-Tiny.
- Taylor is the best clinical tradeoff on DeiT-Small: mel AUROC 0.928, ECE 0.033, DCR 0.80.
- On DeiT-Tiny, Wanda reaches mel sensitivity 0.925 but balanced acc only 0.336 and ECE 0.114. That is prediction collapse, not better diagnosis.
- Random pruning collapses, which is the sanity check that the metrics are actually sensitive.
- Wanda vs magnitude balanced accuracy is significant on both backbones, but melanoma sensitivity on DeiT-Small is not significant. The safety story is not just sensitivity.

### 3) Mechanism analysis
- The activation-outlier hypothesis does not hold.
- On DeiT-Small, correlations between outlier statistics and Wanda damage are near zero: Pearson and Spearman values stay below about 0.11 in magnitude.
- This is an explicit negative result, not a hidden one.

### 4) Non-uniform allocation and rescue
- Wanda uniform 50% on DeiT-Small: balanced acc 0.606, mel sens 0.566, DCR 0.72.
- Binned non-uniform allocation rescues Wanda to balanced acc 0.776 and mel sens 0.685, with DCR 1.49.
- Continuous-temperature allocation is the warning case: balanced acc 0.660 and DCR 5.78.
- X-Pruner is the strongest external 50% baseline on DeiT-Small for standard metrics: balanced acc 0.747, mel sens 0.746, DCR 1.58.
- On DeiT-Tiny, X-Pruner at 50% is also competitive and has DCR 0.34.
- The lesson is that sparsity placement matters, but not every non-uniform policy helps.

### 5) Recovery sweep
- Recovery epochs 5, 10, and 20 barely move magnitude or Taylor on DeiT-Small.
- Wanda on DeiT-Tiny is the only case that shows a visible mid-epoch bump, but it is unstable, so recovery is not the primary fix.
- The recovery figures are useful mostly as a "what does not rescue the model" check.

### 6) Quantization and pipeline
- Dense + INT8 keeps the dense clinical behavior essentially unchanged while shrinking the model and improving latency.
- The reported narrative is that INT8 is the safest first compression stage.
- Taylor-pruned + INT8 is the best compressed pipeline in the report narrative.
- Magnitude-pruned + INT8 is acceptable for lower-risk deployment, but not the best clinical tradeoff.
- Wanda-pruned + INT8 remains the weakest option.

### 7) Knowledge distillation
- Distillation improves the dense Tiny model.
- The improvement does not survive sparsity: the distilled pruned student loses the advantage.
- Distillation is therefore useful for dense + quantized deployment, not as a free pre-treatment before pruning.

### 8) MobileNetV2 baseline
- MobileNetV2 is much more fragile under pruning than the ViT backbones.
- At 50% sparsity, magnitude pruning drops balanced acc to 0.251 and Wanda to 0.143.
- This shows the issue is not just "transformers are weird"; some CNNs are also very brittle.

### 9) Edge latency and deployment
- Dense DeiT-Tiny on the reported host is 11.61 ms on CPU and 2.57 ms on MPS.
- The quantization story is about honest edge deployment, not only parameter count.
- Clinical regime labels are used to summarize whether a pipeline is still deployment-worthy.

### 10) Calibration-size ablation
- Calibration size from 16 to 512 changes the headline less than the choice of pruning criterion.
- The middle sizes are a reasonable operating point; 128 is the default calibration budget in the main experiments.

## Slide Map And Evidence Files

| Slide | Core message | Table files | Figure files |
|---|---|---|---|
| 1 | Hook and title | - | - |
| 2 | Imbalance and DCR | - | - |
| 3 | Models and methodology | - | - |
| 4 | Headline pruning at 50% | `tables/pruning_headline.md`, `tables/paired_tests.md` | `../../results/figures_personal/fig_mel_sens_vs_auroc.png` |
| 5 | DCR inversion | `tables/pruning_headline.md` | `../../results/figures_personal/fig_dcr_bars.png` |
| 6 | Sensitivity collapse | `tables/pruning_headline.md`, `tables/paired_tests.md` | `../../results/figures_personal/fig_mel_sens_vs_auroc.png`, `../../results/figures_local/fig9_calibration_ablation.pdf` |
| 7 | Rescues and warning | `tables/nonuniform.md`, `tables/baselines.md`, `tables/recovery_sweep.md` | `../../results/figures_local/fig4_nonuniform_vs_uniform.pdf`, `../../results/figures_local/fig3_perlayer_bars.pdf`, `../../results/figures_local/fig7_recovery_balacc.pdf`, `../../results/figures_local/fig7_recovery_mel.pdf` |
| 8 | Deployment and takeaways | `../../results/logs_personal/aggregated/agg_quantization_stacking.csv`, `tables/mobilenetv2_baseline.md`, `tables/edge_latency.md` | `../../results/figures_local/fig5_stacking.pdf`, `../../results/figures_local/fig6_kd_pretreatment.pdf`, `../../results/figures_local/fig8_attention_maps.pdf` |

## Quick Numbers To Keep In Your Head
- Magnitude DCR: 2.15 on DeiT-Small, 2.18 on DeiT-Tiny.
- Wanda DCR: 0.71 on DeiT-Small, 0.57 on DeiT-Tiny.
- Taylor mel AUROC: 0.928 on DeiT-Small.
- Wanda on DeiT-Tiny: mel sens 0.925, balanced acc 0.336, ECE 0.114.
- Wanda vs magnitude balanced acc p-value on DeiT-Small: 0.007.
- Binned rescue on Wanda: balanced acc +0.17 on the main slide narrative.
- Continuous-temp DCR: 5.78.
- X-Pruner 50% on DeiT-Small: balanced acc 0.747, mel sens 0.746, DCR 1.58.
- Dense + INT8: about 3.8x smaller and about 30% faster in the report narrative.
- MobileNetV2 50% magnitude: balanced acc 0.251.

## Full Result Files
- Tables: `report/generated/tables/*.md`, `report/generated/tables/*.csv`, `report/generated/tables/*.tex`
- Aggregates: `results/logs_personal/aggregated/agg_*.csv`
- Figures: `results/figures_personal/*.png`, `results/figures_personal/*.pdf`, `results/figures_local/*.pdf`
- The narrative source is `report/does_compression_forget_cancer_revised.tex`
