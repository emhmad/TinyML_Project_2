# Project Context

This file is a full handoff document for the TinyML medical compression project. It summarizes what the project is, what has been implemented, what results already exist, and what remains intentionally out of scope at the current stopping point.

## 1. Project Summary

Project title:

`TinyML Medical ViT Compression for Diagnostic-Safe Edge Dermatology`

Core question:

Can model compression make medical image classifiers small enough for edge deployment without disproportionately harming clinically important classes such as melanoma?

Dataset:

- HAM10000
- 7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

Main model family:

- `DeiT-Tiny`
- `DeiT-Small`

Additional baseline:

- `MobileNetV2`

Primary metrics:

- balanced accuracy
- melanoma sensitivity
- per-class sensitivity for all 7 classes
- model size
- CPU latency

## 2. Source Documents That Started the Project

The project was built from these planning/spec files:

- `TinyML_Enhanced_Final.docx.md`
- `CODEX_PROJECT_SPEC.md`
- report revision docs used later in development
- improvement-spec work that was implemented up to the MobileNetV2 baseline stage

## 3. What Has Been Implemented

### Core experiment pipeline

- HAM10000 metadata preparation and dataset loading
- Fine-tuning of DeiT-Tiny and DeiT-Small
- Dense baseline evaluation
- Calibration artifact generation for Wanda and Taylor
- One-shot pruning matrix with `magnitude`, `wanda`, `taylor`, and `random`
- Per-layer pruning breakdown
- Diagnostic safety analysis across all 7 classes
- Sensitivity-guided non-uniform pruning
- INT8 quantization and pruning-plus-quantization stacking
- Knowledge distillation from DeiT-Small to DeiT-Tiny
- Plotting scripts for the first six report figures
- Final LaTeX report integrating the completed base results

### Improvement work implemented later

- Post-pruning recovery finetuning at 50% sparsity
- Recovery plots
- Attention rollout visualization figure
- Wanda calibration-set-size ablation and plot
- MobileNetV2 dense vs pruned baseline
- Report updated to include these implemented improvements

## 4. Improvement Items Not Implemented

These were intentionally left unfinished at the current stopping point:

- multi-seed experiments
- structured pruning / attention-head pruning

This is acceptable for the current project scope. The existing implementation is still strong enough for GitHub, report submission, and presentation.

## 5. Important Files and Where to Look

### Configuration

- `configs/default.yaml`
- `configs/local_3060ti.yaml`
- `configs/smoke_cpu.yaml`

### Experiment entrypoints

- `experiments/run_all.py`
- `experiments/e1_finetune.py`
- `experiments/e2_baseline_eval.py`
- `experiments/e3_calibration.py`
- `experiments/e4_pruning_matrix.py`
- `experiments/e5_perlayer_breakdown.py`
- `experiments/e6_diagnostic_safety.py`
- `experiments/e7_e10_nonuniform.py`
- `experiments/e11_e13_quantization.py`
- `experiments/e14_e16_distillation.py`
- `experiments/e_recovery_finetune.py`
- `experiments/e_attention_viz.py`
- `experiments/e_calibration_ablation.py`
- `experiments/e_mobilenet_baseline.py`

### Plotting

- `plotting/fig1_melanoma_sensitivity.py`
- `plotting/fig2_balanced_accuracy.py`
- `plotting/fig3_perlayer_bars.py`
- `plotting/fig4_nonuniform_vs_uniform.py`
- `plotting/fig5_stacking.py`
- `plotting/fig6_kd_pretreatment.py`
- `plotting/fig7_recovery.py`
- `plotting/fig9_calibration_ablation.py`

### Report

- `report/final_report.tex`
- `report/references.bib`

## 6. Run Grouping in `experiments/run_all.py`

The project uses pillar IDs:

- `0`: fine-tuning, baseline evaluation, calibration
- `1`: pruning matrix, per-layer breakdown, diagnostic safety
- `2`: non-uniform pruning
- `3`: quantization stacking
- `4`: distillation
- `5`: recovery finetuning
- `6`: calibration ablation and attention visualization
- `7`: MobileNetV2 baseline

## 7. Existing Final Artifacts

### Logs already generated

Located under `results/logs_local/`:

- `attention_viz_samples.csv`
- `baseline_eval.csv`
- `calibration_ablation.csv`
- `diagnostic_safety.csv`
- `distillation_history.csv`
- `finetune_history.csv`
- `kd_pretreatment.csv`
- `layer_sensitivity_scores.csv`
- `mobilenet_results.csv`
- `nonuniform_allocation.csv`
- `perlayer_breakdown.csv`
- `pruning_matrix.csv`
- `quantization_stacking.csv`
- `recovery_finetune.csv`

### Figures already generated

Located under `results/figures_local/`:

- `fig1_melanoma_sensitivity.pdf`
- `fig2_balanced_accuracy.pdf`
- `fig3_perlayer_bars.pdf`
- `fig4_nonuniform_vs_uniform.pdf`
- `fig5_stacking.pdf`
- `fig6_kd_pretreatment.pdf`
- `fig7_recovery_balacc.pdf`
- `fig7_recovery_mel.pdf`
- `fig8_attention_maps.pdf`
- `fig9_calibration_ablation.pdf`
- `attention/` PNG panels for figure 8

### Checkpoints already generated

Located under `results/checkpoints_local/`:

- dense DeiT checkpoints
- distilled Tiny checkpoint
- calibration artifacts
- pruning masks
- recovery masks
- recovery finetuned checkpoints
- MobileNetV2 baseline checkpoint

## 8. Main Results Snapshot

### Dense baselines

| Model | Balanced Accuracy | Melanoma Sensitivity |
|---|---:|---:|
| DeiT-Tiny | 0.840 | 0.758 |
| DeiT-Small | 0.862 | 0.753 |

### 50% sparsity on DeiT-Small

| Criterion | Balanced Accuracy | Melanoma Sensitivity |
|---|---:|---:|
| Magnitude | 0.813 | 0.637 |
| Wanda | 0.685 | 0.444 |
| Taylor | 0.801 | 0.677 |
| Random | much worse than the above | much worse than the above |

### Non-uniform pruning on DeiT-Small

| Setup | Balanced Accuracy | Melanoma Sensitivity |
|---|---:|---:|
| Magnitude uniform 50% | 0.813 | 0.637 |
| Magnitude non-uniform | 0.784 | 0.637 |
| Wanda uniform 50% | 0.685 | 0.444 |
| Wanda non-uniform | 0.811 | 0.632 |

### Quantization stacking on DeiT-Small

| Setup | Balanced Accuracy | Melanoma Sensitivity | Size |
|---|---:|---:|---:|
| Dense | 0.862 | 0.753 | 84.7 MB |
| Quantized only | 0.862 | 0.753 | 22.5 MB |
| Magnitude 50% | 0.813 | 0.637 | 43.2 MB |
| Magnitude 50% + INT8 | 0.804 | 0.632 | 22.5 MB |
| Wanda 50% | 0.685 | 0.444 | 43.2 MB |
| Wanda 50% + INT8 | 0.666 | 0.435 | 22.5 MB |

### Recovery finetuning at 50% sparsity

| Model | Criterion | One-shot Balanced Acc. | Recovered Balanced Acc. | One-shot Mel | Recovered Mel |
|---|---|---:|---:|---:|---:|
| DeiT-Small | Magnitude | 0.813 | 0.873 | 0.637 | 0.776 |
| DeiT-Small | Wanda | 0.685 | 0.838 | 0.444 | 0.758 |
| DeiT-Small | Taylor | 0.801 | 0.858 | 0.677 | 0.821 |
| DeiT-Tiny | Magnitude | 0.804 | 0.845 | 0.677 | 0.789 |
| DeiT-Tiny | Wanda | 0.614 | 0.794 | 0.556 | 0.717 |
| DeiT-Tiny | Taylor | 0.720 | 0.836 | 0.538 | 0.700 |

### Calibration ablation for Wanda on DeiT-Small 50%

| Calibration Size | Balanced Accuracy | Melanoma Sensitivity |
|---|---:|---:|
| 16 | 0.673 | 0.457 |
| 32 | 0.664 | 0.417 |
| 64 | 0.669 | 0.439 |
| 128 | 0.676 | 0.444 |
| 256 | 0.687 | 0.448 |
| 512 | 0.693 | 0.466 |

Interpretation:

- More calibration data helps Wanda slightly.
- Even the best tested calibration size still remains well below magnitude pruning at the same sparsity.

### MobileNetV2 baseline

| Setup | Balanced Accuracy | Melanoma Sensitivity |
|---|---:|---:|
| Dense | 0.838 | 0.686 |
| Magnitude 50% | 0.499 | 0.121 |
| Wanda 50% | 0.170 | 0.000 |

Interpretation:

- Wanda underperforming magnitude is not limited to DeiT models in this project.
- The same failure pattern appears in the CNN baseline as well.

## 9. Main Narrative of the Project

The results support the following story:

- Compression can reduce model size substantially, but clinical safety metrics must be tracked explicitly.
- In this setup, Wanda does not protect melanoma sensitivity better than simpler pruning criteria.
- Magnitude and Taylor are safer defaults than Wanda for this project.
- Non-uniform sparsity allocation is one of the strongest corrective interventions.
- Quantization is relatively safe and practical.
- Recovery finetuning is extremely effective and should be highlighted in any presentation.
- The MobileNetV2 baseline strengthens the claim that Wanda's weakness here is not just a ViT-specific artifact.

## 10. Known Technical Notes

- The repo was originally inside OneDrive and later moved to `C:\Projects\TinyML_Project` to avoid training slowdowns from sync/I-O overhead.
- Some historical metadata entries contained older absolute OneDrive image paths.
- `data/dataset.py` was updated so dataset loading falls back to the moved local dataset directory automatically.
- `experiments/e_attention_viz.py` also prefers the moved local dataset path.
- `experiments/e_recovery_finetune.py` was updated to support resuming interrupted recovery runs safely.
- `experiments/e_mobilenet_baseline.py` was fixed for grouped/depthwise-convolution Wanda scoring and now reuses an existing MobileNet checkpoint if available.

## 11. Report Status

The LaTeX report has already been updated to include:

- recovery finetuning
- attention visualization
- calibration ablation
- MobileNetV2 baseline

Main report files:

- `report/final_report.tex`
- `report/references.bib`

Compiled PDF reviewed previously:

- structurally good
- bibliography and added sections present
- main visible cleanup item noted earlier: replace author placeholders if still present in the source

## 12. Presentation Status

At the current stopping point, the project is presentation-ready.

Why:

- strong main experiment matrix
- real figures already generated
- multiple follow-up analyses beyond the base requirement
- a clear, defensible final narrative
- both quantitative and qualitative evidence

## 13. Recommended Next Actions If Continuing Later

If someone resumes work later, the best next options are:

1. polish the README/report/presentation rather than adding more experiments
2. run multi-seed validation if additional rigor is needed
3. add structured pruning only if deployment-speed realism becomes a grading requirement
4. replace any placeholder names in the report before final submission

## 14. Quick Start Commands

Setup:

```powershell
cd C:\Projects\TinyML_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Smoke test:

```powershell
python -m experiments.run_all --config configs/smoke_cpu.yaml --pillars 0 1
```

Main local run:

```powershell
python -m experiments.run_all --config configs/local_3060ti.yaml --pillars 0
python -m experiments.run_all --config configs/local_3060ti.yaml --pillars 1 2 3 4
```

Improvement runs:

```powershell
python -m experiments.e_recovery_finetune --config configs/local_3060ti.yaml
python -m experiments.run_all --config configs/local_3060ti.yaml --pillars 6
python -m experiments.run_all --config configs/local_3060ti.yaml --pillars 7
```

Figure generation:

```powershell
python -m plotting.fig1_melanoma_sensitivity --config configs/local_3060ti.yaml
python -m plotting.fig2_balanced_accuracy --config configs/local_3060ti.yaml
python -m plotting.fig3_perlayer_bars --config configs/local_3060ti.yaml
python -m plotting.fig4_nonuniform_vs_uniform --config configs/local_3060ti.yaml
python -m plotting.fig5_stacking --config configs/local_3060ti.yaml
python -m plotting.fig6_kd_pretreatment --config configs/local_3060ti.yaml
python -m plotting.fig7_recovery --config configs/local_3060ti.yaml
python -m plotting.fig9_calibration_ablation --config configs/local_3060ti.yaml
```

Report compile:

```powershell
cd report
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```
