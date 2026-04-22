# TinyML Medical ViT Compression for Diagnostic-Safe Edge Dermatology

This repository contains a full TinyML-style medical imaging project on compressing skin-lesion classifiers for edge deployment while tracking diagnostic safety, especially melanoma sensitivity.

The project studies whether compression causes medical Vision Transformers to forget dangerous classes before benign ones. The core pipeline fine-tunes DeiT-Tiny and DeiT-Small on HAM10000, compares one-shot pruning criteria, evaluates non-uniform sparsity allocation, stacks INT8 quantization, tests knowledge distillation, and extends the analysis with recovery finetuning, attention visualization, calibration ablation, and a MobileNetV2 baseline.

## Project Question

Can we compress medical image classifiers enough for practical edge deployment without disproportionately harming high-risk diagnostic classes such as melanoma?

## What Is Implemented

### Core experiments

- Dense fine-tuning of `DeiT-Tiny` and `DeiT-Small` on HAM10000
- Dense baseline evaluation with balanced accuracy and per-class sensitivity
- Calibration artifact generation for Wanda and Taylor pruning
- One-shot unstructured pruning with `magnitude`, `wanda`, `taylor`, and `random`
- Per-layer sensitivity breakdown across attention/MLP groups
- Diagnostic safety analysis across all 7 HAM10000 classes
- Sensitivity-guided non-uniform pruning allocation
- Dynamic INT8 post-training quantization and pruning-plus-quantization stacking
- Knowledge distillation from `DeiT-Small` to `DeiT-Tiny`

### Improvement experiments completed

- Post-pruning recovery finetuning at 50% sparsity
- Attention rollout visualization for dense vs pruned DeiT-Small
- Wanda calibration-set-size ablation
- MobileNetV2 CNN baseline with dense, magnitude-pruned, and Wanda-pruned comparisons

### Improvement items intentionally not implemented

- Multi-seed runs
- Structured pruning / attention-head pruning

## Main Findings

These numbers come from the completed local run under `results/logs_local/`.

- `DeiT-Small` is the strongest dense baseline: balanced accuracy `0.862`, melanoma sensitivity `0.753`.
- At 50% sparsity on `DeiT-Small`, `magnitude` pruning preserves performance much better than `wanda`:
  - magnitude: balanced accuracy `0.813`, melanoma sensitivity `0.637`
  - wanda: balanced accuracy `0.685`, melanoma sensitivity `0.444`
- `Taylor` is competitive and gives the best melanoma sensitivity at 50% sparsity on `DeiT-Small`: `0.677`.
- Non-uniform pruning substantially rescues Wanda on `DeiT-Small`:
  - uniform Wanda: balanced accuracy `0.685`, melanoma sensitivity `0.444`
  - non-uniform Wanda: balanced accuracy `0.811`, melanoma sensitivity `0.632`
- Quantization-only is nearly lossless on `DeiT-Small`:
  - dense: balanced accuracy `0.862`, size `84.7 MB`
  - quantized: balanced accuracy `0.862`, size `22.5 MB`
- Recovery finetuning is highly effective. For example, `DeiT-Small + Wanda 50%` improves from balanced accuracy `0.685` to `0.838` after recovery finetuning, and melanoma sensitivity improves from `0.444` to `0.758`.
- The MobileNetV2 baseline shows the same trend: Wanda underperforms magnitude there too.

## Repository Layout

```text
configs/        Experiment configurations (`default`, `local_3060ti`, `smoke_cpu`)
data/           HAM10000 preprocessing and dataset loading
experiments/    Training and evaluation entrypoints
models/         Model-loading and distillation helpers
pruning/        Pruning hooks, scoring, masking, and non-uniform allocation
quantization/   INT8 quantization helpers
evaluation/     Metrics, model size, and latency utilities
plotting/       Scripts for all report figures
results/        Generated checkpoints, CSV logs, and figures
report/         Final LaTeX report and bibliography
```

## Environment Setup

Recommended setup on Windows PowerShell:

```powershell
cd C:\Projects\TinyML_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Main dependencies are listed in `requirements.txt`:

- `torch`
- `torchvision`
- `timm`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `Pillow`
- `PyYAML`
- `tqdm`

## Dataset Preparation

This project uses the [HAM10000 dataset](https://doi.org/10.7910/DVN/DBW86T).

After downloading and extracting HAM10000 locally, build the processed metadata file with:

```powershell
python -m data.download_ham10000 --source-dir <PATH_TO_RAW_HAM10000> --output-dir data/ham10000
```

Expected output:

- `data/ham10000/processed_metadata.csv`

Notes:

- The processed metadata stores image paths for all 7 classes.
- `data/dataset.py` includes a fallback so older absolute image paths can still resolve after moving the repo.

## Configurations

- `configs/default.yaml`: larger default training setup
- `configs/local_3060ti.yaml`: local GPU run used for the final project results
- `configs/smoke_cpu.yaml`: tiny smoke/debug config for quick validation

## Running the Project

### 1. Smoke test

```powershell
python -m experiments.run_all --config configs/smoke_cpu.yaml --pillars 0 1
```

### 2. Main local run

The completed project results in this repo were generated with the local 3060 Ti config.

```powershell
python -m experiments.run_all --config configs/local_3060ti.yaml --pillars 0
python -m experiments.run_all --config configs/local_3060ti.yaml --pillars 1 2 3 4
```

### 3. Improvement experiments

Recovery finetuning:

```powershell
python -m experiments.e_recovery_finetune --config configs/local_3060ti.yaml
```

Calibration ablation and attention visualization:

```powershell
python -m experiments.run_all --config configs/local_3060ti.yaml --pillars 6
```

MobileNetV2 baseline:

```powershell
python -m experiments.run_all --config configs/local_3060ti.yaml --pillars 7
```

### 4. Generate figures

Core figures:

```powershell
python -m plotting.fig1_melanoma_sensitivity --config configs/local_3060ti.yaml
python -m plotting.fig2_balanced_accuracy --config configs/local_3060ti.yaml
python -m plotting.fig3_perlayer_bars --config configs/local_3060ti.yaml
python -m plotting.fig4_nonuniform_vs_uniform --config configs/local_3060ti.yaml
python -m plotting.fig5_stacking --config configs/local_3060ti.yaml
python -m plotting.fig6_kd_pretreatment --config configs/local_3060ti.yaml
```

Improvement figures:

```powershell
python -m plotting.fig7_recovery --config configs/local_3060ti.yaml
python -m plotting.fig9_calibration_ablation --config configs/local_3060ti.yaml
```

Generated PDFs are written to `results/figures_local/`.

### 5. Compile the report

```powershell
cd report
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

## Orchestration Map

`experiments/run_all.py` groups experiments into pillars:

- `0`: fine-tuning, dense baseline evaluation, calibration
- `1`: pruning matrix, per-layer breakdown, diagnostic safety
- `2`: non-uniform pruning allocation
- `3`: quantization stacking
- `4`: distillation experiments
- `5`: recovery finetuning
- `6`: calibration ablation and attention visualization
- `7`: MobileNetV2 baseline

## Key Output Files

### Logs

- `results/logs_local/baseline_eval.csv`
- `results/logs_local/pruning_matrix.csv`
- `results/logs_local/perlayer_breakdown.csv`
- `results/logs_local/nonuniform_allocation.csv`
- `results/logs_local/quantization_stacking.csv`
- `results/logs_local/kd_pretreatment.csv`
- `results/logs_local/recovery_finetune.csv`
- `results/logs_local/calibration_ablation.csv`
- `results/logs_local/mobilenet_results.csv`

### Figures

- `results/figures_local/fig1_melanoma_sensitivity.pdf`
- `results/figures_local/fig2_balanced_accuracy.pdf`
- `results/figures_local/fig3_perlayer_bars.pdf`
- `results/figures_local/fig4_nonuniform_vs_uniform.pdf`
- `results/figures_local/fig5_stacking.pdf`
- `results/figures_local/fig6_kd_pretreatment.pdf`
- `results/figures_local/fig7_recovery_balacc.pdf`
- `results/figures_local/fig7_recovery_mel.pdf`
- `results/figures_local/fig8_attention_maps.pdf`
- `results/figures_local/fig9_calibration_ablation.pdf`

### Report

- `report/final_report.tex`
- `report/references.bib`

## Reproducibility Notes

- Current reported results are from a single-seed run.
- The final local experiments were executed after moving the repo out of OneDrive for faster I/O.
- `processed_metadata.csv` may contain older absolute image paths, but the dataset loader includes a local fallback to keep the moved repo working.
- Recovery finetuning is resume-safe and skips already completed checkpoints when rerun.
- The MobileNetV2 baseline reuses an existing dense checkpoint if present.

## Current Project Status

This repository is beyond a scaffold: it contains the complete code, logs, figures, and LaTeX report for the implemented project scope.

At the current stopping point, the project is strong for:

- GitHub submission
- final report submission
- project presentation

The main unfinished research extensions are multi-seed validation and structured pruning, but the implemented package already supports a solid and defensible final project narrative.

## Additional Context

A full handoff and status summary is available in `PROJECT_CONTEXT.md`.
