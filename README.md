# TinyML Medical ViT Compression

Initial implementation scaffold for the project described in:

- `TinyML_Enhanced_Final.docx.md`
- `CODEX_PROJECT_SPEC.md`

This repository currently includes:

- dataset preprocessing and loading utilities for HAM10000
- DeiT model loading helpers
- pruning hooks, scoring, masking, and non-uniform allocation utilities
- evaluation, model size, latency, and quantization helpers
- knowledge distillation loss and training loop
- experiment entrypoints for fine-tuning, baseline evaluation, calibration, pruning matrix, and master orchestration

The code is written so later experiment and plotting modules can build on the same shared utilities.
