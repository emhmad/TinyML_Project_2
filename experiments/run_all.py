from __future__ import annotations

import argparse

from experiments import (
    e_activation_stats,
    e_attention_viz,
    e_baselines_paxton_xpruner,
    e_calibration_ablation,
    e_edge_latency,
    e_mobilenet_baseline,
    e_recovery_finetune,
    e_resnet50_baseline,
    e_structured_sparsity,
    e1_finetune,
    e2_baseline_eval,
    e3_calibration,
    e4_pruning_matrix,
    e5_perlayer_breakdown,
    e6_diagnostic_safety,
    e7_e10_nonuniform,
    e11_e13_quantization,
    e14_e16_distillation,
)


def run(config_path: str, pillars: list[int], seed_override: int | None = None) -> None:
    """
    Pillar layout:

      0  finetune + baseline eval + calibration
      1  pruning matrix + per-layer breakdown + diagnostic safety
      2  non-uniform allocation (binned + continuous + OBS + learnable; W11)
      3  quantization + stacking + p95 latency                      (W9)
      4  distillation pre-treatment
      5  recovery fine-tune sweep                                    (W10)
      6  calibration-size ablation + attention overlap (W12)
      7  MobileNetV2 baseline
      8  ResNet-50 baseline                                          (W7)
      9  activation statistics + Paxton/X-Pruner baselines           (W4 + W6)
      10 2:4 / N:M structured sparsity                               (W8)
      11 edge-target latency sweep                                   (W9)
    """
    if 0 in pillars:
        e1_finetune.run(config_path, seed_override=seed_override)
        e2_baseline_eval.run(config_path, seed_override=seed_override)
        e3_calibration.run(config_path, seed_override=seed_override)

    if 1 in pillars:
        e4_pruning_matrix.run(config_path, seed_override=seed_override)
        e5_perlayer_breakdown.run(config_path, seed_override=seed_override)
        e6_diagnostic_safety.run(config_path, seed_override=seed_override)

    if 2 in pillars:
        e7_e10_nonuniform.run(config_path, seed_override=seed_override)

    if 3 in pillars:
        e11_e13_quantization.run(config_path, seed_override=seed_override)

    if 4 in pillars:
        e14_e16_distillation.run(config_path, seed_override=seed_override)

    if 5 in pillars:
        e_recovery_finetune.run(config_path, seed_override=seed_override)

    if 6 in pillars:
        e_calibration_ablation.run(config_path, seed_override=seed_override)
        e_attention_viz.run(config_path, seed_override=seed_override)

    if 7 in pillars:
        e_mobilenet_baseline.run(config_path, seed_override=seed_override)

    if 8 in pillars:
        e_resnet50_baseline.run(config_path, seed_override=seed_override)

    if 9 in pillars:
        e_activation_stats.run(config_path, seed_override=seed_override)
        e_baselines_paxton_xpruner.run(config_path, seed_override=seed_override)

    if 10 in pillars:
        e_structured_sparsity.run(config_path, seed_override=seed_override)

    if 11 in pillars:
        e_edge_latency.run(config_path, seed_override=seed_override)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the implemented TinyML experiment pillars.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pillars", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.pillars, seed_override=args.seed)


if __name__ == "__main__":
    main()
