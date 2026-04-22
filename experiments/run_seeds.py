"""
Drive the full experiment pipeline across a seed sweep (W1).

Each seed gets its own results/checkpoints/figures subdirectories via
`utils.config.apply_seed_to_paths`, so the final `evaluation.aggregate`
step can stitch mean ± std tables and paired t-tests across seeds
without any bespoke path munging.

Usage — single-GPU / single-process:
    python -m experiments.run_seeds --config configs/multi_seed_ciai.yaml

Usage — 4-GPU DDP (CIAI):
    torchrun --nproc_per_node=4 -m experiments.run_seeds \
        --config configs/multi_seed_ciai.yaml

Config-driven controls:
    experiment.seeds                  -> list[int], the sweep
    experiment.pillars                -> default pillar list
    experiment.seed_zero_only_pillars -> pillars that run only on the
                                         first seed (e.g., latency
                                         benchmarks that don't vary
                                         across training seeds).

When `--seeds` is passed on the CLI, only those seeds are run, but the
seed-zero gating still uses the first seed in the *config*'s seed list
so distributed jobs covering different seeds stay consistent about
which seed owns the run-once outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.aggregate import run as aggregate_run
from experiments import run_all
from utils.config import load_config


def _resolve_pillars_for_seed(
    pillars: list[int],
    seed: int,
    canonical_first_seed: int | None,
    seed_zero_only: list[int],
    share_finetune: bool = False,
) -> list[int]:
    """
    `seed_zero_only` pillars run only on the first seed. `share_finetune`,
    when True, treats pillar 0 (fine-tune + baseline eval + calibration)
    as a one-shot step — all later seeds skip it and reuse the
    checkpoint + calibration tensors written during the first seed. This
    is the personal-compute mode where you can only afford to train
    each backbone once but still want multiple seeds' worth of pruning /
    evaluation variance.
    """
    exclude: set[int] = set()
    if canonical_first_seed is not None and seed != canonical_first_seed:
        exclude |= set(seed_zero_only)
        if share_finetune:
            exclude.add(0)
    return [p for p in pillars if p not in exclude]


def run(
    config_path: str,
    pillars: list[int] | None = None,
    seeds: list[int] | None = None,
    aggregate: bool = True,
) -> None:
    config = load_config(config_path)
    experiment_cfg = config.get("experiment", {}) or {}
    config_seeds = list(experiment_cfg.get("seeds") or [experiment_cfg.get("seed", 42)])
    if seeds is None:
        seeds = config_seeds
    if pillars is None:
        pillars = list(experiment_cfg.get("pillars") or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    seed_zero_only = list(experiment_cfg.get("seed_zero_only_pillars") or [])
    share_finetune = bool(experiment_cfg.get("share_finetune_across_seeds", False))
    canonical_first_seed = config_seeds[0] if config_seeds else None

    results_root = Path(config["logging"]["results_dir"])

    for seed in seeds:
        seed_pillars = _resolve_pillars_for_seed(
            pillars,
            int(seed),
            canonical_first_seed,
            seed_zero_only,
            share_finetune=share_finetune,
        )
        skipped = sorted(set(pillars) - set(seed_pillars))
        suffix = f" (skipped pillars {skipped})" if skipped else ""
        print(f"\n=== seed {seed}{suffix} ===")
        run_all.run(config_path, seed_pillars, seed_override=int(seed))

    if aggregate:
        print("\n=== aggregating across seeds ===")
        aggregate_run(results_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed experiment driver.")
    parser.add_argument("--config", default="configs/multi_seed_ciai.yaml")
    parser.add_argument(
        "--pillars",
        nargs="*",
        type=int,
        default=None,
        help="Override the pillar list (otherwise read from experiment.pillars).",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, pillars=args.pillars, seeds=args.seeds, aggregate=not args.no_aggregate)


if __name__ == "__main__":
    main()
