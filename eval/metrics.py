"""
remarl/eval/metrics.py
----------------------
Evaluation metrics comparing REMARL vs baseline MARE.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class EvalResult:
    mean_coverage:     float
    mean_precision:    float
    mean_total_reward: float
    mean_steps:        float
    std_reward:        float
    n_episodes:        int


def aggregate_oracle_results(results: list) -> EvalResult:
    rewards    = [r.total_reward    for r in results]
    coverages  = [r.coverage_score  for r in results]
    precisions = [r.precision_score for r in results]
    return EvalResult(
        mean_coverage=float(np.mean(coverages)),
        mean_precision=float(np.mean(precisions)),
        mean_total_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_steps=0.0,
        n_episodes=len(results),
    )


def print_comparison(remarl: EvalResult, baseline: EvalResult):
    delta = remarl.mean_total_reward - baseline.mean_total_reward
    print(f"\n{'─'*50}")
    print(f"{'Metric':<25} {'REMARL':>10} {'Baseline':>10} {'Delta':>10}")
    print(f"{'─'*50}")
    print(f"{'Mean total reward':<25} {remarl.mean_total_reward:>10.4f} {baseline.mean_total_reward:>10.4f} {delta:>+10.4f}")
    print(f"{'Mean coverage':<25} {remarl.mean_coverage:>10.4f} {baseline.mean_coverage:>10.4f} {remarl.mean_coverage - baseline.mean_coverage:>+10.4f}")
    print(f"{'Mean precision':<25} {remarl.mean_precision:>10.4f} {baseline.mean_precision:>10.4f} {remarl.mean_precision - baseline.mean_precision:>+10.4f}")
    print(f"{'─'*50}\n")
