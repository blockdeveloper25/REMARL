"""
remarl/eval/metrics.py
----------------------
Evaluation metrics comparing REMARL vs baseline MARE.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from scipy import stats


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




def compare_with_significance(
    remarl_results: list,
    baseline_results: list,
) -> dict:
    """
    Run paired t-test comparing REMARL vs MARE baseline.
    Both lists must be equal length — each pair is the same scenario run twice.

    Returns dict with mean scores, delta, p-value, Cohen's d effect size.
    """
    r_rewards = [r.total_reward for r in remarl_results]
    b_rewards = [r.total_reward for r in baseline_results]

    r_coverage = [r.coverage_score for r in remarl_results]
    b_coverage = [r.coverage_score for r in baseline_results]

    # Paired t-test (same scenarios run under both policies)
    t_stat, p_value = stats.ttest_rel(r_rewards, b_rewards)
    t_cov,  p_cov   = stats.ttest_rel(r_coverage, b_coverage)

    # Cohen's d effect size
    diff = np.array(r_rewards) - np.array(b_rewards)
    cohen_d = diff.mean() / (diff.std() + 1e-8)

    mean_r = float(np.mean(r_rewards))
    mean_b = float(np.mean(b_rewards))

    result = {
        "n": len(r_rewards),
        "remarl_mean_reward":   round(mean_r, 4),
        "baseline_mean_reward": round(mean_b, 4),
        "delta_reward":         round(mean_r - mean_b, 4),
        "p_value":              round(p_value, 4),
        "significant":          p_value < 0.05,
        "cohen_d":              round(cohen_d, 3),
        "effect_size":          "large" if abs(cohen_d) > 0.8 else
                                "medium" if abs(cohen_d) > 0.5 else "small",
        "coverage_p_value":     round(p_cov, 4),
        "remarl_mean_coverage": round(float(np.mean(r_coverage)), 4),
        "baseline_mean_coverage": round(float(np.mean(b_coverage)), 4),
    }

    # Print
    print(f"\n{'─'*60}")
    print(f"  REMARL vs MARE Baseline — Statistical Comparison")
    print(f"{'─'*60}")
    print(f"  Episodes evaluated : {result['n']}")
    print(f"  REMARL mean reward : {result['remarl_mean_reward']}")
    print(f"  MARE mean reward   : {result['baseline_mean_reward']}")
    print(f"  Delta              : {result['delta_reward']:+.4f}")
    print(f"  p-value            : {result['p_value']}  "
          f"{'✓ SIGNIFICANT' if result['significant'] else '✗ not significant'}")
    print(f"  Cohen's d          : {result['cohen_d']}  ({result['effect_size']} effect)")
    print(f"  Coverage p-value   : {result['coverage_p_value']}")
    print(f"{'─'*60}\n")

    return result
