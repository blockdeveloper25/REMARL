"""
remarl/eval/benchmark.py
------------------------
Runs REMARL-trained policies against vanilla MARE baseline
on held-out evaluation scenarios and prints a comparison table.

Usage:
    python eval/benchmark.py --checkpoint data/checkpoints/collector_final
    python eval/benchmark.py --checkpoint data/checkpoints/collector_final --domain patient_portal
"""

import argparse
import logging
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("remarl.eval")


def run_eval_episode(env, model=None) -> dict:
    """Run one eval episode. If model=None, uses random actions (baseline proxy)."""
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    oracle_result = None

    for _ in range(env.max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, step_info = env.step(int(action))
        total_reward += reward
        steps += 1

        if step_info.get("oracle_result"):
            oracle_result = step_info["oracle_result"]

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "oracle": oracle_result,
    }


def benchmark(config_path: str, checkpoint_path: str, n_eval: int, domain: str = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from sim.scenario_gen import ScenarioGenerator
    from sim.oracle import Oracle
    from sim.re_env import RESimEnv, AGENT_ACTION_MAP
    from rl.state_encoder import StateEncoder
    from rl.reward import RewardEngine
    from eval.metrics import aggregate_oracle_results, print_comparison

    gen     = ScenarioGenerator(config["env"]["scenario_dir"])
    oracle  = Oracle()
    encoder = StateEncoder(model_name=config["state_encoder"]["model"])
    reward  = RewardEngine()

    class StubAgent:
        def perform_action(self, action_name, workspace):
            workspace.set("req_draft",
                workspace.get("req_draft","") +
                f"\nThe system shall support {action_name.replace('_',' ')}.")
            if action_name in ("write_final_srs","approve_and_document"):
                workspace.set("srs_document", workspace.get("req_draft",""))
            return {"output": f"executed {action_name}"}

    agents = {r: StubAgent() for r in AGENT_ACTION_MAP}

    def make_env():
        return RESimEnv(
            scenario_gen=gen, oracle=oracle,
            state_encoder=encoder, reward_engine=reward,
            agents=agents, agent_role="collector",
            max_steps=config["env"]["max_steps_per_episode"],
        )

    env = make_env()

    # ── Load trained model ─────────────────────────────────────────────
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    trained_model = PPO.load(checkpoint_path, env=DummyVecEnv([make_env]))
    logger.info(f"Loaded model from {checkpoint_path}")

    # ── Run evaluation ─────────────────────────────────────────────────
    remarl_results = []
    baseline_results = []
    opt = {"domain": domain} if domain else None

    print(f"\nRunning {n_eval} evaluation episodes...")
    for i in range(n_eval):
        obs, _ = env.reset(options=opt)

        # REMARL run
        r_ep = run_eval_episode(env, model=trained_model)
        if r_ep["oracle"]:
            remarl_results.append(r_ep["oracle"])

        # Baseline run (same scenario, random policy)
        obs, _ = env.reset(options=opt)
        b_ep = run_eval_episode(env, model=None)
        if b_ep["oracle"]:
            baseline_results.append(b_ep["oracle"])

        if (i + 1) % 10 == 0:
            print(f"  Episode {i+1}/{n_eval}")

    if not remarl_results or not baseline_results:
        print("Not enough episodes with oracle results. Check max_steps setting.")
        return

    remarl_agg  = aggregate_oracle_results(remarl_results)
    baseline_agg = aggregate_oracle_results(baseline_results)
    print_comparison(remarl_agg, baseline_agg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/remarl_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_eval",     type=int, default=50)
    parser.add_argument("--domain",     default=None)
    args = parser.parse_args()
    benchmark(args.config, args.checkpoint, args.n_eval, args.domain)
