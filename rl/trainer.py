"""
remarl/rl/trainer.py
--------------------
High-level REMARL trainer.
Ties together: ScenarioGenerator, Oracle, RESimEnv, PPO policies,
EpisodeMemory, and the evaluation loop.

This is used by train.py but can also be imported directly:

    from rl.trainer import REMARLTrainer
    trainer = REMARLTrainer.from_config("configs/remarl_config.yaml")
    trainer.train()
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class REMARLTrainer:
    """
    Orchestrates training of one or more agent RL policies.

    Args:
        config: parsed YAML config dict
    """

    def __init__(self, config: dict):
        self.config = config
        self._setup_components()

    @classmethod
    def from_config(cls, config_path: str) -> "REMARLTrainer":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ── Public API ────────────────────────────────────────────────────

    def train(self, roles: Optional[List[str]] = None):
        """
        Train PPO policies for the specified agent roles.
        If roles=None, trains all roles in config["training"]["agent_roles"].
        """
        roles = roles or self.config["training"]["agent_roles"]
        for role in roles:
            logger.info(f"\n{'='*56}\nTraining role: {role}\n{'='*56}")
            self._train_role(role)

    def evaluate(self, checkpoint_dir: str, n_episodes: int = 50) -> dict:
        """
        Evaluate all trained checkpoints and return metrics dict.
        """
        from eval.benchmark import benchmark
        results = {}
        for role in self.config["training"]["agent_roles"]:
            path = Path(checkpoint_dir) / f"{role}_final"
            if path.with_suffix(".zip").exists() or path.exists():
                results[role] = benchmark(
                    config_path=None,   # pass config directly
                    checkpoint_path=str(path),
                    n_eval=n_episodes,
                    config=self.config,
                )
        return results

    # ── Private ───────────────────────────────────────────────────────

    def _setup_components(self):
        from sim.scenario_gen import ScenarioGenerator
        from sim.oracle import Oracle
        from rl.state_encoder import StateEncoder
        from rl.reward import RewardEngine
        from rl.memory import EpisodeMemory

        self.gen     = ScenarioGenerator(self.config["env"]["scenario_dir"])
        self.oracle  = Oracle(
            coverage_threshold=self.config["reward"]["coverage_threshold"]
        )
        self.encoder = StateEncoder(
            model_name=self.config["state_encoder"]["model"],
            max_steps=self.config["env"]["max_steps_per_episode"],
        )
        self.reward  = RewardEngine(
            clarity_weight=self.config["reward"]["clarity_weight"],
            consistency_weight=self.config["reward"]["consistency_weight"],
            coverage_delta_weight=self.config["reward"]["coverage_delta_weight"],
        )
        self.memory  = EpisodeMemory(self.config["memory"]["db_path"])

        Path(self.config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["training"]["log_dir"]).mkdir(parents=True, exist_ok=True)

        logger.info("Trainer components initialised.")

    def _make_env_fn(self, role: str):
        """Returns a zero-arg callable that builds a fresh RESimEnv."""
        gen, oracle, encoder, reward = (
            self.gen, self.oracle, self.encoder, self.reward
        )
        config = self.config

        from sim.re_env import RESimEnv, AGENT_ACTION_MAP

        class StubAgent:
            def perform_action(self, action_name, workspace):
                workspace.set(
                    "req_draft",
                    workspace.get("req_draft", "") +
                    f"\nThe system shall support {action_name.replace('_', ' ')}."
                )
                if action_name in ("write_final_srs", "approve_and_document"):
                    workspace.set("srs_document", workspace.get("req_draft", ""))
                return {"output": f"executed {action_name}", "success": True}

        agents = {r: StubAgent() for r in AGENT_ACTION_MAP}

        def env_fn():
            return RESimEnv(
                scenario_gen=gen,
                oracle=oracle,
                state_encoder=encoder,
                reward_engine=reward,
                agents=agents,
                agent_role=role,
                max_steps=config["env"]["max_steps_per_episode"],
                step_penalty=config["reward"]["step_penalty"],
            )

        return env_fn

    def _train_role(self, role: str):
        from rl.policy import create_ppo_policy

        env_fn = self._make_env_fn(role)
        model  = create_ppo_policy(env_fn, self.config, role)

        n_episodes   = self.config["training"]["n_episodes"]
        steps_per_ep = self.config["env"]["max_steps_per_episode"]
        total_steps  = n_episodes * steps_per_ep

        save_dir = Path(self.config["training"]["checkpoint_dir"])

        logger.info(
            f"PPO training | role={role} | "
            f"episodes={n_episodes} | total_steps={total_steps:,}"
        )

        # SB3 checkpoint callback
        from stable_baselines3.common.callbacks import CheckpointCallback
        save_every_steps = (
            self.config["training"]["save_every_n_episodes"] * steps_per_ep
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=save_every_steps,
            save_path=str(save_dir),
            name_prefix=role,
            verbose=1,
        )

        model.learn(
            total_timesteps=total_steps,
            callback=checkpoint_cb,
            progress_bar=True,
            tb_log_name=f"ppo_{role}",
            reset_num_timesteps=True,
        )

        final_path = save_dir / f"{role}_final"
        model.save(str(final_path))
        logger.info(f"Saved: {final_path}.zip")
