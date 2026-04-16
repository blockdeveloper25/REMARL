# REMARL — Multi-Agent Reinforcement Learning for Requirements Engineering

> Extends MARE (Multi-Agent RE Framework) with RL policies that learn
> to elicit, negotiate, and specify requirements better over time.

## Quick start

```bash
# 1. Clone MARE into the mare/ directory
git clone https://github.com/tuyoshivinicius/mare-implementation.git mare/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify MARE still works unchanged
python -m mare.pipeline --idea "a simple todo app"

# 4. Build the scenario cache (one-time)
python sim/scenario_gen.py

# 5. Verify the Gymnasium environment passes SB3's check
python sim/re_env.py

# 6. Run the full training loop
python train.py --config configs/remarl_config.yaml
```

## Project layout

```
remarl/
├── mare/           ← MARE repo (mostly unchanged)
├── rl/             ← RL layer: policies, rewards, state encoding
├── sim/            ← Simulation environment + scenario generator
├── eval/           ← Benchmarking and metrics
├── configs/        ← YAML config files
├── data/           ← Scenarios, episode memory, checkpoints
└── tests/          ← Unit and integration tests
```

## Architecture in one sentence

MARE handles language (LLM prompts). REMARL adds strategic intelligence
(PPO policies that decide *which* action to take and learns from
downstream quality signals).
