# cs238v-project

Starter repository for "Validation of Autonomous Driving Policies Under Actuation Delays" using [Highway-Env](https://github.com/Farama-Foundation/HighwayEnv).

To install project dependencies:
  ```python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Run a first experiment

```bash
python scripts/run_delay_experiment.py
```

This runs rollouts over:
- delays (steps): `0 1 2 3 4`
- traffic densities (`vehicles_count`): `20 35 50`
- rollouts per setting: `50`

Outputs:
- rollout-level CSV: `results/delay_results.csv`
- terminal summary: collision rate vs. delay and traffic

## Control experiment settings

Example custom run:

```bash
python scripts/run_delay_experiment.py \
  --rollouts 100 \
  --delays 0 1 2 4 6 8 \
  --traffic 15 30 45 60 \
  --policy random \
  --policy-frequency 5 \
  --max-steps 250 \
  --output results/exp1.csv
```

## Delay units

The script reports both `delay_steps` and `delay_ms`.

With `--policy-frequency 5`, each control step is `200 ms`, so:
- `delay_steps=1` -> `200 ms`
- `delay_steps=3` -> `600 ms`
- `delay_steps=5` -> `1000 ms`

## To do next!

- Primary safety metric: collision indicator (`collided`)
- Optional robustness metric: minimum TTC (`min_ttc`) if provided by the environment info
- To evaluate a stronger policy later, replace `choose_action()` in `scripts/run_delay_experiment.py` with your trained policy inference
