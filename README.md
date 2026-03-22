# cs238v-project

"Validation of Autonomous Driving Policies Under Actuation Delays" using [Highway-Env](https://github.com/Farama-Foundation/HighwayEnv).

Run these commands to generate experiment results

Part 1: Direct Sampling (baseline)
```
python scripts/run_delay_experiment.py \
  --policy heuristic \
  --rollouts 200 \
  --traffic 8 12 16 20 \
  --delays 0 1 2 3 4 5 \
  --ttc-threshold 1.5 \
  --output results/delay_results_phase1.csv
```

Part 2: Falsification
```
python scripts/falsification_search.py \
  --budget 400 \
  --policy heuristic \
  --delays 0 1 2 3 4 5 6 7 \
  --traffic 8 12 16 20 24 \
  --ttc-threshold 1.5 \
  --output results/falsification_results.csv
```

Part 3: Failure probability estimation (direct monte carlo versus importance sampling)
```
python scripts/failure_probability_estimation.py \
  --budget 500 \
  --policy heuristic \
  --delays 0 1 2 3 4 5 \
  --traffic 8 12 16 20 \
  --ttc-threshold 1.5 \
  --output results/failure_probability_results.csv
```
