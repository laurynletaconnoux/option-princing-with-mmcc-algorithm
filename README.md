# MMCC Dynamic Hedging

Apply the MMCC algorithm to a challenging high-dimensional, path-dependent financial 
problem over a finite-horizon. In particular, we study the dynamic hedging of an Asian 
option under a stochastic volatility model with proportional transaction costs.

## Mathematical Problem

The goal is to minimize the quadratic hedging error

```text
E[(Y_N - g)^2]
```

where `g` is the payoff of a discretely Asian call and `Y_N` is the
terminal wealth of a self-financing portfolio with proportional transaction
costs.

The market follows risk-neutral Heston dynamics. Variance is simulated with
full-truncation Euler, and the stock is simulated with log-Euler.

The Asian running sum is

```text
A_0 = 0
A_{n+1} = A_n + S_{n+1}
g = max(A_N / N - K, 0)
```

The pre-trade state is

```text
s_n = (S_n, V_n, A_n, Q_{n-1}, B_n)
```

At `t_0`, the control is separate:

```text
c_0 = (y_0, q_0)
```

For dates `t_1, ..., t_{N-1}`, there is one MLP policy network per date.

## Architecture

```text
src/mmcc_hedging/
    __init__.py
    params.py      # dataclass parameter objects
    heston.py      # Heston path simulation
    hedging.py     # Asian payoff, transaction costs, portfolio rollout
    network.py     # InitialControl and one-date MLP policies
    mmcc.py        # baseline trainer, MMCC trainer, loss, metrics

scripts/
    run_experiment.py

tests/
    test_smoke.py
```

## Setup

Create and activate a virtual environment:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Install the package:

```bat
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The repository ignores `.venv/`, `v.env/`, caches, and build artifacts.

## Run Tests

```bat
python -m pytest
```

## Run Experiment

```bat
python scripts\run_experiment.py
```

The script runs a tiny joint baseline and a tiny MMCC training loop, then prints
hedging error diagnostics. For real experiments, increase `num_steps`,
`num_paths`, hidden sizes, and training epochs in `scripts/run_experiment.py`.

## Implemented

- Risk-neutral Heston simulation.
- Discretely monitored Asian call payoff.
- Proportional transaction costs.
- Augmented state `(S, V, A, Q, B)`.
- Separate learnable initial control `(y_0, q_0)`.
- One MLP policy per rebalancing date.
- Joint baseline training.
- MMCC backward date-by-date updates with Adam.

## Future Work

- Add plotting and saved experiment outputs.
- Run larger experiments for the report tables.
- Compare baseline training, MMCC, and simple heuristic hedges.
- Add sensitivity analysis for transaction costs and Heston parameters.
