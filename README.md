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
    params.py          # dataclass parameter objects
    heston.py          # Heston path simulation
    hedging.py         # Asian payoff, transaction costs, portfolio rollout
    network.py         # InitialControl and one-date MLP policies
    mmcc.py            # baseline trainer, MMCC trainer, loss, metrics
    visualization_2.py # Plots
scripts/
    run_experiment.py

tests/
    test_smoke.py
```

## Setup

Create and activate a virtual environment:

```bat
.\create_venv.sh
```

## Run Experiment

```bat
python scripts\run_experiment.py
```
