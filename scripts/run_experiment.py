"""Run a small MMCC hedging experiment.

Increase num_paths, num_steps, hidden_sizes, and training epochs in the
parameter objects below for serious experiments.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mmcc_hedging import (  # noqa: E402
    AsianOptionParams,
    BaselineTrainer,
    DatePolicies,
    HedgingEnvironment,
    HedgingParams,
    HestonModel,
    HestonParams,
    InitialControl,
    MMCCTrainer,
    NetworkParams,
    TrainingParams,
)


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """Print scalar evaluation metrics."""

    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def main() -> None:
    """Build the model and run tiny baseline and MMCC trainings."""

    heston_params = HestonParams(
        initial_spot=100.0,
        initial_variance=0.04,
        risk_free_rate=0.02,
        maturity=1.0,
        num_steps=4,
    )
    option_params = AsianOptionParams(
        strike=100.0,
        maturity=heston_params.maturity,
        num_monitoring_dates=heston_params.num_steps,
    )
    hedging_params = HedgingParams(transaction_cost=0.001)
    network_params = NetworkParams(hidden_sizes=(16, 16))
    training_params = TrainingParams(
        num_iterations=1,
        num_paths=64,
        batch_size=32,
        num_epochs_per_date=2,
        num_epochs_initial_control=2,
        baseline_epochs=3,
        evaluation_num_paths=64,
        learning_rate=1e-3,
        seed=123,
        evaluation_seed=456,
        verbose=True,
    )

    market = HestonModel(heston_params)
    env = HedgingEnvironment(market, option_params, hedging_params)

    print("Running joint baseline")
    torch.manual_seed(training_params.seed)
    baseline_initial = InitialControl()
    baseline_policies = DatePolicies(heston_params.num_steps, network_params)
    baseline = BaselineTrainer(
        env,
        baseline_initial,
        baseline_policies,
        training_params,
    )
    baseline.train()
    print_metrics("Baseline evaluation", baseline.evaluate())

    print("\nRunning MMCC")
    torch.manual_seed(training_params.seed)
    mmcc_initial = InitialControl()
    mmcc_policies = DatePolicies(heston_params.num_steps, network_params)
    mmcc = MMCCTrainer(env, mmcc_initial, mmcc_policies, training_params)
    mmcc.train()
    print_metrics("MMCC evaluation", mmcc.evaluate())


if __name__ == "__main__":
    main()
