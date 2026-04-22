"""Run a small MMCC hedging experiment.

Increase num_paths, num_steps, hidden_sizes, and training epochs in the
parameter objects below for serious experiments.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

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
    Visualizer2,
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
        num_steps=20,
    )

    option_params = AsianOptionParams(
        strike=100.0,
        maturity=heston_params.maturity,
        num_monitoring_dates=heston_params.num_steps,
    )

    hedging_params = HedgingParams(transaction_cost=0.001)

    network_params = NetworkParams(
    hidden_sizes=(64, 64),
    spot_scale=heston_params.initial_spot,
    variance_scale=heston_params.theta,
    average_scale=option_params.strike,
    position_scale=1.0,
    cash_scale=option_params.strike,
    )

    training_params = TrainingParams(
    num_iterations=20,
    num_paths=10000,
    batch_size=1024,
    num_epochs_per_date=60,
    num_epochs_initial_control=60,
    baseline_epochs=150,
    evaluation_num_paths=30000,
    learning_rate=5e-4,
    seed=223,
    evaluation_seed=459,
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
    baseline_losses = baseline.train()
    print_metrics("Baseline evaluation", baseline.evaluate())

    print("\nRunning MMCC")
    torch.manual_seed(training_params.seed)
    mmcc_initial = InitialControl()
    mmcc_policies = DatePolicies(heston_params.num_steps, network_params)
    mmcc = MMCCTrainer(env, mmcc_initial, mmcc_policies, training_params)
    mmcc_history = mmcc.train()
    print_metrics("MMCC evaluation", mmcc.evaluate())

    eval_paths = market.simulate(
        training_params.evaluation_num_paths,
        seed=training_params.evaluation_seed,
    )
    baseline_trajectory = env.rollout(eval_paths, baseline_initial, baseline_policies)
    mmcc_trajectory = env.rollout(eval_paths, mmcc_initial, mmcc_policies)

    visualizer2 = Visualizer2(output_dir=Path("outputs"))
    visualizer2.plot_policy_evolution(
        baseline_trajectory,
        mmcc_trajectory,
        num_paths_to_plot=30,
    )
    visualizer2.plot_hedging_error(
        baseline_trajectory,
        mmcc_trajectory,
    )
    visualizer2.plot_convergence(
        baseline_losses,
        mmcc_history,
    )

    print("\nLearned initial control")
    print(f"  y  = {mmcc_initial.premium.detach().item():.6f}")
    print(f"  q0 = {mmcc_initial.position.detach().item():.6f}")


if __name__ == "__main__":
    main()