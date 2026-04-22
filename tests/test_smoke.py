"""Lightweight tests for the simplified MMCC project."""

import torch

from mmcc_hedging import (
    AsianOptionParams,
    DatePolicies,
    HedgingEnvironment,
    HedgingParams,
    HestonModel,
    HestonParams,
    InitialControl,
    MMCCTrainer,
    NetworkParams,
    TrainingParams,
    hedging_loss,
)


def make_env(num_steps: int = 3) -> HedgingEnvironment:
    """Build a small deterministic-ish environment for tests."""

    heston_params = HestonParams(
        initial_spot=100.0,
        initial_variance=0.04,
        risk_free_rate=0.0,
        maturity=1.0,
        num_steps=num_steps,
    )
    return HedgingEnvironment(
        HestonModel(heston_params),
        AsianOptionParams(
            strike=100.0,
            maturity=heston_params.maturity,
            num_monitoring_dates=heston_params.num_steps,
        ),
        HedgingParams(transaction_cost=0.01),
    )


def test_heston_shapes_and_nonnegative_variance() -> None:
    env = make_env(num_steps=4)

    paths = env.market.simulate(num_paths=5, seed=7)

    assert paths.spot.shape == (5, 5)
    assert paths.variance.shape == (5, 5)
    assert torch.all(paths.variance >= 0.0)


def test_rollout_shapes_and_loss() -> None:
    env = make_env(num_steps=3)
    paths = env.market.simulate(num_paths=4, seed=11)
    initial = InitialControl()
    policies = DatePolicies(env.num_steps, NetworkParams(hidden_sizes=(8,)))

    trajectory = env.rollout(paths, initial, policies)
    loss = hedging_loss(trajectory.terminal_wealth, trajectory.payoff)

    assert trajectory.running_sum.shape == (4, 4)
    assert trajectory.target_position.shape == (3, 4)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_mmcc_one_iteration_runs() -> None:
    env = make_env(num_steps=3)
    initial = InitialControl()
    policies = DatePolicies(env.num_steps, NetworkParams(hidden_sizes=(8,)))
    params = TrainingParams(
        num_iterations=1,
        num_paths=4,
        batch_size=2,
        num_epochs_per_date=1,
        num_epochs_initial_control=1,
        baseline_epochs=1,
        evaluation_num_paths=4,
        learning_rate=1e-3,
        seed=5,
        verbose=False,
    )

    history = MMCCTrainer(env, initial, policies, params, logger=lambda _: None).train()

    assert len(history) == 1
    assert list(history[0].date_losses) == [2, 1]
    assert torch.isfinite(torch.tensor(history[0].post_loss))
