"""Policy networks for the hedging controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mmcc_hedging.params import NetworkParams


def state_features(
    spot: Tensor,
    variance: Tensor,
    running_sum: Tensor,
    position: Tensor,
    cash: Tensor,
) -> Tensor:
    """Stack s_n = (S_n, V_n, A_n, q_{n-1}, B_n)."""

    return torch.stack((spot, variance, running_sum, position, cash), dim=-1)


class MLPPolicy(nn.Module):
    """Single-date policy q_n = c(t_n, s_n; theta_n)."""

    def __init__(self, params: NetworkParams | None = None) -> None:
        super().__init__()
        self.params = params or NetworkParams()
        layer_sizes = (
            self.params.input_dim,
            *self.params.hidden_sizes,
            self.params.output_dim,
        )
        layers: list[nn.Module] = []
        for in_features, out_features in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """Return the raw network output."""

        return self.net(features)

    def action(
        self,
        spot: Tensor,
        variance: Tensor,
        running_sum: Tensor,
        position: Tensor,
        cash: Tensor,
    ) -> Tensor:
        """Return the scalar target inventory for a state batch."""

        output = self.forward(state_features(spot, variance, running_sum, position, cash))
        return output.squeeze(-1)


class DatePolicies(nn.Module):
    """One MLP policy for each date t_1, ..., t_{N-1}."""

    def __init__(self, num_steps: int, params: NetworkParams | None = None) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2.")
        self.num_steps = num_steps
        self.policies = nn.ModuleList([MLPPolicy(params) for _ in range(num_steps - 1)])

    def policy(self, time_index: int) -> MLPPolicy:
        """Return the policy associated with t_1, ..., t_{N-1}."""

        if not 1 <= time_index < self.num_steps:
            raise ValueError("time_index must be in 1, ..., N - 1.")
        return self.policies[time_index - 1]

    def action(
        self,
        time_index: int,
        spot: Tensor,
        variance: Tensor,
        running_sum: Tensor,
        position: Tensor,
        cash: Tensor,
    ) -> Tensor:
        """Evaluate the policy at one date."""

        return self.policy(time_index).action(
            spot,
            variance,
            running_sum,
            position,
            cash,
        )


@dataclass(frozen=True, slots=True)
class InitialControlOutput:
    """Batched initial premium y and initial inventory q_0."""

    premium: Tensor
    position: Tensor


class InitialControl(nn.Module):
    """Learnable initial control c_0 = (y, q_0)."""

    def __init__(
        self,
        initial_premium: float = 0.0,
        initial_position: float = 0.0,
    ) -> None:
        super().__init__()
        self.premium = nn.Parameter(torch.tensor(float(initial_premium)))
        self.position = nn.Parameter(torch.tensor(float(initial_position)))

    def forward(self, num_paths: int) -> InitialControlOutput:
        """Broadcast y and q_0 to all Monte Carlo paths."""

        if num_paths <= 0:
            raise ValueError("num_paths must be positive.")
        return InitialControlOutput(
            premium=self.premium.expand(num_paths),
            position=self.position.expand(num_paths),
        )
