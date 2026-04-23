"""Policy networks for the hedging controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mmcc_hedging.params import NetworkParams

     
def state_features(
    time_index: int,
    spot: Tensor,
    variance: Tensor,
    running_sum: Tensor,
    position: Tensor,
    cash: Tensor,
    params: NetworkParams,
) -> Tensor:
    """Build normalized state features for the policy network.

    Raw state:
        s_n = (S_n, V_n, A_n, q_{n-1}, B_n)

    Network features:
        (
            S_n / spot_scale,
            V_n / variance_scale,
            (A_n / n) / average_scale,
            q_{n-1} / position_scale,
            B_n / cash_scale,
        )
    """
    if time_index <= 0:
        raise ValueError("time_index must be at least 1 for policy features.")

    running_average = running_sum / float(time_index)

    return torch.stack(
        (
            spot / params.spot_scale,
            variance / params.variance_scale,
            running_average / params.average_scale,
            position / params.position_scale,
            cash / params.cash_scale,
        ),
        dim=-1,
    )


class MLPPolicy(nn.Module):
    """Single-date policy q_n = c(t_n, s_n; theta_n)."""

    def __init__(self, params: NetworkParams | None = None) -> None:
        super().__init__()
        self.params = params or NetworkParams() # Use default parameters if none are provided.

        layer_sizes = [ 
            self.params.input_dim,
            *self.params.hidden_sizes,
            self.params.output_dim,
        ] # List of layer sizes, including input, hidden, and output layers.

        layers = [] # List to hold the layers of the network.

        for i in range(len(layer_sizes) - 2): # Loop through the hidden layers (excluding the output layer)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.SiLU()) # Activation function after each hidden layer

        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1])) # Linear layer for the output
        self.net = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """Return the raw network output."""
        return self.net(features)

    def action(
        self,
        time_index: int,
        spot: Tensor,
        variance: Tensor,
        running_sum: Tensor,
        position: Tensor,
        cash: Tensor,
    ) -> Tensor:
        """Return the scalar target inventory for a state batch."""
        features = state_features(
            time_index=time_index,
            spot=spot,
            variance=variance,
            running_sum=running_sum,
            position=position,
            cash=cash,
            params=self.params,
        )
        output = self.forward(features)
        return output.squeeze(-1)


class DatePolicies(nn.Module):
    """One MLP policy for each date t_1, ..., t_{N-1}."""

    def __init__(self, num_steps: int, params: NetworkParams | None = None) -> None:
        super().__init__()

        if num_steps < 2:
            raise ValueError("num_steps must be at least 2.") # We need at least two steps to have a non-trivial policy (one for t_1, ..., t_{N-1}).

        self.num_steps = num_steps 

        policies = []
        for _ in range(num_steps - 1):
            policies.append(MLPPolicy(params)) # Create a separate MLPPolicy for each date t_1, ..., t_{N-1}.

        self.policies = nn.ModuleList(policies)

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
            time_index,
            spot,
            variance,
            running_sum,
            position,
            cash,
        )


@dataclass(frozen=True, slots=True)
class InitialControlOutput:
    """Batched initial premium y and initial inventory q_0."""

    premium: Tensor #premium correpsond à y ?
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

