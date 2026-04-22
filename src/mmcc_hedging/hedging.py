"""Asian option hedging dynamics with proportional transaction costs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mmcc_hedging.heston import HestonModel, HestonPaths
from mmcc_hedging.network import DatePolicies, InitialControl
from mmcc_hedging.params import AsianOptionParams, HedgingParams


@dataclass(frozen=True, slots=True)
class HedgingState:
    """Pre-trade augmented state s_n = (S_n, V_n, A_n, q_{n-1}, B_n)."""

    time_index: int
    spot: Tensor
    variance: Tensor
    running_sum: Tensor
    position: Tensor
    cash: Tensor


@dataclass(frozen=True, slots=True)
class HedgingTrajectory:
    """Full simulated hedging trajectory."""

    time: Tensor
    spot: Tensor
    variance: Tensor
    running_sum: Tensor
    position: Tensor
    cash: Tensor
    target_position: Tensor
    payoff: Tensor
    terminal_wealth: Tensor

    @property
    def hedging_error(self) -> Tensor:
        """Return W_N - H path by path."""

        return self.terminal_wealth - self.payoff

    def state_at(self, time_index: int, indices: Tensor | None = None) -> HedgingState:
        """Extract s_n, optionally for a path sub-batch."""

        if not 0 <= time_index < self.time.shape[0]:
            raise ValueError("time_index is outside the simulated grid.")
        if indices is None:
            return HedgingState(
                time_index=time_index,
                spot=self.spot[time_index],
                variance=self.variance[time_index],
                running_sum=self.running_sum[time_index],
                position=self.position[time_index],
                cash=self.cash[time_index],
            )

        idx = indices.to(device=self.spot.device, dtype=torch.long)
        return HedgingState(
            time_index=time_index,
            spot=torch.index_select(self.spot[time_index], dim=0, index=idx),
            variance=torch.index_select(self.variance[time_index], dim=0, index=idx),
            running_sum=torch.index_select(
                self.running_sum[time_index],
                dim=0,
                index=idx,
            ),
            position=torch.index_select(self.position[time_index], dim=0, index=idx),
            cash=torch.index_select(self.cash[time_index], dim=0, index=idx),
        )


class HedgingEnvironment:
    """Couples Heston paths, Asian payoff, and portfolio accounting."""

    def __init__(
        self,
        market: HestonModel,
        option_params: AsianOptionParams,
        hedging_params: HedgingParams,
    ) -> None:
        self.market = market
        self.option_params = option_params
        self.hedging_params = hedging_params
        if option_params.num_monitoring_dates != market.num_steps:
            raise ValueError("Asian monitoring dates must match Heston num_steps.")
        if abs(option_params.maturity - market.params.maturity) > 1e-12:
            raise ValueError("Asian maturity must match Heston maturity.")

    @property
    def num_steps(self) -> int:
        """Return N, the number of hedging dates after t_0."""

        return self.market.num_steps

    def transaction_cost(self, spot: Tensor, trade: Tensor) -> Tensor:
        """Return lambda S_n |q_n - q_{n-1}|."""

        return self.hedging_params.transaction_cost * spot * torch.abs(trade)

    def cash_after_initial_trade(
        self,
        spot: Tensor,
        cash: Tensor,
        previous_position: Tensor,
        premium: Tensor,
        target_position: Tensor,
    ) -> Tensor:
        """Return B_1 after applying c_0 = (y, q_0)."""

        trade = target_position - previous_position
        return self._growth_like(spot) * (
            cash + premium - trade * spot - self.transaction_cost(spot, trade)
        )

    def cash_after_rebalance(
        self,
        spot: Tensor,
        cash: Tensor,
        previous_position: Tensor,
        target_position: Tensor,
    ) -> Tensor:
        """Return B_{n+1} after rebalancing from q_{n-1} to q_n."""

        trade = target_position - previous_position
        return self._growth_like(spot) * (
            cash - trade * spot - self.transaction_cost(spot, trade)
        )

    def payoff(self, running_sum: Tensor) -> Tensor:
        """Return H = max(A_N / N - K, 0)."""

        average = running_sum / self.option_params.num_monitoring_dates
        return torch.clamp(average - self.option_params.strike, min=0.0)

    def terminal_wealth(
        self,
        final_spot: Tensor,
        cash: Tensor,
        position: Tensor,
    ) -> Tensor:
        """Return W_N after liquidation of the final stock inventory."""

        liquidation_cost = self.transaction_cost(final_spot, position)
        return cash + position * final_spot - liquidation_cost

    def rollout(
        self,
        paths: HestonPaths,
        initial_control: InitialControl,
        policies: DatePolicies,
    ) -> HedgingTrajectory:
        """Simulate the full hedging strategy on fixed Heston paths."""

        spot = paths.spot
        variance = paths.variance
        num_paths = paths.num_paths
        dtype = spot.dtype
        device = spot.device

        running_sum = torch.full((num_paths,), 0.0, device=device, dtype=dtype)
        position = torch.full(
            (num_paths,),
            self.hedging_params.initial_position,
            device=device,
            dtype=dtype,
        )
        cash = torch.full(
            (num_paths,),
            self.hedging_params.initial_cash,
            device=device,
            dtype=dtype,
        )

        running_sum_history = [running_sum]
        position_history = [position]
        cash_history = [cash]
        controls: list[Tensor] = []

        for n in range(self.num_steps):
            if n == 0:
                output = initial_control(num_paths)
                target_position = output.position
                cash = self.cash_after_initial_trade(
                    spot[n],
                    cash,
                    position,
                    output.premium,
                    target_position,
                )
            else:
                target_position = policies.action(
                    n,
                    spot[n],
                    variance[n],
                    running_sum,
                    position,
                    cash,
                )
                cash = self.cash_after_rebalance(
                    spot[n],
                    cash,
                    position,
                    target_position,
                )

            position = target_position
            running_sum = running_sum + spot[n + 1]

            controls.append(target_position)
            running_sum_history.append(running_sum)
            position_history.append(position)
            cash_history.append(cash)

        payoff = self.payoff(running_sum)
        terminal_wealth = self.terminal_wealth(spot[-1], cash, position)
        return HedgingTrajectory(
            time=paths.time,
            spot=spot,
            variance=variance,
            running_sum=torch.stack(running_sum_history),
            position=torch.stack(position_history),
            cash=torch.stack(cash_history),
            target_position=torch.stack(controls),
            payoff=payoff,
            terminal_wealth=terminal_wealth,
        )

    def continue_from_state(
        self,
        start_time: int,
        state: HedgingState,
        paths: HestonPaths,
        policies: DatePolicies,
    ) -> tuple[Tensor, Tensor]:
        """Continue a trajectory from a pre-trade state s_t to maturity."""

        if not 1 <= start_time < self.num_steps:
            raise ValueError("start_time must be in 1, ..., N - 1.")

        running_sum = state.running_sum
        position = state.position
        cash = state.cash

        for n in range(start_time, self.num_steps):
            target_position = policies.action(
                n,
                paths.spot[n],
                paths.variance[n],
                running_sum,
                position,
                cash,
            )
            cash = self.cash_after_rebalance(
                paths.spot[n],
                cash,
                position,
                target_position,
            )
            position = target_position
            running_sum = running_sum + paths.spot[n + 1]

        payoff = self.payoff(running_sum)
        terminal_wealth = self.terminal_wealth(paths.spot[-1], cash, position)
        return terminal_wealth, payoff

    def _growth_like(self, tensor: Tensor) -> Tensor:
        return torch.exp(
            torch.as_tensor(
                self.market.params.risk_free_rate * self.market.dt,
                device=tensor.device,
                dtype=tensor.dtype,
            )
        )
