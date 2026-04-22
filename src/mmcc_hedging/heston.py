"""Risk-neutral Heston simulation in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mmcc_hedging.params import HestonParams


@dataclass(frozen=True, slots=True)
class HestonPaths:
    """Simulated Heston paths with shape (N + 1, num_paths)."""

    time: Tensor
    spot: Tensor
    variance: Tensor

    @property
    def num_paths(self) -> int:
        """Return the batch size."""

        return int(self.spot.shape[1])

    def batch(self, indices: Tensor) -> "HestonPaths":
        """Return a path sub-batch."""

        idx = indices.to(device=self.spot.device, dtype=torch.long)
        return HestonPaths(
            time=self.time,
            spot=torch.index_select(self.spot, dim=1, index=idx),
            variance=torch.index_select(self.variance, dim=1, index=idx),
        )


class HestonModel:
    """Heston model with full-truncation Euler variance and log-Euler spot."""

    def __init__(self, params: HestonParams) -> None:
        self.params = params

    @property
    def dt(self) -> float:
        """Return the simulation time step."""

        return self.params.dt

    @property
    def num_steps(self) -> int:
        """Return the number of time steps."""

        return self.params.num_steps

    def sample_increments(
        self,
        num_paths: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Sample correlated Brownian increments (dW_S, dW_V)."""

        device = torch.device("cpu") if device is None else torch.device(device)
        dtype = torch.get_default_dtype() if dtype is None else dtype
        normals = torch.randn(
            (num_paths, 2),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        rho = torch.as_tensor(self.params.correlation, device=device, dtype=dtype)
        independent_weight = torch.sqrt(torch.clamp(1.0 - rho * rho, min=0.0))
        sqrt_dt = torch.as_tensor(self.dt**0.5, device=device, dtype=dtype)

        dW_spot = sqrt_dt * normals[:, 0]
        dW_variance = sqrt_dt * (
            rho * normals[:, 0] + independent_weight * normals[:, 1]
        )
        return torch.stack((dW_spot, dW_variance), dim=-1)

    def step(
        self,
        spot: Tensor,
        variance: Tensor,
        increments: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Advance one time step from (S_n, V_n) to (S_{n+1}, V_{n+1})."""

        if spot.shape != variance.shape:
            raise ValueError("spot and variance must have the same shape.")
        if increments.shape != (*spot.shape, 2):
            raise ValueError("increments must have shape (*spot.shape, 2).")

        p = self.params
        dt = torch.as_tensor(self.dt, device=spot.device, dtype=spot.dtype)
        r = torch.as_tensor(p.risk_free_rate, device=spot.device, dtype=spot.dtype)
        kappa = torch.as_tensor(p.kappa, device=spot.device, dtype=spot.dtype)
        theta = torch.as_tensor(p.theta, device=spot.device, dtype=spot.dtype)
        xi = torch.as_tensor(p.vol_of_variance, device=spot.device, dtype=spot.dtype)

        variance_pos = torch.clamp(variance, min=0.0)
        sqrt_variance = torch.sqrt(variance_pos)
        dW_spot = increments[..., 0]
        dW_variance = increments[..., 1]

        next_spot = spot * torch.exp(
            (r - 0.5 * variance_pos) * dt + sqrt_variance * dW_spot
        )
        raw_next_variance = (
            variance
            + kappa * (theta - variance_pos) * dt
            + xi * sqrt_variance * dW_variance
        )
        next_variance = torch.clamp(raw_next_variance, min=0.0)
        return next_spot, next_variance

    def simulate(
        self,
        num_paths: int,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HestonPaths:
        """Simulate S and V on the full grid t_0, ..., t_N."""

        if num_paths <= 0:
            raise ValueError("num_paths must be positive.")
        device = torch.device("cpu") if device is None else torch.device(device)
        dtype = torch.get_default_dtype() if dtype is None else dtype
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        shape = (self.num_steps + 1, num_paths)
        spot = torch.empty(shape, device=device, dtype=dtype)
        variance = torch.empty(shape, device=device, dtype=dtype)
        spot[0] = self.params.initial_spot
        variance[0] = self.params.initial_variance

        for n in range(self.num_steps):
            increments = self.sample_increments(
                num_paths,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            spot[n + 1], variance[n + 1] = self.step(spot[n], variance[n], increments)

        time = torch.linspace(
            0.0,
            self.params.maturity,
            self.num_steps + 1,
            device=device,
            dtype=dtype,
        )
        return HestonPaths(time=time, spot=spot, variance=variance)
