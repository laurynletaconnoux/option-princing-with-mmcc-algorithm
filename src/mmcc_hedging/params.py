"""Small dataclass parameter objects for the MMCC experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class HestonParams:
    """Risk-neutral Heston model parameters."""

    initial_spot: float = 100.0
    initial_variance: float = 0.04
    risk_free_rate: float = 0.02
    kappa: float = 2.0
    theta: float = 0.04
    vol_of_variance: float = 0.5
    correlation: float = -0.7
    maturity: float = 1.0
    num_steps: int = 12

    @property
    def dt(self) -> float:
        """Return the time step size."""

        return self.maturity / self.num_steps

    def __post_init__(self) -> None:
        if self.initial_spot <= 0.0:
            raise ValueError("initial_spot must be positive.")
        if self.initial_variance < 0.0:
            raise ValueError("initial_variance must be non-negative.")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if self.theta < 0.0:
            raise ValueError("theta must be non-negative.")
        if self.vol_of_variance < 0.0:
            raise ValueError("vol_of_variance must be non-negative.")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must be in [-1, 1].")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")


@dataclass(frozen=True, slots=True)
class AsianOptionParams:
    """Discretely monitored arithmetic Asian call parameters."""

    strike: float = 100.0
    maturity: float = 1.0
    num_monitoring_dates: int = 12

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError("strike must be positive.")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if self.num_monitoring_dates <= 0:
            raise ValueError("num_monitoring_dates must be positive.")


@dataclass(frozen=True, slots=True)
class HedgingParams:
    """Portfolio and transaction cost parameters."""

    transaction_cost: float = 0.001
    initial_cash: float = 0.0
    initial_position: float = 0.0

    def __post_init__(self) -> None:
        if self.transaction_cost < 0.0:
            raise ValueError("transaction_cost must be non-negative.")


@dataclass(frozen=True, slots=True)
class NetworkParams:
    """Feed-forward policy network parameters."""

    input_dim: int = 5
    hidden_sizes: tuple[int, ...] = (64, 64)
    output_dim: int = 1

    spot_scale: float = 100.0
    variance_scale: float = 0.0225
    average_scale: float = 100.0
    position_scale: float = 1.0
    cash_scale: float = 100.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")

        if self.spot_scale <= 0.0:
            raise ValueError("spot_scale must be positive.")
        if self.variance_scale <= 0.0:
            raise ValueError("variance_scale must be positive.")
        if self.average_scale <= 0.0:
            raise ValueError("average_scale must be positive.")
        if self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive.")
        if self.cash_scale <= 0.0:
            raise ValueError("cash_scale must be positive.")


@dataclass(frozen=True, slots=True)
class TrainingParams:
    """Training hyperparameters for baseline and MMCC."""

    num_iterations: int = 3
    num_paths: int = 2048
    batch_size: int = 512
    num_epochs_per_date: int = 100
    num_epochs_initial_control: int = 100
    baseline_epochs: int = 100
    evaluation_num_paths: int = 2048
    learning_rate: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    seed: int = 1234
    evaluation_seed: int = 4321
    device: str = "cpu"
    dtype: Literal["float32", "float64"] = "float32"
    log_every: int = 1
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if self.num_paths <= 0:
            raise ValueError("num_paths must be positive.")
        if not 0 < self.batch_size <= self.num_paths:
            raise ValueError("batch_size must be in {1, ..., num_paths}.")
        if self.num_epochs_per_date <= 0:
            raise ValueError("num_epochs_per_date must be positive.")
        if self.num_epochs_initial_control <= 0:
            raise ValueError("num_epochs_initial_control must be positive.")
        if self.baseline_epochs <= 0:
            raise ValueError("baseline_epochs must be positive.")
        if self.evaluation_num_paths <= 0:
            raise ValueError("evaluation_num_paths must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if len(self.adam_betas) != 2:
            raise ValueError("adam_betas must contain two values.")
        if not 0.0 <= self.adam_betas[0] < 1.0:
            raise ValueError("adam_betas[0] must be in [0, 1).")
        if not 0.0 <= self.adam_betas[1] < 1.0:
            raise ValueError("adam_betas[1] must be in [0, 1).")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.seed < 0 or self.evaluation_seed < 0:
            raise ValueError("seeds must be non-negative.")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")
