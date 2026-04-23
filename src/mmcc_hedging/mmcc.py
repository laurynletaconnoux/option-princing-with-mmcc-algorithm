"""Baseline and MMCC training loops."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mmcc_hedging.hedging import HedgingEnvironment, HedgingTrajectory
from mmcc_hedging.heston import HestonPaths
from mmcc_hedging.network import DatePolicies, InitialControl
from mmcc_hedging.params import TrainingParams


def hedging_loss(terminal_wealth: Tensor, payoff: Tensor) -> Tensor:
    """Return E[(Y_N - g(.))^2] estimated by a Monte Carlo average."""

    return torch.mean((terminal_wealth - payoff) ** 2)


def evaluate(trajectory: HedgingTrajectory) -> dict[str, float]:
    """Return a few scalar diagnostics for a hedging trajectory."""

    error = trajectory.hedging_error.detach()
    probs = torch.tensor([0.05, 0.5, 0.95], device=error.device)
    quantiles = torch.quantile(error, probs)
    return {
        "mean_error": float(error.mean().cpu()),
        "mse": float(torch.mean(error**2).cpu()),
        "variance": float(error.var(unbiased=False).cpu()),
        "q05": float(quantiles[0].cpu()),
        "q50": float(quantiles[1].cpu()),
        "q95": float(quantiles[2].cpu()),
    }


@dataclass(frozen=True, slots=True)
class MMCCLog:
    """One MMCC outer iteration."""

    iteration: int
    pre_loss: float
    post_loss: float
    initial_loss: float
    date_losses: dict[int, float]


class BaselineTrainer:
    """Simple joint training baseline for debugging."""

    def __init__(
        self,
        env: HedgingEnvironment,
        initial_control: InitialControl,
        policies: DatePolicies,
        params: TrainingParams,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.env = env
        self.initial_control = initial_control
        self.policies = policies
        self.params = params
        self.logger = logger or print
        self.device = _device(params.device)
        self.dtype = _dtype(params.dtype)
        self._move_modules()

    def train(self) -> list[float]:
        """Train all controls jointly with Adam."""

        torch.manual_seed(self.params.seed)
        optimizer = torch.optim.Adam(
            self._all_parameters(),
            lr=self.params.learning_rate,
            betas=self.params.adam_betas,
            weight_decay=self.params.weight_decay,
        )
        losses: list[float] = []

        for epoch in range(self.params.baseline_epochs):
            paths = self.env.market.simulate(
                self.params.num_paths,
                seed=self.params.seed + epoch,
                device=self.device,
                dtype=self.dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            trajectory = self.env.rollout(paths, self.initial_control, self.policies)
            loss = hedging_loss(trajectory.terminal_wealth, trajectory.payoff)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            if self.params.verbose and epoch % self.params.log_every == 0:
                self.logger(f"baseline epoch={epoch} loss={losses[-1]:.6f}")

        return losses

    @torch.no_grad()
    def evaluate(self, seed: int | None = None) -> dict[str, float]:
        """Evaluate the current controls on fresh paths."""

        paths = self.env.market.simulate(
            self.params.evaluation_num_paths,
            seed=self.params.evaluation_seed if seed is None else seed,
            device=self.device,
            dtype=self.dtype,
        )
        return evaluate(self.env.rollout(paths, self.initial_control, self.policies))

    def _all_parameters(self) -> list[nn.Parameter]:
        return list(self.initial_control.parameters()) + list(self.policies.parameters())

    def _move_modules(self) -> None:
        self.initial_control.to(device=self.device, dtype=self.dtype)
        self.policies.to(device=self.device, dtype=self.dtype)


class MMCCTrainer:
    """Backward date-by-date MMCC training with Adam subproblems."""

    def __init__(
        self,
        env: HedgingEnvironment,
        initial_control: InitialControl,
        policies: DatePolicies,
        params: TrainingParams,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.env = env
        self.initial_control = initial_control
        self.policies = policies
        self.params = params
        self.logger = logger or print
        self.device = _device(params.device)
        self.dtype = _dtype(params.dtype)
        self._move_modules()

    def train(self) -> list[MMCCLog]:
        """Run Algorithm 1: simulate, update dates backward, update c_0."""

        torch.manual_seed(self.params.seed)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.params.seed + 10_000)
        history: list[MMCCLog] = []

        for iteration in range(self.params.num_iterations):
            paths = self.env.market.simulate(
                self.params.num_paths,
                seed=self.params.seed + iteration,
                device=self.device,
                dtype=self.dtype,
            )
            with torch.no_grad():
                reference = self.env.rollout(paths, self.initial_control, self.policies)
                pre_loss = hedging_loss(reference.terminal_wealth, reference.payoff)

            date_losses: dict[int, float] = {}
            for time_index in range(self.env.num_steps - 1, 0, -1):
                date_losses[time_index] = self._update_date_policy(
                    time_index,
                    paths,
                    reference,
                    generator,
                )

            initial_loss = self._update_initial_control(paths, generator)
            with torch.no_grad():
                updated = self.env.rollout(paths, self.initial_control, self.policies)
                post_loss = hedging_loss(updated.terminal_wealth, updated.payoff)

            log = MMCCLog(
                iteration=iteration,
                pre_loss=float(pre_loss.cpu()),
                post_loss=float(post_loss.cpu()),
                initial_loss=initial_loss,
                date_losses=date_losses,
            )
            history.append(log)

            if self.params.verbose and iteration % self.params.log_every == 0:
                losses = ", ".join(
                    f"t{time_index}={loss:.6f}"
                    for time_index, loss in date_losses.items()
                )
                self.logger(
                    f"mmcc iter={iteration} pre={log.pre_loss:.6f} "
                    f"post={log.post_loss:.6f} initial={initial_loss:.6f} "
                    f"[{losses}]"
                )

        return history

    @torch.no_grad()
    def evaluate(self, seed: int | None = None) -> dict[str, float]:
        """Evaluate the current MMCC strategy on fresh paths."""

        paths = self.env.market.simulate(
            self.params.evaluation_num_paths,
            seed=self.params.evaluation_seed if seed is None else seed,
            device=self.device,
            dtype=self.dtype,
        )
        return evaluate(self.env.rollout(paths, self.initial_control, self.policies))

    def _update_date_policy(
        self,
        time_index: int,
        paths: HestonPaths,
        reference: HedgingTrajectory,
        generator: torch.Generator,
    ) -> float:
        policy = self.policies.policy(time_index)

        def loss_for_indices(indices: Tensor) -> Tensor:
            batch_paths = paths.batch(indices)
            start_state = reference.state_at(time_index, indices)
            wealth, payoff = self.env.continue_from_state(
                time_index,
                start_state,
                batch_paths,
                self.policies,
            )
            return hedging_loss(wealth, payoff)

        return self._optimize_subproblem(
            parameters=policy.parameters(),
            loss_for_indices=loss_for_indices,
            num_epochs=self.params.num_epochs_per_date,
            paths=paths,
            generator=generator,
        )

    def _update_initial_control(
        self,
        paths: HestonPaths,
        generator: torch.Generator,
    ) -> float:
        def loss_for_indices(indices: Tensor) -> Tensor:
            batch_paths = paths.batch(indices)
            trajectory = self.env.rollout(
                batch_paths,
                self.initial_control,
                self.policies,
            )
            return hedging_loss(trajectory.terminal_wealth, trajectory.payoff)

        return self._optimize_subproblem(
            parameters=self.initial_control.parameters(),
            loss_for_indices=loss_for_indices,
            num_epochs=self.params.num_epochs_initial_control,
            paths=paths,
            generator=generator,
        )

    def _optimize_subproblem(
        self,
        *,
        parameters: Iterable[nn.Parameter],
        loss_for_indices: Callable[[Tensor], Tensor],
        num_epochs: int,
        paths: HestonPaths,
        generator: torch.Generator,
    ) -> float:
        selected = list(parameters)
        all_indices = torch.arange(paths.num_paths, device=self.device)
        with torch.no_grad():
            before = loss_for_indices(all_indices).detach()
        snapshot = [parameter.detach().clone() for parameter in selected]
        previous_flags = self._set_only_trainable(selected)

        optimizer = torch.optim.Adam(
            selected,
            lr=self.params.learning_rate,
            betas=self.params.adam_betas,
            weight_decay=self.params.weight_decay,
        )

        try:
            for _ in range(num_epochs):
                indices = self._sample_indices(paths.num_paths, generator)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for_indices(indices)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                after = loss_for_indices(all_indices).detach()
            if torch.isfinite(after) and after <= before:
                return float(after.cpu())

            with torch.no_grad():
                for parameter, value in zip(selected, snapshot, strict=True):
                    parameter.copy_(value)
            return float(before.cpu())
        finally:
            for parameter, flag in previous_flags:
                parameter.requires_grad_(flag)
            for parameter in self._all_parameters():
                parameter.grad = None

    def _sample_indices(self, num_paths: int, generator: torch.Generator) -> Tensor:
        if self.params.batch_size == num_paths:
            return torch.arange(num_paths, device=self.device)
        return torch.randint(
            low=0,
            high=num_paths,
            size=(self.params.batch_size,),
            generator=generator,
            device=self.device,
        )

    def _all_parameters(self) -> list[nn.Parameter]:
        return list(self.initial_control.parameters()) + list(self.policies.parameters())

    def _set_only_trainable(
        self,
        selected: list[nn.Parameter],
    ) -> list[tuple[nn.Parameter, bool]]:
        selected_ids = {id(parameter) for parameter in selected}
        previous = [
            (parameter, parameter.requires_grad)
            for parameter in self._all_parameters()
        ]
        for parameter, _ in previous:
            parameter.requires_grad_(id(parameter) in selected_ids)
        return previous

    def _move_modules(self) -> None:
        self.initial_control.to(device=self.device, dtype=self.dtype)
        self.policies.to(device=self.device, dtype=self.dtype)


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'.")
