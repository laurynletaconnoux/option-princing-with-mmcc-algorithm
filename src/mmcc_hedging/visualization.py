from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from mmcc_hedging.hedging import HedgingTrajectory
from mmcc_hedging.mmcc import MMCCLog


@dataclass(slots=True)
class Visualizer:
    output_dir: Path

    def render(
        self,
        *,
        baseline_losses: list[float],
        mmcc_history: list[MMCCLog],
        baseline_metrics: dict[str, float],
        mmcc_metrics: dict[str, float],
        baseline_trajectory: HedgingTrajectory,
        mmcc_trajectory: HedgingTrajectory,
        num_paths_to_plot: int = 30,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib est requis pour la visualisation. "
                "Installe-le avec: pip install matplotlib"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        baseline_hedging_error = baseline_trajectory.hedging_error
        mmcc_hedging_error = mmcc_trajectory.hedging_error

        self._print_summary(baseline_metrics, mmcc_metrics)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        self._plot_training_losses(axes[0], baseline_losses, mmcc_history)
        self._plot_metrics(axes[1], baseline_metrics, mmcc_metrics)
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_overview.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        self._plot_hedging_error_histogram(ax, baseline_hedging_error, mmcc_hedging_error)
        fig.tight_layout()
        fig.savefig(self.output_dir / "hedging_error_distribution.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        self._plot_position_paths(axes[0], baseline_trajectory, num_paths_to_plot, "Baseline positions")
        self._plot_position_paths(axes[1], mmcc_trajectory, num_paths_to_plot, "MMCC positions")
        fig.tight_layout()
        fig.savefig(self.output_dir / "position_paths.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        self._plot_cash_paths(axes[0], baseline_trajectory, num_paths_to_plot, "Baseline cash")
        self._plot_cash_paths(axes[1], mmcc_trajectory, num_paths_to_plot, "MMCC cash")
        fig.tight_layout()
        fig.savefig(self.output_dir / "cash_paths.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        self._plot_turnover_boxplot(ax, baseline_trajectory, mmcc_trajectory)
        fig.tight_layout()
        fig.savefig(self.output_dir / "turnover_comparison.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        self._plot_mmcc_date_losses(ax, mmcc_history)
        fig.tight_layout()
        fig.savefig(self.output_dir / "mmcc_date_losses.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        self._plot_mmcc_gap_to_best(mmcc_history)
        fig.savefig(self.output_dir / "mmcc_gap_to_best.png", dpi=160)
        plt.close(fig)



    def _plot_position_paths(self, ax, trajectory: HedgingTrajectory, num_paths: int, title: str) -> None:
        time = trajectory.time.detach().cpu().numpy()
        position = trajectory.position.detach().cpu().numpy()   # shape (N+1, P)
        n = min(num_paths, position.shape[1])

        for i in range(n):
            ax.plot(time, position[:, i], alpha=0.35, linewidth=1.0)

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Inventory")
        ax.grid(alpha=0.25)

    def _plot_cash_paths(self, ax, trajectory: HedgingTrajectory, num_paths: int, title: str) -> None:
        time = trajectory.time.detach().cpu().numpy()
        cash = trajectory.cash.detach().cpu().numpy()
        n = min(num_paths, cash.shape[1])

        for i in range(n):
            ax.plot(time, cash[:, i], alpha=0.35, linewidth=1.0)

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cash account")
        ax.grid(alpha=0.25)

    def _plot_turnover_boxplot(
        self,
        ax,
        baseline_trajectory: HedgingTrajectory,
        mmcc_trajectory: HedgingTrajectory,
    ) -> None:
        baseline_turnover = torch.abs(
            baseline_trajectory.position[1:] - baseline_trajectory.position[:-1]
        ).sum(dim=0).detach().cpu().numpy()

        mmcc_turnover = torch.abs(
            mmcc_trajectory.position[1:] - mmcc_trajectory.position[:-1]
        ).sum(dim=0).detach().cpu().numpy()

        ax.boxplot([baseline_turnover, mmcc_turnover], labels=["Baseline", "MMCC"])
        ax.set_title("Turnover distribution")
        ax.set_ylabel(r"$\sum_n |q_n - q_{n-1}|$")
        ax.grid(alpha=0.25)

    def _plot_mmcc_date_losses(self, ax, mmcc_history: list[MMCCLog]) -> None:
        if not mmcc_history:
            return

        import numpy as np

        all_dates = sorted(mmcc_history[0].date_losses.keys())
        matrix = np.array([
            [log.date_losses[d] for d in all_dates]
            for log in mmcc_history
        ]).T

        image = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_title("MMCC date subproblem losses")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time index")
        ax.set_yticks(range(len(all_dates)), [str(d) for d in all_dates])
        ax.figure.colorbar(image, ax=ax)

    def _plot_position_quantiles(self, ax, trajectory: HedgingTrajectory, title: str) -> None:
        time = trajectory.time.detach().cpu()
        q = trajectory.position.detach().cpu()

        q10 = torch.quantile(q, 0.10, dim=1)
        q50 = torch.quantile(q, 0.50, dim=1)
        q90 = torch.quantile(q, 0.90, dim=1)

        ax.plot(time.numpy(), q50.numpy(), label="Median")
        ax.fill_between(time.numpy(), q10.numpy(), q90.numpy(), alpha=0.25, label="10%-90%")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Inventory")
        ax.grid(alpha=0.25)
        ax.legend()

    def _plot_training_losses(
        self,
        ax,
        baseline_losses: list[float],
        mmcc_history: list[MMCCLog],
    ) -> None:
        baseline_x = list(range(len(baseline_losses)))
        mmcc_x = [log.iteration for log in mmcc_history]
        mmcc_pre = [log.pre_loss for log in mmcc_history]
        mmcc_post = [log.post_loss for log in mmcc_history]

        if baseline_losses:
            ax.plot(baseline_x, baseline_losses, marker="o", label="Baseline (joint)")
        if mmcc_pre:
            ax.plot(mmcc_x, mmcc_pre, marker="o", label="MMCC pre-update")
        if mmcc_post:
            ax.plot(mmcc_x, mmcc_post, marker="o", label="MMCC post-update")

        ax.set_title("Training loss")
        ax.set_xlabel("Epoch / iteration")
        ax.set_ylabel("MSE hedging loss")
        ax.grid(alpha=0.25)
        ax.legend()

    def _plot_metrics(
        self,
        ax,
        baseline_metrics: dict[str, float],
        mmcc_metrics: dict[str, float],
    ) -> None:
        keys = ["mse", "variance", "mean_error"]
        labels = ["MSE", "Variance", "Mean error"]

        baseline_values = [baseline_metrics[key] for key in keys]
        mmcc_values = [mmcc_metrics[key] for key in keys]

        x = torch.arange(len(keys), dtype=torch.float32)
        width = 0.35

        ax.bar((x - width / 2).tolist(), baseline_values, width=width, label="Baseline")
        ax.bar((x + width / 2).tolist(), mmcc_values, width=width, label="MMCC")

        ax.set_xticks(x.tolist(), labels)
        ax.set_title("Evaluation metrics")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    def _plot_hedging_error_histogram(
        self,
        ax,
        baseline_hedging_error: torch.Tensor,
        mmcc_hedging_error: torch.Tensor,
    ) -> None:
        baseline = baseline_hedging_error.detach().cpu().numpy()
        mmcc = mmcc_hedging_error.detach().cpu().numpy()

        ax.hist(baseline, bins=30, alpha=0.55, density=True, label="Baseline")
        ax.hist(mmcc, bins=30, alpha=0.55, density=True, label="MMCC")

        ax.set_title("Distribution of hedging error")
        ax.set_xlabel(r"$W_N - H$")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()

    def _print_summary(
        self,
        baseline_metrics: dict[str, float],
        mmcc_metrics: dict[str, float],
    ) -> None:
        baseline_mse = baseline_metrics["mse"]
        mmcc_mse = mmcc_metrics["mse"]
        improvement = 100.0 * (baseline_mse - mmcc_mse) / max(abs(baseline_mse), 1e-12)

        print("\nRésumé rapide")
        print(f"  Baseline MSE : {baseline_mse:.6f}")
        print(f"  MMCC MSE     : {mmcc_mse:.6f}")
        print(f"  Gain relatif : {improvement:.2f}%")

    def _plot_mmcc_gap_to_best(
        self,
        mmcc_history: list[MMCCLog],
        filename: str = "mmcc_gap_to_best.png",
    ) -> None:
        """
        Plot the convergence gap J_k - min_j J_j to visualize convergence speed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for visualization. "
                "Install it with: pip install matplotlib"
            ) from exc

        if not mmcc_history:
            raise ValueError("mmcc_history must not be empty.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        iterations = [log.iteration for log in mmcc_history]
        values = torch.tensor([log.post_loss for log in mmcc_history], dtype=torch.float64)
        best_value = torch.min(values)
        gap = values - best_value

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(iterations, gap.numpy(), marker="*", linewidth=1.0, label="MMCC gap to best")

        if torch.all(gap[1:] > 0):
            ax.set_yscale("log")

        ax.set_title("MMCC convergence gap")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$J_k - \min_j J_j$")
        ax.grid(alpha=0.25)
        ax.legend()

        output_path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)