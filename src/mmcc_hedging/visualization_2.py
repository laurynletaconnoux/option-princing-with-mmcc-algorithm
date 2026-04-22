"""Secondary visualisation class – policy evolution plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mmcc_hedging.hedging import HedgingTrajectory


@dataclass(slots=True)
class Visualizer2:
    """Produces visualisation plots focused on policy behaviour."""

    output_dir: Path


    def plot_policy_evolution(
        self,
        baseline_trajectory: HedgingTrajectory,
        mmcc_trajectory: HedgingTrajectory,
        *,
        num_paths_to_plot: int = 30,
    ) -> None:
        """Plot the evolution of the hedge position (policy output) over time.

        For each of the two strategies (baseline and MMCC), a fan of
        individual paths is drawn in light colour together with the
        cross-sectional mean and a ±1-std band.

        ``target_position`` has shape ``(N, P)`` – one control per
        hedging date (t_0 … t_{N-1}) per path.  We align it with
        ``time[:N]`` from the trajectory.

        The figure is saved to ``output_dir/policy_evolution.png``.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib est requis pour la visualisation. "
                "Installe-le avec : pip install matplotlib"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 2, figure=fig)

        ax_base = fig.add_subplot(gs[0])
        ax_mmcc = fig.add_subplot(gs[1], sharey=ax_base)

        self._draw_policy_panel(
            ax_base,
            baseline_trajectory,
            num_paths_to_plot,
            title="Baseline – position evolution",
            color="steelblue",
        )
        self._draw_policy_panel(
            ax_mmcc,
            mmcc_trajectory,
            num_paths_to_plot,
            title="MMCC – position evolution",
            color="darkorange",
        )

        # Remove duplicate y-label on the right panel
        ax_mmcc.set_ylabel("")

        fig.suptitle("Hedging policy evolution", fontsize=13, y=1.01)
        fig.tight_layout()

        out_path = self.output_dir / "policy_evolution.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer2] Figure saved: {out_path}")

    def _draw_policy_panel(
        self,
        ax,
        trajectory: HedgingTrajectory,
        num_paths_to_plot: int,
        title: str,
        color: str,
    ) -> None:
        """Draw one policy-evolution panel on *ax*."""
        import numpy as np

        # target_position : (N, P) – controls applied at t_0 … t_{N-1}
        # time            : (N+1,) – t_0 … t_N
        target = trajectory.target_position.detach().cpu().numpy()  # (N, P)
        time = trajectory.time.detach().cpu().numpy()               # (N+1,)

        N, P = target.shape
        t = time[:N]  # align with the N control dates

        n_plot = min(num_paths_to_plot, P)

        # Individual paths (thin, transparent)
        for i in range(n_plot):
            ax.plot(t, target[:, i], color=color, alpha=0.15, linewidth=0.8)

        # Cross-sectional statistics (all paths)
        mean = target.mean(axis=1)
        std = target.std(axis=1)

        ax.plot(t, mean, color=color, linewidth=2.0, label="Mean")
        ax.fill_between(
            t,
            mean - std,
            mean + std,
            color=color,
            alpha=0.25,
            label="±1 std",
        )

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Target position $q_n$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)


    def plot_hedging_error(
        self,
        baseline_trajectory: HedgingTrajectory,
        mmcc_trajectory: HedgingTrajectory,
    ) -> None:
        """Plot hedging-error diagnostics for baseline and MMCC.

        The figure contains three panels:

        * **Histogramme** – overlapping distributions of $W_N - H$ for
          both strategies.
        * **Boîte à moustaches** – side-by-side box-plots showing median,
          IQR, and outliers.
        * **Métriques clés** – a text table with mean, std, MSE and the
          5 % / 50 % / 95 % quantiles for each strategy.

        The figure is saved to ``output_dir/hedging_error.png``.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib est requis pour la visualisation. "
                "Installe-le avec : pip install matplotlib"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        base_err = baseline_trajectory.hedging_error.detach().cpu().numpy()
        mmcc_err = mmcc_trajectory.hedging_error.detach().cpu().numpy()

        def _stats(err: "np.ndarray") -> dict:
            return {
                "Mean": float(err.mean()),
                "Std dev": float(err.std()),
                "MSE": float((err**2).mean()),
                "Q5%": float(np.quantile(err, 0.05)),
                "Median": float(np.median(err)),
                "Q95%": float(np.quantile(err, 0.95)),
            }

        base_stats = _stats(base_err)
        mmcc_stats = _stats(mmcc_err)

        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 1, 1.2])

        ax_hist = fig.add_subplot(gs[0])
        ax_box = fig.add_subplot(gs[1])
        ax_txt = fig.add_subplot(gs[2])

        # ── Histogramme ──────────────────────────────────────────────────
        all_vals = np.concatenate([base_err, mmcc_err])
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)

        ax_hist.hist(base_err, bins=bins, color="steelblue", alpha=0.55,
                     label="Baseline", density=True)
        ax_hist.hist(mmcc_err, bins=bins, color="darkorange", alpha=0.55,
                     label="MMCC", density=True)

        for val, color in [(base_stats["Median"], "steelblue"),
                           (mmcc_stats["Median"], "darkorange")]:
            ax_hist.axvline(val, color=color, linestyle="--", linewidth=1.5)

        ax_hist.axvline(0, color="black", linestyle=":", linewidth=1.0,
                        label="Zero")
        ax_hist.set_title("Hedging error distribution")
        ax_hist.set_xlabel("$W_N - H$")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(alpha=0.25)

        # ── Boîte à moustaches ───────────────────────────────────────────
        bp = ax_box.boxplot(
            [base_err, mmcc_err],
            labels=["Baseline", "MMCC"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )
        colors = ["steelblue", "darkorange"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax_box.axhline(0, color="black", linestyle=":", linewidth=1.0)
        ax_box.set_title("Box plot")
        ax_box.set_ylabel("$W_N - H$")
        ax_box.grid(axis="y", alpha=0.25)

        # ── Tableau de métriques ─────────────────────────────────────────
        ax_txt.axis("off")
        row_labels = list(base_stats.keys())
        cell_text = [
            [f"{base_stats[k]:.4f}", f"{mmcc_stats[k]:.4f}"]
            for k in row_labels
        ]
        table = ax_txt.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=["Baseline", "MMCC"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax_txt.set_title("Key metrics", pad=12)

        fig.suptitle("Hedging error — Baseline vs MMCC", fontsize=13, y=1.01)
        fig.tight_layout()

        out_path = self.output_dir / "hedging_error.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer2] Figure saved: {out_path}")

    # ------------------------------------------------------------------ #
    # Convergence                                                          #
    # ------------------------------------------------------------------ #

    def plot_convergence(
        self,
        baseline_losses: list[float],
        mmcc_history: list,
    ) -> None:
        """Plot training convergence curves for baseline and MMCC.

        The figure contains two panels:

        * **Baseline** – MSE loss per epoch.
        * **MMCC** – pre-update and post-update loss per outer iteration,
          plus one curve per hedging date showing the date-level sub-problem
          loss.

        The figure is saved to ``output_dir/convergence.png``.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib est requis pour la visualisation. "
                "Installe-le avec : pip install matplotlib"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 2, figure=fig)

        ax_base = fig.add_subplot(gs[0])
        ax_mmcc = fig.add_subplot(gs[1])

        # ── Baseline ─────────────────────────────────────────────────────
        epochs = np.arange(1, len(baseline_losses) + 1)
        ax_base.plot(epochs, baseline_losses, color="steelblue",
                     linewidth=1.8, marker="o", markersize=4, label="MSE loss")
        ax_base.set_title("Baseline – convergence")
        ax_base.set_xlabel("Epoch")
        ax_base.set_ylabel("MSE loss")
        ax_base.legend(fontsize=8)
        ax_base.grid(alpha=0.25)

        # ── MMCC ─────────────────────────────────────────────────────────
        if mmcc_history:
            iters = np.array([log.iteration + 1 for log in mmcc_history])
            pre   = np.array([log.pre_loss   for log in mmcc_history])
            post  = np.array([log.post_loss  for log in mmcc_history])

            ax_mmcc.plot(iters, pre,  color="darkorange", linewidth=1.8,
                         linestyle="--", marker="s", markersize=4,
                         label="Pre-update")
            ax_mmcc.plot(iters, post, color="darkorange", linewidth=2.0,
                         marker="o", markersize=4, label="Post-update")

            # One curve per hedging date
            date_keys = sorted(mmcc_history[0].date_losses.keys(), reverse=True)
            cmap = plt.cm.get_cmap("Blues", len(date_keys) + 2)
            for i, t in enumerate(date_keys):
                date_vals = np.array([log.date_losses[t] for log in mmcc_history])
                ax_mmcc.plot(iters, date_vals, color=cmap(i + 2), linewidth=1.2,
                             linestyle=":", alpha=0.8, label=f"Date t={t}")

        ax_mmcc.set_title("MMCC – convergence")
        ax_mmcc.set_xlabel("Iteration")
        ax_mmcc.set_ylabel("MSE loss")
        ax_mmcc.legend(fontsize=7)
        ax_mmcc.grid(alpha=0.25)

        fig.suptitle("Convergence speed — Baseline vs MMCC", fontsize=13, y=1.01)
        fig.tight_layout()

        out_path = self.output_dir / "convergence.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer2] Figure saved: {out_path}")
