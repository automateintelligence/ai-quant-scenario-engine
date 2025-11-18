"""
Diagnostic plotting for distribution fit quality assessment.

Generates visualizations of fitted distributions against empirical data with
quality metrics and parameter legends.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from quant_scenario_engine.distributions.distribution_audit import ModelSpec
from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.utils.logging import get_logger

log = get_logger(__name__, component="fit_diagnostics")


def plot_distribution_fits(
    returns: np.ndarray,
    fit_results: Sequence[FitResult],
    candidate_models: Sequence[ModelSpec],
    symbol: str = "UNKNOWN",
    output_path: Optional[Path] = None,
    show_plot: bool = True,
) -> Figure:
    """
    Generate comprehensive fit diagnostic plots for all candidate distributions.

    Creates a multi-panel figure showing:
    1. PDF overlay: Empirical histogram + fitted distribution PDFs
    2. CDF comparison: Empirical CDF vs fitted CDFs
    3. Q-Q plots: Quantile-quantile plots for each distribution
    4. Tail focus: Zoomed view of left tail (losses)

    Parameters
    ----------
    returns : np.ndarray
        Historical log returns used for fitting
    fit_results : Sequence[FitResult]
        Fit results containing parameters and quality metrics
    candidate_models : Sequence[ModelSpec]
        Model specifications with fitted instances
    symbol : str
        Asset symbol for plot title
    output_path : Optional[Path]
        If provided, save figure to this path
    show_plot : bool
        If True, display plot interactively

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Distribution Fit Diagnostics: {symbol}", fontsize=16, fontweight="bold")

    ax_pdf = axes[0, 0]
    ax_cdf = axes[0, 1]
    ax_qq = axes[1, 0]
    ax_tail = axes[1, 1]

    # Colorblind-friendly color palette (Wong palette + line styles + markers)
    # Each model gets a unique combination of color, line style, and marker
    model_styles = {
        "laplace": {
            "color": "#0072B2",      # Blue
            "linestyle": "-",         # Solid
            "marker": "o",           # Circle
            "markersize": 6,
        },
        "student_t": {
            "color": "#E69F00",      # Orange
            "linestyle": "--",        # Dashed
            "marker": "s",           # Square
            "markersize": 6,
        },
        "garch_t": {
            "color": "#009E73",      # Green
            "linestyle": "-.",        # Dash-dot
            "marker": "^",           # Triangle up
            "markersize": 6,
        },
    }

    # Default style for unknown models
    default_style = {
        "color": "#CC79A7",      # Purple
        "linestyle": ":",         # Dotted
        "marker": "D",           # Diamond
        "markersize": 6,
    }

    # Empirical data
    hist_data = returns
    x_range = np.linspace(returns.min(), returns.max(), 500)

    # --- Panel 1: PDF Overlay ---
    ax_pdf.hist(hist_data, bins=50, density=True, alpha=0.3, color="gray", label="Empirical")

    for spec, fr in zip(candidate_models, fit_results):
        if not fr.fit_success:
            continue

        try:
            # Sample from fitted distribution to estimate PDF
            fitter = spec.cls
            samples = fitter.sample(n_paths=10000, n_steps=1).flatten()

            # Kernel density estimate
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(samples)
            pdf_vals = kde(x_range)

            # Get style for this model
            style = model_styles.get(spec.name, default_style)
            label = _format_legend_label(spec.name, fr)
            ax_pdf.plot(
                x_range, pdf_vals,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                alpha=0.9
            )
        except Exception as e:
            log.warning(f"Failed to plot PDF for {spec.name}: {e}")

    ax_pdf.set_xlabel("Log Return")
    ax_pdf.set_ylabel("Density")
    ax_pdf.set_title("Probability Density Functions")
    ax_pdf.legend(loc="upper left", fontsize=8)
    ax_pdf.grid(True, alpha=0.3)

    # --- Panel 2: CDF Comparison ---
    # Empirical CDF
    sorted_returns = np.sort(hist_data)
    empirical_cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    ax_cdf.plot(sorted_returns, empirical_cdf, label="Empirical", color="black",
                linewidth=2, alpha=0.7)

    for spec, fr in zip(candidate_models, fit_results):
        if not fr.fit_success:
            continue

        try:
            # Generate CDF from samples
            fitter = spec.cls
            samples = fitter.sample(n_paths=50000, n_steps=1).flatten()
            sorted_samples = np.sort(samples)
            model_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

            # Interpolate to common x-range for plotting
            cdf_interp = np.interp(sorted_returns, sorted_samples, model_cdf)

            # Get style for this model
            style = model_styles.get(spec.name, default_style)
            ax_cdf.plot(
                sorted_returns, cdf_interp,
                label=spec.name.capitalize(),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                alpha=0.9
            )
        except Exception as e:
            log.warning(f"Failed to plot CDF for {spec.name}: {e}")

    ax_cdf.set_xlabel("Log Return")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_title("Cumulative Distribution Functions")
    ax_cdf.legend(loc="lower right", fontsize=9)
    ax_cdf.grid(True, alpha=0.3)

    # --- Panel 3: Q-Q Plots ---
    quantiles = np.linspace(0.01, 0.99, 100)
    empirical_quantiles = np.quantile(hist_data, quantiles)

    for spec, fr in zip(candidate_models, fit_results):
        if not fr.fit_success:
            continue

        try:
            fitter = spec.cls
            samples = fitter.sample(n_paths=50000, n_steps=1).flatten()
            model_quantiles = np.quantile(samples, quantiles)

            # Get style for this model
            style = model_styles.get(spec.name, default_style)
            ax_qq.scatter(
                empirical_quantiles, model_quantiles,
                label=spec.name.capitalize(),
                color=style["color"],
                marker=style["marker"],
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5
            )
        except Exception as e:
            log.warning(f"Failed to plot Q-Q for {spec.name}: {e}")

    # Perfect fit reference line
    qq_min = min(empirical_quantiles.min(), model_quantiles.min())
    qq_max = max(empirical_quantiles.max(), model_quantiles.max())
    ax_qq.plot([qq_min, qq_max], [qq_min, qq_max], "k--", linewidth=1.5,
               alpha=0.5, label="Perfect Fit")

    ax_qq.set_xlabel("Empirical Quantiles")
    ax_qq.set_ylabel("Model Quantiles")
    ax_qq.set_title("Q-Q Plots (All Distributions)")
    ax_qq.legend(loc="lower right", fontsize=9)
    ax_qq.grid(True, alpha=0.3)
    ax_qq.set_aspect("equal", adjustable="box")

    # --- Panel 4: Left Tail Focus ---
    # Shows how well the model (fitted to all data) captures the left tail behavior
    # Use full data histogram but zoom x-axis to tail region
    tail_threshold = np.quantile(hist_data, 0.10)

    # Plot histogram of ALL data (not just tail) so scale matches model PDFs
    ax_tail.hist(hist_data, bins=50, density=True, alpha=0.3, color="darkred",
                 label="Empirical (All Data)")

    # Define tail x-range for zooming and model evaluation
    tail_min = hist_data.min()
    tail_x_range = np.linspace(tail_min, tail_threshold, 200)

    for spec, fr in zip(candidate_models, fit_results):
        if not fr.fit_success:
            continue

        try:
            # Use actual PDF function instead of sample+KDE for better accuracy
            if spec.name == "laplace":
                from scipy.stats import laplace
                loc = fr.params.get("loc", 0.0)
                scale = fr.params.get("scale", 1.0)
                tail_pdf = laplace.pdf(tail_x_range, loc=loc, scale=scale)

            elif spec.name == "student_t":
                from scipy.stats import t
                df = fr.params.get("df", 5.0)
                loc = fr.params.get("loc", 0.0)
                scale = fr.params.get("scale", 1.0)
                tail_pdf = t.pdf(tail_x_range, df=df, loc=loc, scale=scale)

            elif spec.name == "garch_t":
                # GARCH-t is conditional, so we sample and use larger sample for better tail estimate
                fitter = spec.cls
                samples = fitter.sample(n_paths=100000, n_steps=1).flatten()

                # Use KDE with scott's bandwidth on full sample, then evaluate in tail
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(samples, bw_method='scott')
                tail_pdf = kde(tail_x_range)
            else:
                # Unknown model - skip
                continue

            # Get style for this model
            style = model_styles.get(spec.name, default_style)
            ax_tail.plot(
                tail_x_range, tail_pdf,
                label=spec.name.capitalize(),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                alpha=0.9
            )
        except Exception as e:
            log.warning(f"Failed to plot tail for {spec.name}: {e}")

    ax_tail.set_xlabel("Log Return")
    ax_tail.set_ylabel("Density")
    ax_tail.set_title("Left Tail Focus (Bottom 10% - Extreme Losses)")
    ax_tail.set_xlim(tail_min, tail_threshold)  # Zoom x-axis to tail region only
    ax_tail.legend(loc="upper left", fontsize=9)
    ax_tail.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved fit diagnostic plot to {output_path}")

    # Show if requested
    if show_plot:
        plt.show()

    return fig


def _format_legend_label(model_name: str, fit_result: FitResult) -> str:
    """
    Format a legend label with model name, parameters, and quality metrics.

    Example output:
    "Laplace (μ=0.001, σ=0.015) | AIC=1234.5 | HT=✓"
    """
    params_str = ", ".join(f"{k}={v:.4f}" for k, v in fit_result.params.items())

    # Format quality metrics
    aic_str = f"AIC={fit_result.aic:.1f}" if np.isfinite(fit_result.aic) else "AIC=inf"
    bic_str = f"BIC={fit_result.bic:.1f}" if np.isfinite(fit_result.bic) else "BIC=inf"
    ht_indicator = "✓" if fit_result.heavy_tailed else "✗"

    label = (
        f"{model_name.capitalize()} ({params_str})\n"
        f"{aic_str} | {bic_str} | HeavyTail={ht_indicator}"
    )

    return label


__all__ = ["plot_distribution_fits"]
