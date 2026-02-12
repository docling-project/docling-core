"""Visualization utilities for collection statistics.

This module provides utilities for creating charts from CollectionStats data.
Requires matplotlib to be installed (available with 'examples' extra).

Install with: pip install docling-core[examples]
"""

from pathlib import Path
from typing import Literal

try:
    import matplotlib.figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from docling_core.transforms.profiler.doc_profiler import CollectionStats, Histogram


class StatsVisualizer:
    """Visualizer for creating charts from CollectionStats data."""

    @staticmethod
    def _check_matplotlib() -> None:
        """Check if matplotlib is available."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install docling-core[examples]"
            )

    @staticmethod
    def plot_histogram(
        histogram: Histogram,
        title: str = "Distribution",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        color: str = "steelblue",
        figsize: tuple[int, int] = (10, 6),
    ) -> "matplotlib.figure.Figure":
        """Plot a histogram from Histogram data.

        Args:
            histogram: Histogram object containing bins and frequencies
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Bar color
            figsize: Figure size as (width, height)

        Returns:
            matplotlib Figure object

        Raises:
            ImportError: If matplotlib is not installed
        """
        StatsVisualizer._check_matplotlib()

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate bin centers for plotting
        bins = histogram.bins
        frequencies = histogram.frequencies

        if len(bins) > 0 and len(frequencies) > 0:
            # bins has n+1 edges, frequencies has n values
            bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(frequencies))]
            bin_width = histogram.bin_width

            ax.bar(bin_centers, frequencies, width=bin_width * 0.9, color=color, edgecolor="black", alpha=0.7)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_deciles(
        deciles: list[float],
        title: str = "Decile Distribution",
        ylabel: str = "Value",
        color: str = "coral",
        figsize: tuple[int, int] = (10, 6),
    ) -> "matplotlib.figure.Figure":
        """Plot deciles as a line chart.

        Args:
            deciles: List of 9 decile values [d1, d2, ..., d9] (10th, 20th, ..., 90th percentiles)
            title: Plot title
            ylabel: Y-axis label
            color: Line color
            figsize: Figure size as (width, height)

        Returns:
            matplotlib Figure object

        Raises:
            ImportError: If matplotlib is not installed
        """
        StatsVisualizer._check_matplotlib()

        fig, ax = plt.subplots(figsize=figsize)

        decile_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        percentile_labels = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        ax.plot(decile_labels, deciles, marker="o", linewidth=2, markersize=8, color=color)
        ax.fill_between(decile_labels, deciles, alpha=0.3, color=color)

        # Highlight median (d5 = 50th percentile)
        ax.axvline(x=5, color="red", linestyle="--", alpha=0.5, label="Median (d5)")

        ax.set_xlabel("Decile", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(decile_labels)
        ax.set_xticklabels([f"d{d} (p{p})" for d, p in zip(decile_labels, percentile_labels)])
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_collection_overview(
        stats: CollectionStats,
        metrics: list[Literal["pages", "tables", "pictures", "texts"]] | None = None,
        figsize: tuple[int, int] = (16, 10),
    ) -> "matplotlib.figure.Figure":
        """Create a comprehensive overview plot with multiple histograms.

        Args:
            stats: CollectionStats object
            metrics: List of metrics to plot. If None, plots all available metrics.
            figsize: Figure size as (width, height)

        Returns:
            matplotlib Figure object with subplots

        Raises:
            ImportError: If matplotlib is not installed
        """
        StatsVisualizer._check_matplotlib()

        if metrics is None:
            metrics = ["pages", "tables", "pictures", "texts"]

        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        metric_config = {
            "pages": {
                "histogram": stats.histogram_pages,
                "title": "Pages per Document",
                "color": "steelblue",
            },
            "tables": {
                "histogram": stats.histogram_tables,
                "title": "Tables per Document",
                "color": "forestgreen",
            },
            "pictures": {
                "histogram": stats.histogram_pictures,
                "title": "Pictures per Document",
                "color": "coral",
            },
            "texts": {
                "histogram": stats.histogram_texts,
                "title": "Text Items per Document",
                "color": "mediumpurple",
            },
        }

        for idx, metric in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            config = metric_config[metric]
            histogram = config["histogram"]
            bins = histogram.bins
            frequencies = histogram.frequencies

            if len(bins) > 0 and len(frequencies) > 0:
                bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(frequencies))]
                bin_width = histogram.bin_width

                ax.bar(
                    bin_centers,
                    frequencies,
                    width=bin_width * 0.9,
                    color=config["color"],
                    edgecolor="black",
                    alpha=0.7,
                )

            ax.set_xlabel("Count", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(config["title"], fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        fig.suptitle(
            f"Collection Statistics Overview ({stats.num_documents} documents)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_deciles_comparison(
        stats: CollectionStats,
        metrics: list[Literal["pages", "tables", "pictures", "texts"]] | None = None,
        figsize: tuple[int, int] = (12, 6),
    ) -> "matplotlib.figure.Figure":
        """Create a comparison plot of deciles for multiple metrics.

        Args:
            stats: CollectionStats object
            metrics: List of metrics to plot. If None, plots all available metrics.
            figsize: Figure size as (width, height)

        Returns:
            matplotlib Figure object

        Raises:
            ImportError: If matplotlib is not installed
        """
        StatsVisualizer._check_matplotlib()

        if metrics is None:
            metrics = ["pages", "tables", "pictures", "texts"]

        fig, ax = plt.subplots(figsize=figsize)

        decile_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        metric_config = {
            "pages": {"deciles": stats.deciles_pages, "label": "Pages", "color": "steelblue"},
            "tables": {"deciles": stats.deciles_tables, "label": "Tables", "color": "forestgreen"},
            "pictures": {"deciles": stats.deciles_pictures, "label": "Pictures", "color": "coral"},
            "texts": {"deciles": stats.deciles_texts, "label": "Text Items", "color": "mediumpurple"},
        }

        for metric in metrics:
            config = metric_config[metric]
            ax.plot(
                decile_labels,
                config["deciles"],
                marker="o",
                linewidth=2,
                markersize=6,
                label=config["label"],
                color=config["color"],
            )

        ax.axvline(x=5, color="red", linestyle="--", alpha=0.3, label="Median (d5)")

        ax.set_xlabel("Decile", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Decile Comparison Across Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(decile_labels)
        ax.set_xticklabels([f"d{d}" for d in decile_labels])
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best")

        plt.tight_layout()
        return fig

    @staticmethod
    def save_figure(fig: "matplotlib.figure.Figure", filepath: str | Path, dpi: int = 300) -> None:
        """Save a matplotlib figure to file.

        Args:
            fig: matplotlib Figure object
            filepath: Output file path (supports .png, .pdf, .svg, etc.)
            dpi: Resolution in dots per inch
        """
        StatsVisualizer._check_matplotlib()
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    @staticmethod
    def show_figure(fig: "matplotlib.figure.Figure") -> None:
        """Display a matplotlib figure.

        Args:
            fig: matplotlib Figure object
        """
        StatsVisualizer._check_matplotlib()
        plt.show()

