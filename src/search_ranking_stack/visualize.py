"""
Visualization module for search ranking metrics.

Creates:
1. Grouped bar chart comparing all stages (saved as PNG)
2. Rich console table for terminal output
3. ESCI label distribution stacked bar chart

Blog Section: Results visualization
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from .config import RESULTS_DIR

console = Console()

# Friendly display names for metrics
METRIC_DISPLAY_NAMES = {
    "ndcg_cut_10": "NDCG@10",
    "recip_rank": "MRR@10",
    "recall_100": "Recall@100",
}

# Colors for the bar chart
METRIC_COLORS = {
    "ndcg_cut_10": "#3498db",  # Blue
    "recip_rank": "#e67e22",  # Orange
    "recall_100": "#27ae60",  # Green
}

# Colors for label distribution
LABEL_COLORS = {
    "Exact": "#27ae60",  # Green
    "Substitute": "#3498db",  # Blue
    "Complement": "#f39c12",  # Yellow/Orange
    "Irrelevant": "#e74c3c",  # Red
}


def plot_comparison(
    all_metrics: dict[str, dict[str, float]],
    output_path: Path | str | None = None,
    show: bool = False,
) -> Path:
    """
    Create a grouped bar chart comparing metrics across stages.

    Args:
        all_metrics: {stage_name: {metric_name: value}}
        output_path: Where to save the chart (default: results/metrics_comparison.png)
        show: Whether to display the chart interactively

    Returns:
        Path to saved chart
    """
    if output_path is None:
        output_path = RESULTS_DIR / "metrics_comparison.png"
    output_path = Path(output_path)

    # Setup
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    stages = list(all_metrics.keys())
    metrics = list(METRIC_DISPLAY_NAMES.keys())
    n_stages = len(stages)

    # Bar positions
    x = range(n_stages)
    bar_width = 0.25

    # Plot each metric as a group of bars
    for i, metric in enumerate(metrics):
        values = [all_metrics[stage].get(metric, 0) for stage in stages]
        positions = [p + i * bar_width for p in x]

        bars = ax.bar(
            positions,
            values,
            bar_width,
            label=METRIC_DISPLAY_NAMES[metric],
            color=METRIC_COLORS[metric],
            alpha=0.9,
        )

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Customize chart
    ax.set_xlabel("Search Pipeline Stage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Search Ranking Stack: Progressive NDCG@10 Improvement on Amazon ESCI",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # X-axis ticks
    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(stages, rotation=15, ha="right")

    # Y-axis range
    ax.set_ylim(0, 1.0)

    # Legend
    ax.legend(loc="upper left", framealpha=0.95)

    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    console.print(f"\n[bold green]✓[/bold green] Chart saved to: {output_path}")

    if show:
        plt.show()

    plt.close()
    return output_path


def plot_label_distribution(
    label_dist: dict[str, dict[str, float]],
    output_path: Path | str | None = None,
    show: bool = False,
) -> Path:
    """
    Create a stacked bar chart showing ESCI label distribution per stage.

    Args:
        label_dist: {stage_name: {label_name: fraction}}
        output_path: Where to save (default: results/label_distribution.png)
        show: Whether to display interactively

    Returns:
        Path to saved chart
    """
    if output_path is None:
        output_path = RESULTS_DIR / "label_distribution.png"
    output_path = Path(output_path)

    # Setup
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = list(label_dist.keys())
    labels = ["Exact", "Substitute", "Complement", "Irrelevant"]
    x = range(len(stages))
    bar_width = 0.6

    # Stack bars
    bottom = [0.0] * len(stages)
    for label in labels:
        values = [label_dist[stage].get(label, 0) for stage in stages]
        ax.bar(
            x,
            values,
            bar_width,
            label=label,
            color=LABEL_COLORS[label],
            bottom=bottom,
            alpha=0.9,
        )
        # Update bottom for stacking
        bottom = [b + v for b, v in zip(bottom, values)]

    # Customize
    ax.set_xlabel("Search Pipeline Stage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Top-10 Results", fontsize=12, fontweight="bold")
    ax.set_title(
        "ESCI Label Distribution in Top-10 Results by Stage",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)

    # Legend
    ax.legend(loc="upper right", framealpha=0.95)

    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    console.print(f"[bold green]✓[/bold green] Label distribution chart saved to: {output_path}")

    if show:
        plt.show()

    plt.close()
    return output_path


def print_label_distribution(label_dist: dict[str, dict[str, float]]):
    """Print label distribution table to console."""
    table = Table(
        title="Top-10 Label Distribution by Stage",
        title_style="bold cyan",
        header_style="bold",
    )

    table.add_column("Stage", style="bold")
    table.add_column("Exact", justify="right", style="green")
    table.add_column("Substitute", justify="right", style="blue")
    table.add_column("Complement", justify="right", style="yellow")
    table.add_column("Irrelevant", justify="right", style="red")

    for stage, dist in label_dist.items():
        table.add_row(
            stage,
            f"{dist.get('Exact', 0):.1%}",
            f"{dist.get('Substitute', 0):.1%}",
            f"{dist.get('Complement', 0):.1%}",
            f"{dist.get('Irrelevant', 0):.1%}",
        )

    console.print()
    console.print(table)


def print_table(all_metrics: dict[str, dict[str, float]]):
    """
    Print a rich table of metrics to the console.

    Args:
        all_metrics: {stage_name: {metric_name: value}}
    """
    table = Table(
        title="Search Ranking Stack Results",
        title_style="bold cyan",
        header_style="bold",
    )

    # Columns
    table.add_column("Stage", style="bold")
    for metric in METRIC_DISPLAY_NAMES.values():
        table.add_column(metric, justify="right")

    # Rows
    for stage, metrics in all_metrics.items():
        row = [stage]
        for metric_key in METRIC_DISPLAY_NAMES:
            value = metrics.get(metric_key, 0)
            row.append(f"{value:.4f}")
        table.add_row(*row)

    console.print()
    console.print(table)


def save_metrics(all_metrics: dict[str, dict[str, float]], output_path: Path | str | None = None):
    """
    Save metrics to a JSON file.

    Args:
        all_metrics: {stage_name: {metric_name: value}}
        output_path: Where to save (default: results/metrics.json)
    """
    if output_path is None:
        output_path = RESULTS_DIR / "metrics.json"
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    console.print(f"[bold green]✓[/bold green] Metrics saved to: {output_path}")
