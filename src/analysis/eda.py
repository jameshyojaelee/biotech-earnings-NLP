"""Exploratory data analysis helpers for earnings NLP."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

COLORS = {
    "primary": "#64ffda",
    "accent": "#ff7edb",
    "muted": "#6b7280",
    "bg": "#0b1220",
    "panel": "#0f172a",
    "grid": "#1f2937",
    "text": "#e5e7eb",
}


def set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": COLORS["grid"],
            "grid.color": COLORS["grid"],
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
        },
    )
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def load_events_with_features(config_path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    path = Path(config.get("events_with_features_path", "data_processed/events_with_features.parquet"))
    return pd.read_parquet(path)


def _format_percent_axis(ax, axis: str = "x") -> None:
    formatter = PercentFormatter(xmax=1, decimals=1)
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(formatter)


def plot_histograms(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    show: bool = True,
    save_dir: Optional[Path] = None,
) -> None:
    set_plot_style()
    cols = cols or [c for c in df.columns if c.startswith("ret_") or c.startswith("abn_ret_")]
    if not cols:
        print("No return columns to plot.")
        return
    for col in cols:
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(series, kde=True, ax=ax, color=COLORS["primary"], edgecolor=COLORS["grid"])
        ax.axvline(0, color=COLORS["muted"], linestyle="--", linewidth=1)
        mean_val = series.mean()
        median_val = series.median()
        ax.axvline(mean_val, color=COLORS["accent"], linestyle="-", linewidth=1.5, label=f"Mean {mean_val:.2%}")
        ax.axvline(median_val, color=COLORS["muted"], linestyle=":", linewidth=1.5, label=f"Median {median_val:.2%}")
        _format_percent_axis(ax, "x")
        ax.set_title(f"{col} distribution (n={len(series)})", color=COLORS["text"])
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.legend(frameon=False)
        plt.tight_layout()
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"hist_{col}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_scatter_sentiment_vs_returns(
    df: pd.DataFrame,
    return_col: str = "abn_ret_5d",
    show: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    set_plot_style()
    if return_col not in df:
        print(f"Missing return column {return_col}")
        return
    data = df[["qa_sent_score", return_col]].dropna()
    if data.empty:
        print("No data to plot scatter.")
        return
    corr = data["qa_sent_score"].corr(data[return_col])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.regplot(
        data=data,
        x="qa_sent_score",
        y=return_col,
        scatter_kws={"alpha": 0.7, "color": COLORS["primary"]},
        line_kws={"color": COLORS["accent"]},
        ax=ax,
    )
    ax.axhline(0, color=COLORS["muted"], linestyle="--", linewidth=1)
    ax.axvline(0, color=COLORS["muted"], linestyle="--", linewidth=1)
    _format_percent_axis(ax, "y")
    title_corr = f", r={corr:.2f}" if pd.notna(corr) else ""
    ax.set_title(f"QA sentiment vs {return_col} (n={len(data)}{title_corr})", color=COLORS["text"])
    ax.set_xlabel("QA sentiment score")
    ax.set_ylabel(return_col.replace("_", " "))
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_box_by_sentiment_bucket(
    df: pd.DataFrame,
    return_col: str = "abn_ret_5d",
    show: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    set_plot_style()
    if return_col not in df:
        print(f"Missing return column {return_col}")
        return
    subset = df.dropna(subset=["qa_sent_score", return_col]).copy()
    if subset.empty:
        print("No data to plot sentiment buckets.")
        return
    try:
        subset["sent_bucket"] = pd.qcut(subset["qa_sent_score"], 3, labels=["Low", "Mid", "High"])
    except ValueError:
        print("Not enough unique sentiment values to create buckets.")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(
        data=subset,
        x="sent_bucket",
        y=return_col,
        ax=ax,
        palette=[COLORS["primary"], COLORS["muted"], COLORS["accent"]],
    )
    sns.stripplot(
        data=subset,
        x="sent_bucket",
        y=return_col,
        ax=ax,
        color=COLORS["text"],
        alpha=0.4,
        jitter=0.2,
        size=3,
    )
    ax.axhline(0, color=COLORS["muted"], linestyle="--", linewidth=1)
    _format_percent_axis(ax, "y")
    ax.set_title(f"{return_col} by sentiment tercile (n={len(subset)})", color=COLORS["text"])
    ax.set_xlabel("QA sentiment tercile")
    ax.set_ylabel(return_col.replace("_", " "))
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    events = load_events_with_features()
    plot_histograms(events)
    plot_scatter_sentiment_vs_returns(events)
    plot_box_by_sentiment_bucket(events)
