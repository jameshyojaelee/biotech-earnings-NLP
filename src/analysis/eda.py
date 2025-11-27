"""Exploratory data analysis helpers for earnings NLP."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_events_with_features(config_path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    path = Path(config.get("events_with_features_path", "data_processed/events_with_features.parquet"))
    return pd.read_parquet(path)


def plot_histograms(df: pd.DataFrame) -> None:
    cols = [c for c in df.columns if c.startswith("ret_") or c.startswith("abn_ret_")]
    if not cols:
        print("No return columns to plot.")
        return
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()


def plot_scatter_sentiment_vs_returns(df: pd.DataFrame, return_col: str = "abn_ret_5d") -> None:
    if return_col not in df:
        print(f"Missing return column {return_col}")
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="qa_sent_score", y=return_col, hue="sector" if "sector" in df else None)
    plt.axhline(0, color="grey", linestyle="--", linewidth=1)
    plt.axvline(0, color="grey", linestyle="--", linewidth=1)
    plt.title("Q&A sentiment vs returns")
    plt.tight_layout()
    plt.show()


def plot_box_by_sentiment_bucket(df: pd.DataFrame, return_col: str = "abn_ret_5d") -> None:
    if return_col not in df:
        print(f"Missing return column {return_col}")
        return
    df = df.dropna(subset=["qa_sent_score", return_col]).copy()
    df["sent_bucket"] = pd.qcut(df["qa_sent_score"], 3, labels=["Low", "Mid", "High"])
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="sent_bucket", y=return_col)
    plt.title("Returns by sentiment tercile")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    events = load_events_with_features()
    plot_histograms(events)
    plot_scatter_sentiment_vs_returns(events)
    plot_box_by_sentiment_bucket(events)
