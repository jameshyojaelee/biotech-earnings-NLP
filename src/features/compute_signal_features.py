"""Compute signal extraction features for trial/guidance/safety/regulatory mentions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .signal_extraction import add_signal_features


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add signal extraction features to events.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--text-column", default="qa_text", help="Text column to scan for signals")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    features_path = Path(config.get("events_with_features_path", "data_processed/events_with_features.parquet"))
    sections_path = Path(config.get("events_with_sections_path", "data_processed/events_with_sections.parquet"))
    source_path = features_path if features_path.exists() else sections_path

    df = pd.read_parquet(source_path)
    df_with_signals = add_signal_features(df, text_column=args.text_column)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_signals.to_parquet(features_path, index=False)
    print(f"Saved signal features to {features_path}")


if __name__ == "__main__":
    main()
