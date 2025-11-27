"""Compute FinBERT sentiment features for prepared remarks and Q&A sections."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .sentiment_finbert import add_sentiment_features


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add FinBERT sentiment features to events.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    input_path = Path(config.get("events_with_sections_path", "data_processed/events_with_sections.parquet"))
    output_path = Path(config.get("events_with_features_path", "data_processed/events_with_features.parquet"))

    events = pd.read_parquet(input_path)
    events_with_sentiment = add_sentiment_features(events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_with_sentiment.to_parquet(output_path, index=False)
    print(f"Saved events with sentiment to {output_path}")


if __name__ == "__main__":
    main()
