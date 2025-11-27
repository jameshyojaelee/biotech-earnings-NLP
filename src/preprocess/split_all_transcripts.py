"""Split all transcripts into prepared remarks and Q&A sections."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .transcript_splitter import add_sections_to_events


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split transcripts into prepared remarks and Q&A sections.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    input_path = Path(config.get("events_with_returns_path", "data_processed/events_with_returns.parquet"))
    output_path = Path(config.get("events_with_sections_path", "data_processed/events_with_sections.parquet"))

    events = pd.read_parquet(input_path)
    events_with_sections = add_sections_to_events(events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_with_sections.to_parquet(output_path, index=False)

    print(f"Saved transcripts with sections to {output_path}")
    print("Example splits:")
    for idx, row in events_with_sections.head(3).iterrows():
        print(f"Ticker {row['ticker']} on {row['earnings_date']}")
        print(f"Prepared remarks preview: {row['prepared_text'][:120]}...")
        print(f"Q&A preview: {row['qa_text'][:120]}...")
        print("---")


if __name__ == "__main__":
    main()
