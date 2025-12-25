"""Create a small gold-labeling set for signal extraction evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_event_id(ticker: str, earnings_date) -> str:
    if pd.isna(earnings_date):
        return f"{ticker}|unknown"
    date_str = pd.to_datetime(earnings_date).date().isoformat()
    return f"{ticker}|{date_str}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a gold labeling set for signal extraction.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of calls to sample")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--min-qa-words", type=int, default=200, help="Minimum QA word count")
    parser.add_argument("--max-text-chars", type=int, default=4000, help="Max chars for each text field")
    parser.add_argument("--output", default="", help="Optional output path override")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    sections_path = Path(config.get("events_with_sections_path", "data_processed/events_with_sections.parquet"))
    df = pd.read_parquet(sections_path)

    df["qa_word_count"] = df["qa_text"].fillna("").str.split().str.len()
    eligible = df[df["qa_word_count"] >= args.min_qa_words].copy()
    if eligible.empty:
        eligible = df.copy()

    sample_n = min(args.n_samples, len(eligible))
    sampled = eligible.sample(n=sample_n, random_state=args.seed).copy()
    sampled["event_id"] = sampled.apply(
        lambda row: _build_event_id(row.get("ticker", ""), row.get("earnings_date")), axis=1
    )

    labeled = sampled[
        ["event_id", "ticker", "company", "earnings_date", "prepared_text", "qa_text", "qa_word_count"]
    ].copy()
    labeled["prepared_text"] = labeled["prepared_text"].fillna("").str.slice(0, args.max_text_chars)
    labeled["qa_text"] = labeled["qa_text"].fillna("").str.slice(0, args.max_text_chars)

    for signal in ["trial_update", "guidance_change", "safety_signal", "regulatory_mention"]:
        labeled[signal] = ""

    labeled["notes"] = ""

    output_path = Path(args.output) if args.output else Path(config.get("gold_labels_path", "data_processed/gold/gold_labels.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)
    print(f"Saved gold labeling template to {output_path}")


if __name__ == "__main__":
    main()
