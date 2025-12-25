"""Normalize transcripts into speaker segments and extract metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .segment_transcripts import (
    extract_transcript_metadata,
    segment_transcript_text,
    segments_from_structured,
    segments_to_frame,
)


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
    parser = argparse.ArgumentParser(description="Normalize transcripts into speaker segments and metadata.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    returns_path = Path(config.get("events_with_returns_path", "data_processed/events_with_returns.parquet"))
    base_path = Path(config.get("events_base_path", "data_processed/events_base.parquet"))
    input_path = returns_path if returns_path.exists() else base_path

    events = pd.read_parquet(input_path)
    segment_rows = []
    metadata_rows = []

    for _, row in events.iterrows():
        transcript = row.get("transcript", "") or ""
        metadata = extract_transcript_metadata(transcript)
        exec_names = metadata.get("executive_names") or []
        analyst_names = metadata.get("analyst_names") or []

        segments_field = row.get("segments") if isinstance(row, dict) else row.get("segments", None)
        if segments_field is not None and isinstance(segments_field, (list, pd.DataFrame)):
            segments = segments_from_structured(segments_field, exec_names, analyst_names)
        else:
            segments = segment_transcript_text(transcript, exec_names, analyst_names)

        segment_df = segments_to_frame(segments)
        event_id = _build_event_id(row.get("ticker", ""), row.get("earnings_date"))
        if not segment_df.empty:
            segment_df["ticker"] = row.get("ticker")
            segment_df["company"] = row.get("company")
            segment_df["earnings_date"] = row.get("earnings_date")
            segment_df["year"] = row.get("year")
            segment_df["quarter"] = row.get("quarter")
            segment_df["event_id"] = event_id
            segment_rows.append(segment_df)

        has_timestamps = any(seg.start_time for seg in segments)
        speaker_names = {seg.speaker_name for seg in segments if seg.speaker_name}
        metadata_rows.append(
            {
                "event_id": event_id,
                "executive_list_raw": metadata.get("executive_list_raw", ""),
                "analyst_list_raw": metadata.get("analyst_list_raw", ""),
                "executive_names": metadata.get("executive_names", []),
                "analyst_names": metadata.get("analyst_names", []),
                "executive_count": metadata.get("executive_count", 0),
                "analyst_count": metadata.get("analyst_count", 0),
                "segment_count": len(segments),
                "speaker_count": len(speaker_names),
                "has_timestamps": has_timestamps,
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    events_with_metadata = events.copy()
    events_with_metadata["event_id"] = events_with_metadata.apply(
        lambda row: _build_event_id(row.get("ticker", ""), row.get("earnings_date")), axis=1
    )
    events_with_metadata = events_with_metadata.merge(metadata_df, on="event_id", how="left")

    events_output_path = Path(config.get("events_with_metadata_path", "data_processed/events_with_metadata.parquet"))
    segments_output_path = Path(config.get("segments_path", "data_processed/transcript_segments.parquet"))
    events_output_path.parent.mkdir(parents=True, exist_ok=True)
    segments_output_path.parent.mkdir(parents=True, exist_ok=True)

    events_with_metadata.to_parquet(events_output_path, index=False)
    if segment_rows:
        all_segments = pd.concat(segment_rows, ignore_index=True)
        all_segments.to_parquet(segments_output_path, index=False)
    else:
        pd.DataFrame().to_parquet(segments_output_path, index=False)

    print(f"Saved events with metadata to {events_output_path}")
    print(f"Saved transcript segments to {segments_output_path}")


if __name__ == "__main__":
    main()
