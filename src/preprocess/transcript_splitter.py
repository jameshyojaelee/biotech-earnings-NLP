"""Split transcripts into prepared remarks and Q&A sections."""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd


def normalize_transcript(text: str) -> str:
    """Clean whitespace and standardize markers."""
    if text is None:
        return ""
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub("\n{2,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _find_split_index(text: str) -> int:
    markers = [
        r"question[- ]and[- ]answer",
        r"question\s+and\s+answer",
        r"q\s*&\s*a",
        r"q&a",
    ]
    for pattern in markers:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.start()

    # Fallback: look for analyst or operator cues that often introduce Q&A.
    fallback = re.search(r"(operator:|analyst\s+q:|analyst:)", text, flags=re.IGNORECASE)
    if fallback:
        return fallback.start()
    return -1


def find_qa_start(text: str, normalized: bool = False) -> int:
    """Return the character index where Q&A likely starts, or -1 if unknown."""
    cleaned = text if normalized else normalize_transcript(text)
    return _find_split_index(cleaned)


def split_prepared_and_qa(text: str) -> Tuple[str, str]:
    """Return prepared remarks and Q&A text segments."""
    normalized = normalize_transcript(text)
    idx = _find_split_index(normalized)
    if idx == -1:
        return normalized, ""
    prepared = normalized[:idx].strip()
    qa = normalized[idx:].strip()
    return prepared, qa


def add_sections_to_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Add prepared_text and qa_text columns to the events DataFrame."""
    prepared, qa = [], []
    for _, row in events_df.iterrows():
        prep_text, qa_text = split_prepared_and_qa(row.get("transcript", ""))
        prepared.append(prep_text)
        qa.append(qa_text)
    events_df = events_df.copy()
    events_df["prepared_text"] = prepared
    events_df["qa_text"] = qa
    return events_df
