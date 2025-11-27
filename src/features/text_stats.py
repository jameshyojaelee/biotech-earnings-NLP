"""Text stats for hedging and biotech risk language in Q&A sections."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

HEDGE_TERMS = [
    "may",
    "might",
    "could",
    "uncertain",
    "uncertainty",
    "visibility",
    "approximately",
    "around",
    "potentially",
    "possible",
    "expect",
    "believe",
    "should",
]

RISK_TERMS = [
    "fda",
    "trial hold",
    "clinical hold",
    "adverse event",
    "adverse events",
    "safety signal",
    "black box",
    "recall",
    "delay",
    "setback",
    "pdufa",
    "crl",
    "phase i",
    "phase ii",
    "phase iii",
    "enrollment",
    "dropout",
    "serious adverse",
]


def count_terms(text: str, term_list: Iterable[str]) -> int:
    """Count occurrences of terms in a case-insensitive way."""
    if not text:
        return 0
    lower = text.lower()
    count = 0
    for term in term_list:
        pattern = r"\b" + re.escape(term) + r"\b"
        count += len(re.findall(pattern, lower))
    return count


def compute_qa_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hedging/risk counts and rates for Q&A text."""
    df = df.copy()
    qa_word_counts = []
    hedge_counts = []
    risk_counts = []

    texts = df["qa_text"] if "qa_text" in df else pd.Series([], dtype=str)
    for text in texts:
        words = text.split()
        qa_word_counts.append(len(words))
        hedge_counts.append(count_terms(text, HEDGE_TERMS))
        risk_counts.append(count_terms(text, RISK_TERMS))

    df["qa_word_count"] = qa_word_counts
    df["qa_hedge_terms"] = hedge_counts
    df["qa_risk_terms"] = risk_counts

    df["qa_hedge_rate"] = df["qa_hedge_terms"] / df["qa_word_count"].replace(0, pd.NA)
    df["qa_risk_rate"] = df["qa_risk_terms"] / df["qa_word_count"].replace(0, pd.NA)
    df["qa_hedge_rate"] = df["qa_hedge_rate"].fillna(0.0)
    df["qa_risk_rate"] = df["qa_risk_rate"].fillna(0.0)

    return df
