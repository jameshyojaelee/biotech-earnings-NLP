"""FinBERT-based sentiment scoring for earnings call transcripts."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm
from transformers import pipeline


@lru_cache(maxsize=1)
def load_finbert_pipeline():
    """Load and cache the FinBERT sentiment pipeline."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def chunk_text(text: str, max_tokens: int = 256) -> List[str]:
    """Split text into roughly token-sized chunks to keep inference stable."""
    if not text:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i : i + max_tokens]))
    return chunks


def _aggregate_scores(results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    scores = {"pos": [], "neg": [], "neu": []}
    for res in results:
        label = res["label"].lower()
        score = res["score"]
        if "pos" in label:
            scores["pos"].append(score)
        elif "neg" in label:
            scores["neg"].append(score)
        else:
            scores["neu"].append(score)
    # Use mean to smooth across chunks
    agg = {k: (float(np.mean(v)) if v else 0.0) for k, v in scores.items()}
    agg["sentiment_score"] = agg["pos"] - agg["neg"]
    return agg


def score_text_sentiment(text: str) -> Dict[str, float]:
    """Score a block of text with FinBERT, returning pos/neg/neu and score."""
    if not text:
        return {"pos": 0.0, "neg": 0.0, "neu": 0.0, "sentiment_score": 0.0}

    clf = load_finbert_pipeline()
    chunks = chunk_text(text)
    results = []
    for chunk in chunks:
        output = clf(chunk, truncation=True)
        # pipeline returns a list of predictions
        if isinstance(output, list):
            results.extend(output)
        else:
            results.append(output)
    return _aggregate_scores(results)


def add_sentiment_features(events_df) -> object:
    """Add sentiment features for prepared remarks and Q&A, plus tone shift."""
    events_df = events_df.copy()
    prepared_scores = []
    qa_scores = []

    iterator = tqdm(events_df.itertuples(), total=len(events_df), desc="Scoring sentiment")
    for row in iterator:
        prep_scores = score_text_sentiment(getattr(row, "prepared_text", ""))
        qa_scores_row = score_text_sentiment(getattr(row, "qa_text", ""))
        prepared_scores.append(prep_scores)
        qa_scores.append(qa_scores_row)

    for idx, scores in enumerate(prepared_scores):
        events_df.loc[events_df.index[idx], "prep_sent_pos"] = scores["pos"]
        events_df.loc[events_df.index[idx], "prep_sent_neg"] = scores["neg"]
        events_df.loc[events_df.index[idx], "prep_sent_neu"] = scores["neu"]
        events_df.loc[events_df.index[idx], "prep_sent_score"] = scores["sentiment_score"]

    for idx, scores in enumerate(qa_scores):
        events_df.loc[events_df.index[idx], "qa_sent_pos"] = scores["pos"]
        events_df.loc[events_df.index[idx], "qa_sent_neg"] = scores["neg"]
        events_df.loc[events_df.index[idx], "qa_sent_neu"] = scores["neu"]
        events_df.loc[events_df.index[idx], "qa_sent_score"] = scores["sentiment_score"]

    events_df["tone_shift"] = events_df["qa_sent_score"] - events_df["prep_sent_score"]
    return events_df
