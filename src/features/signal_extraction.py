"""Extract biotech-relevant signals from transcript text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd


@dataclass
class SignalMatch:
    signal: str
    phrase: str
    start: int
    end: int
    snippet: str


SIGNAL_PATTERNS: Dict[str, List[str]] = {
    "trial_update": [
        r"\bphase\s+(?:i|ii|iii|iv|1|2|3|4)(?:[ab]?)\s*(?:/|and)?\s*(?:i|ii|iii|iv|1|2|3|4)?\b",
        r"\b(pivotal|registrational)\s+(trial|study)\b",
        r"\btop[- ]line\s+data\b",
        r"\b(data|results)\s+(readout|read-out)\b",
        r"\b(interim|final)\s+analysis\b",
        r"\benroll(?:ment)?\s+(?:complete|completed|finish(?:ed)?|fully)\b",
        r"\bfirst\s+patient\s+(?:dosed|enrolled|treated)\b",
        r"\b(initiated|initiation|start(?:ed)?|launch(?:ed)?)\s+(?:the\s+)?(?:trial|study|enrollment)\b",
        r"\b(dose[- ]escalation|expansion)\s+cohort\b",
    ],
    "guidance_change": [
        r"\b(raise|raised|increase|increased|boost|boosted|lift|lifted)\s+(?:our\s+)?(guidance|outlook|forecast)\b",
        r"\b(lower|lowered|reduce|reduced|cut|cutting|decrease|decreased)\s+(?:our\s+)?(guidance|outlook|forecast)\b",
        r"\b(reaffirm|reiterat(?:e|ed)|maintain|maintained|keep|kept)\s+(?:our\s+)?(guidance|outlook|forecast)\b",
        r"\b(update|updated|narrow|narrowed|widen|widened|withdraw|withdrew|suspend|suspended)\s+(?:our\s+)?(guidance|outlook|forecast)\b",
    ],
    "safety_signal": [
        r"\b(serious\s+adverse\s+event|adverse\s+event|adverse\s+events)\b",
        r"\b(safety\s+signal|safety\s+concern|safety\s+issue)\b",
        r"\b(dose[- ]limiting\s+toxicity|dlt)\b",
        r"\b(tolerability|toxicity|toxicities)\b",
        r"\b(treatment[- ]related|drug[- ]related)\s+(death|fatalit(?:y|ies))\b",
        r"\bpatient\s+death\b",
    ],
    "regulatory_mention": [
        r"\b(fda|ema|mhra|pmda|health\s+canada)\b",
        r"\b(bla|nda|snda|maa|ind)\b",
        r"\b(pdufa|crl|complete\s+response\s+letter)\b",
        r"\b(adcom|advisory\s+committee)\b",
        r"\b(approval|approved|accelerated\s+approval|priority\s+review|fast\s+track|breakthrough\s+therapy)\b",
        r"\b(regulatory\s+filing|submission|filed\s+our|label(?:ing)?)\b",
        r"\b(clinical\s+hold)\b",
    ],
}


def _compile_patterns() -> Dict[str, List[re.Pattern]]:
    compiled: Dict[str, List[re.Pattern]] = {}
    for signal, patterns in SIGNAL_PATTERNS.items():
        compiled[signal] = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    return compiled


def _build_snippet(text: str, start: int, end: int, window: int = 80) -> str:
    left = max(start - window, 0)
    right = min(end + window, len(text))
    snippet = text[left:right].strip()
    return snippet.replace("\n", " ")


def find_signal_matches(text: str, compiled: Dict[str, List[re.Pattern]] | None = None) -> List[SignalMatch]:
    if not text:
        return []
    if compiled is None:
        compiled = _compile_patterns()

    matches: List[SignalMatch] = []
    for signal, patterns in compiled.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                phrase = match.group(0)
                snippet = _build_snippet(text, match.start(), match.end())
                matches.append(
                    SignalMatch(
                        signal=signal,
                        phrase=phrase,
                        start=match.start(),
                        end=match.end(),
                        snippet=snippet,
                    )
                )
    return matches


def extract_signal_features(text: str) -> dict:
    compiled = _compile_patterns()
    matches = find_signal_matches(text, compiled=compiled)
    features: Dict[str, object] = {}
    by_signal: Dict[str, List[SignalMatch]] = {signal: [] for signal in SIGNAL_PATTERNS}
    for match in matches:
        by_signal[match.signal].append(match)

    for signal, signal_matches in by_signal.items():
        phrases = [match.phrase for match in signal_matches]
        snippets = [match.snippet for match in signal_matches][:3]
        features[f"{signal}_count"] = len(signal_matches)
        features[f"{signal}_flag"] = bool(signal_matches)
        features[f"{signal}_phrases"] = json.dumps(list(dict.fromkeys(phrases)))
        features[f"{signal}_snippets"] = json.dumps(list(dict.fromkeys(snippets)))

    features["signal_total_count"] = sum(features[f"{signal}_count"] for signal in SIGNAL_PATTERNS)
    features["signal_types_present"] = json.dumps(
        [signal for signal in SIGNAL_PATTERNS if features.get(f"{signal}_flag")]  # type: ignore[arg-type]
    )
    return features


def add_signal_features(
    df,
    text_column: str = "qa_text",
) -> object:
    """Add signal extraction features to the DataFrame."""
    features = []
    for _, row in df.iterrows():
        text = row.get(text_column, "") if hasattr(row, "get") else ""
        features.append(extract_signal_features(text))
    features_df = df.copy()
    feature_frame = pd.DataFrame(features)
    return pd.concat([features_df.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
