"""Normalize transcripts into speaker-level segments with metadata."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional

import pandas as pd

from .speaker_roles import classify_speaker_role
from .transcript_splitter import find_qa_start, normalize_transcript, split_prepared_and_qa


SPEAKER_REGEX = re.compile(
    r"(?:(?<=^)|(?<=[\.\?\!\n]))\s*(?P<label>[A-Z][\w\.\'&/\- ]{1,70})\s*:\s*"
)
TIMESTAMP_REGEX = re.compile(r"^\s*\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*")

HEADER_EXEC_REGEX = re.compile(
    r"executives?\s*:\s*(?P<body>.+?)(?=(analysts?\s*:|operator\s*:|q\s*&\s*a|question[- ]and[- ]answer|$))",
    flags=re.IGNORECASE | re.DOTALL,
)
HEADER_ANALYST_REGEX = re.compile(
    r"analysts\s*:\s*(?P<body>.+?)(?=(operator\s*:|q\s*&\s*a|question[- ]and[- ]answer|$))",
    flags=re.IGNORECASE | re.DOTALL,
)

NAME_TITLE_REGEX = re.compile(
    r"(?P<name>[A-Z][A-Za-z\.\'\- ]+?)\s*-\s*(?P<title>[^:]+?)(?=(?:[A-Z][A-Za-z\.\'\- ]+?\s*-\s*|$))"
)

IGNORE_LABELS = {"executives", "executive", "analysts", "q&a", "question-and-answer"}
KNOWN_SHORT_LABELS = {"CEO", "CFO", "COO", "CIO", "CTO", "CMO", "CSO", "IR"}


@dataclass
class Segment:
    segment_index: int
    speaker_name: str
    speaker_role: str
    section: str
    text: str
    start_char: int
    end_char: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    start_time_seconds: Optional[float] = None
    end_time_seconds: Optional[float] = None
    segment_source: str = "heuristic"


def _parse_time_to_seconds(time_str: Optional[str]) -> Optional[float]:
    if not time_str:
        return None
    parts = time_str.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    return None


def _extract_timestamp(text: str) -> tuple[str | None, float | None, str]:
    match = TIMESTAMP_REGEX.match(text)
    if not match:
        return None, None, text
    ts = match.group("ts")
    remainder = text[match.end() :].strip()
    return ts, _parse_time_to_seconds(ts), remainder


def _extract_header_people(text: str, pattern: re.Pattern) -> tuple[str, List[str]]:
    match = pattern.search(text)
    if not match:
        return "", []
    body = match.group("body").strip()
    names = [m.group("name").strip() for m in NAME_TITLE_REGEX.finditer(body)]
    if not names:
        names = [item.strip() for item in re.split(r"[;\|]", body) if item.strip()]
    return body, names


def extract_transcript_metadata(text: str) -> dict:
    """Extract executive/analyst lists from a transcript header when present."""
    normalized = normalize_transcript(text)
    exec_raw, exec_names = _extract_header_people(normalized, HEADER_EXEC_REGEX)
    analyst_raw, analyst_names = _extract_header_people(normalized, HEADER_ANALYST_REGEX)
    return {
        "executive_list_raw": exec_raw,
        "analyst_list_raw": analyst_raw,
        "executive_names": exec_names,
        "analyst_names": analyst_names,
        "executive_count": len(exec_names),
        "analyst_count": len(analyst_names),
    }


def _strip_header_sections(text: str) -> str:
    cleaned = HEADER_EXEC_REGEX.sub(" ", text)
    cleaned = HEADER_ANALYST_REGEX.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _normalize_for_segmentation(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(
        r"(q\s*&\s*a)\s+(operator|analyst|analysts)\s*:",
        r"\1. \2:",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _is_valid_label(label: str) -> bool:
    cleaned = label.strip()
    if not cleaned:
        return False
    lower = cleaned.lower()
    if lower in IGNORE_LABELS:
        return False
    if lower in {"operator", "analyst", "analysts"}:
        return True
    if cleaned.upper() in KNOWN_SHORT_LABELS:
        return True
    if len(cleaned) > 70:
        return False
    if len(cleaned.split()) > 5:
        return False
    if cleaned.isupper() and len(cleaned) <= 4:
        return False
    return " " in cleaned or "." in cleaned or "-" in cleaned


def segment_transcript_text(
    text: str,
    executive_names: Iterable[str] | None = None,
    analyst_names: Iterable[str] | None = None,
) -> List[Segment]:
    """Split a transcript into speaker segments using heuristic labels."""
    normalized = _normalize_for_segmentation(text)
    normalized = _strip_header_sections(normalized)
    qa_start = find_qa_start(normalized, normalized=True)
    matches = [m for m in SPEAKER_REGEX.finditer(normalized) if _is_valid_label(m.group("label"))]

    segments: List[Segment] = []
    if not matches:
        prepared, qa = split_prepared_and_qa(normalized)
        if prepared:
            segments.append(
                Segment(
                    segment_index=0,
                    speaker_name="Unknown",
                    speaker_role="other",
                    section="prepared",
                    text=prepared,
                    start_char=0,
                    end_char=len(prepared),
                    segment_source="fallback",
                )
            )
        if qa:
            segments.append(
                Segment(
                    segment_index=len(segments),
                    speaker_name="Unknown",
                    speaker_role="other",
                    section="qa",
                    text=qa,
                    start_char=len(prepared),
                    end_char=len(prepared) + len(qa),
                    segment_source="fallback",
                )
            )
        return segments

    for idx, match in enumerate(matches):
        label = match.group("label").strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        segment_text = normalized[start:end].strip()
        if not segment_text:
            continue

        start_time, start_seconds, cleaned_text = _extract_timestamp(segment_text)
        speaker_role = classify_speaker_role(label, executive_names, analyst_names)
        section = "qa" if qa_start != -1 and start >= qa_start else "prepared"
        segments.append(
            Segment(
                segment_index=len(segments),
                speaker_name=label,
                speaker_role=speaker_role,
                section=section,
                text=cleaned_text,
                start_char=start,
                end_char=end,
                start_time=start_time,
                start_time_seconds=start_seconds,
                segment_source="heuristic",
            )
        )
    return segments


def segments_from_structured(
    segments: Iterable[object],
    executive_names: Iterable[str] | None = None,
    analyst_names: Iterable[str] | None = None,
) -> List[Segment]:
    """Normalize structured segments into the common Segment schema."""
    df = pd.DataFrame(segments)
    if df.empty:
        return []

    text_col = None
    for candidate in ["text", "content", "segment_text", "body"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col and "text" not in df.columns:
        df["text"] = df[text_col]
    if "speaker_role" not in df.columns:
        for candidate in ["speaker", "speaker_name", "role", "speaker_title"]:
            if candidate in df.columns:
                df["speaker_role"] = df[candidate]
                break
    if "speaker_name" not in df.columns:
        for candidate in ["speaker_name", "speaker", "speaker_role"]:
            if candidate in df.columns:
                df["speaker_name"] = df[candidate]
                break

    qa_started = False
    output: List[Segment] = []
    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        role = str(row.get("speaker_role", ""))
        speaker_role = classify_speaker_role(role, executive_names, analyst_names)
        if speaker_role == "analyst":
            qa_started = True
        if speaker_role == "operator" and output:
            qa_started = True
        section = "qa" if qa_started else "prepared"

        start_time = None
        end_time = None
        for candidate in ["timestamp", "start_time", "start", "begin"]:
            value = row.get(candidate)
            if value:
                start_time = str(value)
                break
        for candidate in ["end_time", "end", "finish"]:
            value = row.get(candidate)
            if value:
                end_time = str(value)
                break

        output.append(
            Segment(
                segment_index=len(output),
                speaker_name=str(row.get("speaker_name", role)),
                speaker_role=speaker_role,
                section=section,
                text=text,
                start_char=0,
                end_char=0,
                start_time=start_time,
                end_time=end_time,
                start_time_seconds=_parse_time_to_seconds(start_time),
                end_time_seconds=_parse_time_to_seconds(end_time),
                segment_source="structured",
            )
        )
    return output


def segments_to_frame(segments: Iterable[Segment]) -> pd.DataFrame:
    return pd.DataFrame([segment.__dict__ for segment in segments])
