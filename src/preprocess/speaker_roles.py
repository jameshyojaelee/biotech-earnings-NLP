"""Shared speaker-role classification utilities."""

from __future__ import annotations

from typing import Iterable


MANAGEMENT_TITLES = (
    "ceo",
    "cfo",
    "coo",
    "cio",
    "cto",
    "cmo",
    "cso",
    "president",
    "vice president",
    "vp",
    "chairman",
    "chief",
    "executive",
    "founder",
    "investor relations",
    "ir",
)


def classify_speaker_role(
    role: str,
    executive_names: Iterable[str] | None = None,
    analyst_names: Iterable[str] | None = None,
) -> str:
    """Classify a speaker role label as analyst/operator/management/other."""
    role_lower = role.lower().strip()
    if not role_lower:
        return "other"

    if "analyst" in role_lower:
        return "analyst"
    if "operator" in role_lower:
        return "operator"

    if analyst_names:
        for name in analyst_names:
            if name and name.lower() in role_lower:
                return "analyst"

    if executive_names:
        for name in executive_names:
            if name and name.lower() in role_lower:
                return "management"

    if any(title in role_lower for title in MANAGEMENT_TITLES):
        return "management"

    return "other"
