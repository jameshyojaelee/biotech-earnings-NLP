# Run tests: pytest -q
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocess.transcript_splitter import add_sections_to_events, split_prepared_and_qa  # noqa: E402


def test_split_with_marker():
    text = """
    Good afternoon and welcome.
    Question-and-Answer Session
    Analyst: Thanks for taking my question.
    """
    prepared, qa = split_prepared_and_qa(text)
    assert "good afternoon" in prepared.lower()
    assert "question-and-answer" in qa.lower()


def test_fallback_split_operator():
    text = "Opening remarks. Operator: We will now begin the Q&A."  # no explicit marker
    prepared, qa = split_prepared_and_qa(text)
    assert prepared.lower().startswith("opening remarks")
    assert qa.lower().startswith("operator") or "q&a" in qa.lower()


def test_add_sections_dataframe():
    df = pd.DataFrame({"ticker": ["ABC"], "transcript": ["Intro. Q&A Analyst: Hello"]})
    out = add_sections_to_events(df)
    assert "prepared_text" in out.columns
    assert "qa_text" in out.columns
