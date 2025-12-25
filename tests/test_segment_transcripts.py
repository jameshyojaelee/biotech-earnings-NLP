# Run tests: pytest -q
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocess.segment_transcripts import extract_transcript_metadata, segment_transcript_text  # noqa: E402


def test_segment_transcript_text_with_metadata_and_timestamps():
    transcript = (
        "Executives: Jane Doe - CEO Analysts: John Smith - Big Bank "
        "Operator: Welcome to the call. "
        "Jane Doe: Prepared remarks go here. "
        "Q&A Operator: We will begin the Q&A. "
        "Analyst: [00:01:23] What about guidance? "
        "Jane Doe: We raised guidance."
    )
    metadata = extract_transcript_metadata(transcript)
    segments = segment_transcript_text(
        transcript,
        executive_names=metadata.get("executive_names"),
        analyst_names=metadata.get("analyst_names"),
    )

    assert segments, "Expected at least one segment"
    assert any(seg.speaker_role == "operator" for seg in segments)
    assert any(seg.section == "qa" for seg in segments)
    assert any(seg.start_time == "00:01:23" for seg in segments)
    assert any(seg.speaker_role == "management" for seg in segments)
