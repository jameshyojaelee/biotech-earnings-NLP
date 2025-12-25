# Run tests: pytest -q
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from features.signal_extraction import extract_signal_features  # noqa: E402


def test_signal_extraction_detects_key_signals():
    text = (
        "We initiated a Phase 2 trial and reported top-line data. "
        "We raised guidance for the full year. "
        "There were no serious adverse events reported. "
        "The FDA granted priority review."
    )
    features = extract_signal_features(text)

    assert features["trial_update_flag"] is True
    assert features["guidance_change_flag"] is True
    assert features["safety_signal_flag"] is True
    assert features["regulatory_mention_flag"] is True

    snippets = json.loads(features["trial_update_snippets"])
    assert any("Phase 2" in snippet or "phase 2" in snippet for snippet in snippets)
