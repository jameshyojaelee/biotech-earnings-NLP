"""Evaluate signal extraction against a gold-labeled set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
SIGNALS = ["trial_update", "guidance_change", "safety_signal", "regulatory_mention"]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_event_id(ticker: str, earnings_date) -> str:
    if pd.isna(earnings_date):
        return f"{ticker}|unknown"
    date_str = pd.to_datetime(earnings_date).date().isoformat()
    return f"{ticker}|{date_str}"


def _coerce_label(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return 1
        if value in {"0", "false", "no", "n", ""}:
            return 0
    try:
        return 1 if int(value) == 1 else 0
    except (TypeError, ValueError):
        return 0


def _parse_json_list(value) -> list:
    if not value or pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate signal extraction features.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--gold-path", default="", help="Gold label CSV path")
    parser.add_argument("--predictions-path", default="", help="Predictions parquet path")
    parser.add_argument("--output-dir", default="", help="Directory for evaluation outputs")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    gold_path = Path(args.gold_path) if args.gold_path else Path(config.get("gold_labels_path", "data_processed/gold/gold_labels.csv"))
    predictions_path = Path(args.predictions_path) if args.predictions_path else Path(config.get("events_with_features_path", "data_processed/events_with_features.parquet"))
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.get("signal_eval_dir", "data_processed/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = pd.read_csv(gold_path)
    preds = pd.read_parquet(predictions_path)

    if "event_id" not in gold.columns:
        gold["event_id"] = gold.apply(lambda row: _build_event_id(row.get("ticker", ""), row.get("earnings_date")), axis=1)
    if "event_id" not in preds.columns:
        preds["event_id"] = preds.apply(lambda row: _build_event_id(row.get("ticker", ""), row.get("earnings_date")), axis=1)

    merged = gold.merge(preds, on="event_id", how="inner", suffixes=("_gold", ""))
    if merged.empty:
        raise ValueError("No overlapping event_id rows between gold labels and predictions.")

    metrics_rows = []
    failure_rows = []

    for signal in SIGNALS:
        label_col = signal if signal in merged.columns else f"{signal}_label"
        if label_col not in merged.columns:
            raise ValueError(f"Missing gold label column for {signal}. Expected '{signal}' or '{signal}_label'.")
        pred_col = f"{signal}_flag"
        if pred_col not in merged.columns:
            raise ValueError(f"Missing prediction column '{pred_col}'. Run compute_signal_features first.")

        y_true = merged[label_col].apply(_coerce_label)
        y_pred = merged[pred_col].fillna(False).astype(bool).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        metrics_rows.append(
            {
                "signal": signal,
                "support": int(y_true.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        fp_rows = merged[(y_true == 0) & (y_pred == 1)].copy()
        fn_rows = merged[(y_true == 1) & (y_pred == 0)].copy()
        for failure_type, subset in [("fp", fp_rows), ("fn", fn_rows)]:
            for _, row in subset.iterrows():
                snippets = _parse_json_list(row.get(f"{signal}_snippets"))
                failure_rows.append(
                    {
                        "signal": signal,
                        "failure_type": failure_type,
                        "ticker": row.get("ticker"),
                        "company": row.get("company"),
                        "earnings_date": row.get("earnings_date"),
                        "qa_text_preview": str(row.get("qa_text", ""))[:400],
                        "predicted_snippets": json.dumps(snippets),
                        "label": int(row.get(label_col, 0)),
                        "prediction": int(row.get(pred_col, 0)),
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "signal_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    failure_df = pd.DataFrame(failure_rows)
    failure_path = output_dir / "signal_failure_modes.csv"
    failure_df.to_csv(failure_path, index=False)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved failure modes to {failure_path}")


if __name__ == "__main__":
    main()
