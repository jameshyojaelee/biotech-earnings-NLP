"""Streamlit dashboard for biotech earnings NLP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st
import yaml

SIGNAL_LABELS = {
    "trial_update": "Trial update",
    "guidance_change": "Guidance change",
    "safety_signal": "Safety signal",
    "regulatory_mention": "Regulatory mention",
}


@st.cache_data
def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_data(config: dict) -> pd.DataFrame:
    path = Path(config.get("events_with_features_path", "data_processed/events_with_features.parquet"))
    df = pd.read_parquet(path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    return df


def _parse_json_list(value) -> list:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        import json

        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def _format_metric(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def _render_signal_card(row) -> None:
    signals = []
    for key, label in SIGNAL_LABELS.items():
        flag_col = f"{key}_flag"
        count_col = f"{key}_count"
        if flag_col in row and bool(row.get(flag_col)):
            count = int(row.get(count_col, 1))
            signals.append(f"{label} ({count})")

    snippets = []
    for key in SIGNAL_LABELS:
        snippet_col = f"{key}_snippets"
        snippets.extend(_parse_json_list(row.get(snippet_col)))

    snippet_text = snippets[0] if snippets else ""
    date_val = row.get("earnings_date")
    date_str = pd.to_datetime(date_val).date().isoformat() if pd.notna(date_val) else "unknown"
    tone_shift = row.get("tone_shift")
    tone_shift_display = f"{tone_shift:.3f}" if pd.notna(tone_shift) else "n/a"

    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-card-header">{row.get('ticker', '')} — {date_str}</div>
            <div class="signal-card-sub">{row.get('company', '')}</div>
            <div class="signal-card-metrics">
                <span>Abn 1d: {_format_metric(row.get('abn_ret_1d'))}</span>
                <span>Abn 5d: {_format_metric(row.get('abn_ret_5d'))}</span>
                <span>Tone shift: {tone_shift_display}</span>
            </div>
            <div class="signal-card-signals">{", ".join(signals) if signals else "No detected signals"}</div>
            <div class="signal-card-snippet">{snippet_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Biotech Earnings NLP", layout="wide")
    st.title("Biotech Earnings NLP Dashboard")
    st.write(
        "Track how prepared remarks and Q&A tone relate to post-earnings returns for Health Care tickers."
    )

    st.info(
        "Stock returns = % price change; abnormal returns = stock minus benchmark (XBI) to isolate firm-specific moves.\n"
        "Event window = days after the earnings date (e.g., +1d, +5d)."
    )

    st.markdown(
        """
        <style>
        .signal-card {
            background: #0f172a;
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.25);
        }
        .signal-card-header {
            font-weight: 700;
            font-size: 1.05rem;
            color: #f8fafc;
            margin-bottom: 4px;
        }
        .signal-card-sub {
            color: #cbd5f5;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }
        .signal-card-metrics span {
            display: inline-block;
            margin-right: 12px;
            font-size: 0.85rem;
            color: #e2e8f0;
        }
        .signal-card-signals {
            margin-top: 8px;
            font-weight: 600;
            color: #38bdf8;
        }
        .signal-card-snippet {
            margin-top: 8px;
            color: #e2e8f0;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    config_path = Path("config/config.yaml")
    config = load_config(config_path)
    df = load_data(config)

    if df.empty:
        st.warning("No data available.")
        return

    st.sidebar.header("Filters")
    tickers = sorted(df["ticker"].dropna().unique())
    selected_tickers = st.sidebar.multiselect("Tickers", tickers, default=tickers)

    date_min = df["earnings_date"].min().date()
    date_max = df["earnings_date"].max().date()
    date_range = st.sidebar.date_input("Earnings date range", (date_min, date_max))
    search_query = st.sidebar.text_input("Search transcripts", "")

    available_signals = [key for key in SIGNAL_LABELS if f"{key}_flag" in df.columns]
    selected_signals = st.sidebar.multiselect(
        "Signal filters", [SIGNAL_LABELS[key] for key in available_signals]
    )
    signal_match_mode = st.sidebar.radio("Signal match", ["Any", "All"], horizontal=True)

    max_results = st.sidebar.slider("Max results", 5, 200, 40)

    filtered = df.copy()
    if selected_tickers:
        filtered = filtered[filtered["ticker"].isin(selected_tickers)]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["earnings_date"].dt.date >= start_date)
            & (filtered["earnings_date"].dt.date <= end_date)
        ]

    if search_query:
        query = search_query.lower()
        filtered = filtered[
            filtered["qa_text"].fillna("").str.lower().str.contains(query)
            | filtered["prepared_text"].fillna("").str.lower().str.contains(query)
        ]

    if selected_signals and available_signals:
        selected_keys = [
            key for key, label in SIGNAL_LABELS.items() if label in selected_signals
        ]
        if selected_keys:
            if signal_match_mode == "All":
                for key in selected_keys:
                    filtered = filtered[filtered[f"{key}_flag"].fillna(False)]
            else:
                mask = False
                for key in selected_keys:
                    mask = mask | filtered[f"{key}_flag"].fillna(False)
                filtered = filtered[mask]

    filtered = filtered.sort_values("earnings_date", ascending=False)

    st.subheader("Signal cards")
    st.caption(f"{len(filtered)} calls match the current filters.")

    if filtered.empty:
        st.info("No calls match the current filters.")
    else:
        card_rows = filtered.head(max_results)
        for i in range(0, len(card_rows), 2):
            cols = st.columns(2)
            for col, (_, row) in zip(cols, card_rows.iloc[i : i + 2].iterrows()):
                with col:
                    _render_signal_card(row)

    st.subheader("Event detail")
    if filtered.empty:
        st.info("Select filters to view event details.")
        return

    filtered = filtered.copy()
    filtered["event_label"] = (
        filtered["ticker"].astype(str)
        + " | "
        + filtered["earnings_date"].dt.date.astype(str)
        + " | "
        + filtered["company"].fillna("")
    )
    selected_label = st.selectbox("Choose an event", filtered["event_label"].tolist())
    row = filtered[filtered["event_label"] == selected_label].iloc[0]
    ticker = row.get("ticker")

    return_options = [col for col in ["abn_ret_1d", "abn_ret_5d"] if col in df.columns]
    return_col = return_options[0] if return_options else None
    other_col = "abn_ret_5d" if return_col == "abn_ret_1d" else "abn_ret_1d"

    st.subheader(f"{ticker} earnings on {pd.to_datetime(row.get('earnings_date')).date()}")
    col1, col2, col3 = st.columns(3)
    col1.metric(return_col.replace("_", " ") if return_col else "Abn ret", _format_metric(row.get(return_col)))
    col2.metric(other_col.replace("_", " "), _format_metric(row.get(other_col)))
    col3.metric("Tone shift (Q&A - Prepared)", f"{row.get('tone_shift', float('nan')):.3f}")

    ticker_df = df[df["ticker"] == ticker].sort_values("earnings_date")
    if not ticker_df.empty:
        st.markdown("### Sentiment trend (Q&A)")
        st.line_chart(ticker_df.set_index("earnings_date")["qa_sent_score"], height=250)

        st.markdown("### Returns vs sentiment")
        st.bar_chart(ticker_df.set_index("earnings_date")[["abn_ret_1d", "abn_ret_5d"]])

        if return_col:
            st.markdown("### QA sentiment vs abnormal return")
            scatter_data = ticker_df[["qa_sent_score", return_col, "earnings_date"]].dropna()
            if not scatter_data.empty:
                chart = (
                    alt.Chart(scatter_data)
                    .mark_circle(size=70, opacity=0.8, color="#64ffda")
                    .encode(
                        x="qa_sent_score",
                        y=return_col,
                        tooltip=["earnings_date", "qa_sent_score", return_col],
                    )
                )
                trend = chart.transform_regression("qa_sent_score", return_col).mark_line(color="#ff7edb")
                st.altair_chart(chart + trend, use_container_width=True)
            else:
                st.info("Not enough data to plot scatter.")

    st.markdown("### Transcripts")
    with st.expander("Prepared remarks"):
        st.write(row.get("prepared_text", ""))
    with st.expander("Q&A"):
        st.write(row.get("qa_text", ""))

    csv_bytes = filtered.drop(columns=["event_label"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered events", data=csv_bytes, file_name="filtered_events.csv", mime="text/csv"
    )

    st.markdown(
        "**Finance terms**: Earnings date = event anchor; abnormal return = stock minus XBI benchmark; event window = days after the event; tone shift = Q&A sentiment minus prepared sentiment."
    )


if __name__ == "__main__":
    main()
