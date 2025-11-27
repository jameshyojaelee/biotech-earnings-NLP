"""Streamlit dashboard for biotech earnings NLP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


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

    config_path = Path("config/config.yaml")
    config = load_config(config_path)
    df = load_data(config)

    tickers = sorted(df["ticker"].unique())
    ticker = st.sidebar.selectbox("Select ticker", tickers)
    ticker_df = df[df["ticker"] == ticker].sort_values("earnings_date")

    if ticker_df.empty:
        st.warning("No data for selected ticker")
        return

    call_dates = ticker_df["earnings_date"].dt.date.astype(str).tolist()
    selected_date_str = st.sidebar.selectbox("Select call date", call_dates, index=len(call_dates) - 1)
    selected_date = pd.to_datetime(selected_date_str)
    row = ticker_df[ticker_df["earnings_date"].dt.date == selected_date.date()].iloc[-1]

    st.subheader(f"{ticker} earnings on {selected_date.date()}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Abnormal 1d", f"{row.get('abn_ret_1d', float('nan')):.2%}")
    col2.metric("Abnormal 5d", f"{row.get('abn_ret_5d', float('nan')):.2%}")
    col3.metric("Tone shift (Q&A - Prepared)", f"{row.get('tone_shift', float('nan')):.3f}")

    st.markdown("### Sentiment trend (Q&A)")
    st.line_chart(ticker_df.set_index("earnings_date")["qa_sent_score"], height=250)

    st.markdown("### Returns vs sentiment")
    st.bar_chart(ticker_df.set_index("earnings_date")[["abn_ret_1d", "abn_ret_5d"]])

    st.markdown("### Transcripts")
    with st.expander("Prepared remarks"):
        st.write(row.get("prepared_text", ""))
    with st.expander("Q&A"):
        st.write(row.get("qa_text", ""))

    st.markdown(
        "**Finance terms**: Earnings date = event anchor; abnormal return = stock minus XBI benchmark; event window = days after the event; tone shift = Q&A sentiment minus prepared sentiment."
    )


if __name__ == "__main__":
    main()
