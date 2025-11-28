"""Utilities for downloading prices and computing event-window returns."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# Stock returns measure percentage price change; a benchmark ETF (here XBI) is
# used to represent the sector so abnormal returns (stock minus benchmark)
# isolate firm-specific moves around the event window (the days immediately
# after the earnings date).

def download_price_history(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for tickers using yfinance."""
    tickers = list(tickers)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle the different shapes returned by yfinance:
    # - MultiIndex columns when requesting multiple tickers
    # - Single-index columns for one ticker
    if isinstance(data, pd.DataFrame):
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"].copy()
            else:
                prices = data["Close"].copy()
        else:
            if "Adj Close" in data.columns:
                prices = data["Adj Close"].to_frame(name=tickers[0])
            else:
                prices = data["Close"].to_frame(name=tickers[0])
    else:  # Single series
        prices = data.to_frame(name=tickers[0])

    prices.index = pd.to_datetime(prices.index)
    return prices


def _price_on_or_before(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> Optional[float]:
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna().sort_index()
    available = series.loc[series.index <= date]
    if available.empty:
        return None
    return float(available.iloc[-1])


def _price_on_or_after(
    prices: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    offset_to_next_business_day: bool = False,
) -> Optional[float]:
    """Return the earliest available price on or after `date`.

    If `offset_to_next_business_day` is True, the search starts from the next
    business day rather than the provided date.
    """
    if offset_to_next_business_day:
        date = pd.Timestamp(date) + pd.offsets.BDay()

    if ticker not in prices.columns:
        return None

    series = prices[ticker].dropna().sort_index()
    available = series.loc[series.index >= date]
    if available.empty:
        return None
    return float(available.iloc[0])


def compute_event_window_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    window_days: List[int],
) -> pd.DataFrame:
    """Compute stock and abnormal returns for specified forward windows."""
    events = events.copy()
    events["earnings_date"] = pd.to_datetime(events["earnings_date"], errors="coerce")

    for window in window_days:
        ret_col = f"ret_{window}d"
        bench_col = f"bench_ret_{window}d"
        abn_col = f"abn_ret_{window}d"
        stock_returns: List[Optional[float]] = []
        bench_returns: List[Optional[float]] = []
        abn_returns: List[Optional[float]] = []

        for row in events.itertuples():
            event_date = pd.to_datetime(row.earnings_date)
            end_date = event_date + pd.Timedelta(days=window)

            base_price = _price_on_or_before(prices, row.ticker, event_date)
            end_price = _price_on_or_after(prices, row.ticker, end_date)
            bench_base = _price_on_or_before(prices, benchmark_ticker, event_date)
            bench_end = _price_on_or_after(prices, benchmark_ticker, end_date)

            if base_price is None or end_price is None:
                stock_returns.append(np.nan)
            else:
                stock_returns.append((end_price - base_price) / base_price)

            if bench_base is None or bench_end is None:
                bench_returns.append(np.nan)
            else:
                bench_returns.append((bench_end - bench_base) / bench_base)

            if pd.isna(stock_returns[-1]) or pd.isna(bench_returns[-1]):
                abn_returns.append(np.nan)
            else:
                abn_returns.append(stock_returns[-1] - bench_returns[-1])

        events[ret_col] = stock_returns
        events[bench_col] = bench_returns
        events[abn_col] = abn_returns

    return events


if __name__ == "__main__":
    raise SystemExit("Use src/finance/compute_returns_for_events.py instead.")
