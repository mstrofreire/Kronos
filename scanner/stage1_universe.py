"""Stage 1 of the Nasdaq scanner: build a clean universe for Kronos.

Pulls Nasdaq-listed equities from the Yahoo screener, downloads ~3 years of daily
adjusted candles and keeps only stocks that are liquid, have enough clean history
for a 512-candle context, and show no behaviour Kronos cannot read (imminent
earnings, recent huge gaps, frozen or extreme volatility).

Outputs (in --out-dir):
    universe.csv          one row per stock with the filter verdict and reason
    prices/<SYMBOL>.csv   daily OHLCV + amount for every stock that passed
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as Q

NASDAQ_EXCHANGES = ["NMS", "NGM", "NCM"]
SPAC_WORDS = ("acquisition", "spac", "blank check")


def screen_nasdaq(min_price, min_mcap):
    query = Q("and", [
        Q("is-in", ["exchange"] + NASDAQ_EXCHANGES),
        Q("gt", ["intradayprice", min_price]),
        Q("gt", ["intradaymarketcap", min_mcap]),
    ])
    quotes, offset = [], 0
    while True:
        res = yf.screen(query, size=250, offset=offset, sortField="intradaymarketcap", sortAsc=False)
        quotes += res["quotes"]
        offset += 250
        if offset >= res.get("total", 0) or not res["quotes"]:
            break
    rows = []
    for q in quotes:
        rows.append({
            "symbol": q["symbol"],
            "name": q.get("longName") or q.get("shortName") or "",
            "quote_type": q.get("quoteType"),
            "price": q.get("regularMarketPrice"),
            "market_cap": q.get("marketCap"),
            "avg_volume_3m": q.get("averageDailyVolume3Month"),
            "first_trade": pd.to_datetime(q.get("firstTradeDateMilliseconds"), unit="ms"),
            "next_earnings": pd.to_datetime(q.get("earningsTimestampStart") or q.get("earningsTimestamp"), unit="s"),
        })
    return pd.DataFrame(rows).drop_duplicates("symbol")


def download_history(symbols, years):
    data = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        raw = yf.download(batch, period=f"{years}y", interval="1d", auto_adjust=True,
                          group_by="ticker", threads=True, progress=False)
        for s in batch:
            if s not in raw.columns.get_level_values(0):
                continue
            df = raw[s].dropna(how="all").rename(columns=str.lower)
            if df.empty:
                continue
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
            df.index.name = "timestamps"
            data[s] = df
        time.sleep(1)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="scanner_data")
    p.add_argument("--min-price", type=float, default=10)
    p.add_argument("--min-mcap", type=float, default=2e9)
    p.add_argument("--min-dollar-volume", type=float, default=20e6)
    p.add_argument("--min-history-days", type=int, default=630)  # ~2.5 years of sessions
    p.add_argument("--earnings-days", type=int, default=7)
    p.add_argument("--max-gap", type=float, default=0.25)
    p.add_argument("--vol-top-pct", type=float, default=5)
    args = p.parse_args()
    os.makedirs(os.path.join(args.out_dir, "prices"), exist_ok=True)

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    uni = screen_nasdaq(args.min_price, args.min_mcap)
    print(f"Yahoo screener: {len(uni)} Nasdaq stocks with price > {args.min_price} and market cap > {args.min_mcap:,.0f}")
    uni["reason"] = ""

    def reject(mask, reason):
        uni.loc[mask & (uni["reason"] == ""), "reason"] = reason

    reject(uni["quote_type"] != "EQUITY", "not common equity")
    reject(uni["name"].str.lower().str.contains("|".join(SPAC_WORDS)), "SPAC / blank check")
    reject(uni["price"] * uni["avg_volume_3m"] < args.min_dollar_volume, "dollar volume too low")
    reject(uni["first_trade"] > now - pd.Timedelta(days=int(args.min_history_days * 365 / 252)), "listed too recently")
    soon = (uni["next_earnings"] >= now - pd.Timedelta(days=1)) & (uni["next_earnings"] <= now + pd.Timedelta(days=args.earnings_days))
    reject(soon, f"earnings within {args.earnings_days} days")

    pending = uni.loc[uni["reason"] == "", "symbol"].tolist()
    print(f"Downloading daily history for {len(pending)} stocks...")
    hist = download_history(pending, years=3)

    vol60 = {}
    for s in pending:
        df = hist.get(s)
        if df is None or len(df) < args.min_history_days:
            reject(uni["symbol"] == s, "not enough clean history")
            continue
        if df[["open", "high", "low", "close"]].isna().any().any() or (df["close"] <= 0).any():
            reject(uni["symbol"] == s, "bad price data")
            continue
        ret = df["close"].pct_change().dropna()
        if ret.tail(63).abs().max() > args.max_gap:
            reject(uni["symbol"] == s, f"daily move > {args.max_gap:.0%} in last 3 months")
            continue
        vol60[s] = ret.tail(60).std()

    vols = pd.Series(vol60)
    if len(vols):
        frozen = vols[vols < 0.004].index  # < 0.4% daily: typically a pending acquisition
        reject(uni["symbol"].isin(frozen), "frozen price (likely pending deal)")
        cutoff = np.percentile(vols, 100 - args.vol_top_pct)
        reject(uni["symbol"].isin(vols[vols > cutoff].index), f"top {args.vol_top_pct:g}% volatility")
    uni["vol_60d"] = uni["symbol"].map(vols)

    uni["passed"] = uni["reason"] == ""
    for s in uni.loc[uni["passed"], "symbol"]:
        hist[s].to_csv(os.path.join(args.out_dir, "prices", f"{s}.csv"))
    uni.to_csv(os.path.join(args.out_dir, "universe.csv"), index=False)

    print(f"\nPassed: {uni['passed'].sum()} of {len(uni)}")
    print("Rejected by reason:")
    print(uni.loc[~uni["passed"], "reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
