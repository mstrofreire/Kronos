"""Forecast the next 24 hourly BTC/USDT candles with Kronos.

Usage (from the repo root):
    python examples/forecast_btc_24h.py                 # fetch recent candles from Binance
    python examples/forecast_btc_24h.py --csv btc.csv   # or use your own hourly CSV

The CSV needs columns: timestamps, open, high, low, close, volume (amount optional).
Kronos returns one averaged path per call, so this script samples several independent
paths with predict_batch and reports their median plus a 10%-90% range.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_binance(symbol, interval, limit):
    resp = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit + 1},
        timeout=30,
    )
    resp.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "amount", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(resp.json(), columns=cols)
    df = df.iloc[:-1]  # drop the candle that is still open
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = df[c].astype(float)
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="hourly OHLCV CSV instead of fetching from Binance")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--paths", type=int, default=20, help="number of sampled forecast paths")
    parser.add_argument("--model", default="NeoQuasar/Kronos-small")
    parser.add_argument("--out", default="btc_forecast")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.tail(args.lookback).reset_index(drop=True)
    else:
        df = fetch_binance(args.symbol, "1h", args.lookback)
    cols = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    x_ts = df["timestamps"]
    y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(hours=1), periods=args.hours, freq="h"))

    preds = predictor.predict_batch(
        df_list=[df[cols]] * args.paths,
        x_timestamp_list=[x_ts] * args.paths,
        y_timestamp_list=[y_ts] * args.paths,
        pred_len=args.hours,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )
    closes = np.stack([p["close"].values for p in preds])  # (paths, hours)

    summary = pd.DataFrame({
        "timestamp": y_ts.values,
        "close_p10": np.percentile(closes, 10, axis=0),
        "close_median": np.median(closes, axis=0),
        "close_p90": np.percentile(closes, 90, axis=0),
    })
    summary.to_csv(f"{args.out}.csv", index=False)

    last = df["close"].iloc[-1]
    end = summary.iloc[-1]
    print(f"\nLast close ({x_ts.iloc[-1]} UTC): {last:,.2f}")
    print(f"Forecast at {end['timestamp']} UTC: median {end['close_median']:,.2f} "
          f"({(end['close_median'] / last - 1) * 100:+.2f}%), "
          f"range {end['close_p10']:,.2f} - {end['close_p90']:,.2f}")
    up = (closes[:, -1] > last).mean() * 100
    print(f"Paths ending above last close: {up:.0f}%")

    hist = df.tail(96)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist["timestamps"], hist["close"], color="tab:blue", label="Historical close")
    ax.plot(summary["timestamp"], summary["close_median"], color="tab:red", label="Forecast (median)")
    ax.fill_between(summary["timestamp"], summary["close_p10"], summary["close_p90"],
                    color="tab:red", alpha=0.2, label="10%-90% range")
    ax.set_title(f"{args.symbol} - Kronos {args.hours}h forecast ({args.paths} paths)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=120)
    print(f"Saved {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()
