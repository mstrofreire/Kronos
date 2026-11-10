"""
Scan a list of tickers with Kronos and estimate short-term rebound odds.

Data sources (first one that works):
  1. CSV files in --csv-dir named <TICKER>.csv with columns
     timestamps/date, open, high, low, close, volume
  2. Twelve Data API (daily bars). The key is read from TWELVEDATA_API_KEY,
     or injected by a proxy via the Authorization header.

Example:
  python examples/rebound_scan.py --tickers NVDA AVGO AMD LRCX MRVL --horizon 10 --paths 30
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = "timestamps" if "timestamps" in df.columns else "date"
    df["timestamps"] = pd.to_datetime(df[ts_col])
    return df.sort_values("timestamps").reset_index(drop=True)


def load_twelvedata(ticker, n=600):
    params = {"symbol": ticker, "interval": "1day", "outputsize": n, "format": "JSON"}
    headers = {}
    key = os.environ.get("TWELVEDATA_API_KEY")
    if key:
        params["apikey"] = key
    else:
        headers["Authorization"] = "apikey placeholder"  # replaced by proxy if configured
    r = requests.get("https://api.twelvedata.com/time_series", params=params, headers=headers, timeout=30)
    data = r.json()
    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data: {data.get('message')}")
    df = pd.DataFrame(data["values"])
    df["timestamps"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df.sort_values("timestamps").reset_index(drop=True)


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["NVDA", "AVGO", "AMD", "LRCX", "MRVL"])
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--paths", type=int, default=30)
    ap.add_argument("--model", default="NeoQuasar/Kronos-small")
    args = ap.parse_args()

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    rows = []
    for t in args.tickers:
        csv = os.path.join(args.csv_dir, f"{t}.csv") if args.csv_dir else None
        df = load_csv(csv) if csv and os.path.exists(csv) else load_twelvedata(t)
        df = df.tail(args.lookback).reset_index(drop=True)
        last = df["close"].iloc[-1]
        x_ts = df["timestamps"]
        y_ts = pd.Series(pd.bdate_range(x_ts.iloc[-1] + pd.Timedelta(days=1), periods=args.horizon))

        finals, maxes = [], []
        for _ in range(args.paths):
            p = predictor.predict(df=df[["open", "high", "low", "close", "volume"]], x_timestamp=x_ts,
                                  y_timestamp=y_ts, pred_len=args.horizon, T=1.0, top_p=0.9,
                                  sample_count=1, verbose=False)
            finals.append(p["close"].iloc[-1] / last - 1)
            maxes.append(p["high"].max() / last - 1)
        finals, maxes = np.array(finals), np.array(maxes)
        rows.append({
            "ticker": t,
            "last_date": x_ts.iloc[-1].date(),
            "last_close": round(float(last), 2),
            "rsi14": round(float(rsi(df["close"]).iloc[-1]), 1),
            "drawdown_from_high_%": round(float(last / df["high"].max() - 1) * 100, 1),
            "median_ret_%": round(float(np.median(finals)) * 100, 2),
            "p10_%": round(float(np.percentile(finals, 10)) * 100, 2),
            "p90_%": round(float(np.percentile(finals, 90)) * 100, 2),
            "prob_up": round(float((finals > 0).mean()), 2),
            "median_max_high_%": round(float(np.median(maxes)) * 100, 2),
        })
        print(", ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)

    out = pd.DataFrame(rows).sort_values("prob_up", ascending=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
