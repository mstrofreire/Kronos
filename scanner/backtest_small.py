"""Quick walk-forward backtest of the Kronos weekly signal on a handful of stocks.

Every 5 trading days, using only data up to that day, Kronos samples paths for the
next 5 sessions of each stock and of QQQ. We then compare against what happened:
    - direction hit rate of Kronos vs a naive "always up" call and a 20-day reversal rule
    - weekly rank correlation (IC) between predicted and realized excess return vs QQQ
    - calibration: how often the realized return falls inside Kronos' p10-p90 range
    - a long-only portfolio of the top-k stocks by Kronos score vs the reversal rule,
      equal weight across all stocks, and QQQ
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yfinance as yf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer
from scanner.fast_inference import forecast_paths, FEATURES

DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,AMZN,GOOGL,AVGO,COST,AMD,RKLB,RIVN"


def download(symbols, years):
    raw = yf.download(symbols, period=f"{years}y", interval="1d", auto_adjust=True,
                      group_by="ticker", progress=False)
    frames = {}
    for s in symbols:
        df = raw[s].dropna().rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
        frames[s] = df.rename_axis("timestamps").reset_index()
    return frames


def rank_corr(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--lookback", type=int, default=256)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--paths", type=int, default=32)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--out", default="backtest_small")
    args = p.parse_args()
    symbols = args.symbols.split(",")
    H = args.horizon

    torch.manual_seed(0)
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    model = Kronos.from_pretrained(args.model).eval()

    frames = download(symbols + ["QQQ"], args.years)
    dates = frames["QQQ"]["timestamps"]
    for s, df in frames.items():  # align every series on QQQ trading days
        frames[s] = df.set_index("timestamps").reindex(dates).ffill().reset_index()
    first = max(args.lookback, max(df["close"].first_valid_index() for df in frames.values()) + args.lookback)
    origins = list(range(first, len(dates) - H, H))
    print(f"{len(origins)} weekly origins from {dates[origins[0]].date()} to {dates[origins[-1]].date()}")

    rows, start = [], time.time()
    for n, t in enumerate(origins, 1):
        for s in symbols + ["QQQ"]:
            df = frames[s]
            win = df.iloc[t - args.lookback + 1:t + 1]
            future = dates.iloc[t + 1:t + 1 + H]
            preds = forecast_paths(tokenizer, model, win[FEATURES], win["timestamps"], future, args.paths)
            last = win["close"].iloc[-1]
            r = preds[:, -1, FEATURES.index("close")] / last - 1
            rows.append({
                "date": dates[t], "symbol": s,
                "exp_ret": np.median(r), "p_up": (r > 0).mean(),
                "p10": np.percentile(r, 10), "p90": np.percentile(r, 90),
                "ret20": last / df["close"].iloc[t - 20] - 1,
                "realized": df["close"].iloc[t + H] / last - 1,
            })
        if n % 10 == 0 or n == len(origins):
            print(f"{n}/{len(origins)} weeks, {time.time() - start:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out}.csv", index=False)
    mkt = res[res.symbol == "QQQ"].set_index("date")
    st = res[res.symbol != "QQQ"].copy()
    st["mkt_exp"] = st["date"].map(mkt["exp_ret"])
    st["mkt_real"] = st["date"].map(mkt["realized"])
    st["excess_pred"] = st["exp_ret"] - st["mkt_exp"]
    st["score"] = st["excess_pred"] / (st["p90"] - st["p10"]).clip(lower=1e-6)
    st["excess_real"] = st["realized"] - st["mkt_real"]

    print("\n== Direction (5 trading days) ==")
    print(f"Kronos sign(exp_ret) hit rate: {(np.sign(st.exp_ret) == np.sign(st.realized)).mean():.1%}")
    print(f"Naive 'always up' hit rate:    {(st.realized > 0).mean():.1%}")
    print(f"Reversal sign(-ret20) hit rate:{(np.sign(-st.ret20) == np.sign(st.realized)).mean():.1%}")
    print(f"QQQ Kronos hit rate:           {(np.sign(mkt.exp_ret) == np.sign(mkt.realized)).mean():.1%}  "
          f"(always up: {(mkt.realized > 0).mean():.1%})")

    ic_k = st.groupby("date").apply(lambda g: rank_corr(g.score, g.excess_real), include_groups=False)
    ic_r = st.groupby("date").apply(lambda g: rank_corr(-g.ret20, g.excess_real), include_groups=False)
    print("\n== Cross-sectional ranking (weekly rank correlation with realized excess) ==")
    print(f"Kronos score IC: mean {ic_k.mean():+.3f}, positive in {(ic_k > 0).mean():.0%} of weeks")
    print(f"Reversal IC:     mean {ic_r.mean():+.3f}, positive in {(ic_r > 0).mean():.0%} of weeks")
    print(f"Corr between the two signals: {rank_corr(st.score, -st.ret20):+.2f}")

    inside = ((st.realized >= st.p10) & (st.realized <= st.p90)).mean()
    print(f"\n== Calibration == realized inside Kronos p10-p90: {inside:.0%} (ideal 80%)")

    k = args.top_k
    port = pd.DataFrame({
        f"Kronos top {k}": st.sort_values("score").groupby("date").tail(k).groupby("date").realized.mean(),
        f"Reversal top {k}": st.sort_values("ret20").groupby("date").head(k).groupby("date").realized.mean(),
        f"Kronos bottom {k}": st.sort_values("score").groupby("date").head(k).groupby("date").realized.mean(),
        "All 10 equal weight": st.groupby("date").realized.mean(),
        "QQQ": mkt["realized"],
    })
    port.to_csv(f"{args.out}_portfolios.csv")
    print(f"\n== Weekly portfolios, {len(port)} weeks, no costs ==")
    summary = pd.DataFrame({
        "total return": (1 + port).prod() - 1,
        "avg week": port.mean(),
        "win weeks": (port > 0).mean(),
        "worst week": port.min(),
        "max drawdown": ((1 + port).cumprod() / (1 + port).cumprod().cummax() - 1).min(),
    })
    print(summary.to_string(float_format=lambda v: f"{v:+.1%}"))


if __name__ == "__main__":
    main()
