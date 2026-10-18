"""Stage 2 of the Nasdaq scanner: rank the Stage 1 universe with Kronos.

For every stock (and QQQ as the market reference) Kronos samples many independent
forecast paths for the next --horizon trading days, with two context lengths.
Per stock and context we measure, from the final close of each path:
    exp_ret   median return
    p_up      share of paths ending above the last close
    spread    p90 - p10 return range
and build a market-relative, risk-adjusted score:
    score = (exp_ret - exp_ret_QQQ) / spread
A stock is "consistent" when both contexts agree on the sign of its excess return.
Sampling uses the cached fast path in scanner/fast_inference.py (eval mode).

Outputs (in --data-dir): stage2_ranking.csv (all stocks) and shortlist.csv (top N).
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
from scanner.fast_inference import forecast_paths

COLS = ["open", "high", "low", "close", "volume", "amount"]


def load_prices(path):
    df = pd.read_csv(path, parse_dates=["timestamps"])
    return df


def qqq_prices():
    raw = yf.download("QQQ", period="3y", interval="1d", auto_adjust=True, progress=False)
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df.index.name = "timestamps"
    return df.reset_index()


def forecast_returns(tokenizer, model, df, lookback, horizon, paths):
    """Array of `paths` sampled returns of the close `horizon` trading days ahead."""
    win = df.tail(lookback).reset_index(drop=True)
    future = pd.Series(pd.bdate_range(win["timestamps"].iloc[-1] + pd.offsets.BDay(1), periods=horizon))
    preds = forecast_paths(tokenizer, model, win, win["timestamps"], future, paths)
    return preds[:, -1, COLS.index("close")] / win["close"].iloc[-1] - 1


def summarize(rets):
    return {
        "exp_ret": float(np.median(rets)),
        "p_up": float((rets > 0).mean()),
        "spread": float(np.percentile(rets, 90) - np.percentile(rets, 10)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="scanner_data")
    p.add_argument("--model", default="NeoQuasar/Kronos-small")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--paths", type=int, default=32)
    p.add_argument("--lookbacks", default="507,256", help="lookback + horizon must stay <= 512")
    p.add_argument("--shortlist", type=int, default=40)
    p.add_argument("--limit", type=int, default=0, help="only process the first N stocks (for timing)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    lookbacks = [int(x) for x in args.lookbacks.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    model = Kronos.from_pretrained(args.model).eval()

    uni = pd.read_csv(os.path.join(args.data_dir, "universe.csv"))
    symbols = uni.loc[uni["passed"], "symbol"].tolist()
    if args.limit:
        symbols = symbols[:args.limit]
    frames = {s: load_prices(os.path.join(args.data_dir, "prices", f"{s}.csv")) for s in symbols}
    frames["QQQ"] = qqq_prices()

    results = {s: {} for s in frames}
    start = time.time()
    order = ["QQQ"] + symbols
    for n, sym in enumerate(order, 1):
        for lb in lookbacks:
            results[sym][lb] = summarize(forecast_returns(tokenizer, model, frames[sym], lb, args.horizon, args.paths))
        if n % 25 == 0 or n == len(order):
            print(f"{n}/{len(order)} stocks, {time.time() - start:.0f}s elapsed", flush=True)

    rows = []
    for s in symbols:
        row = {"symbol": s, "last_close": frames[s]["close"].iloc[-1]}
        scores, signs = [], []
        for lb in lookbacks:
            r, m = results[s][lb], results["QQQ"][lb]
            excess = r["exp_ret"] - m["exp_ret"]
            row.update({f"exp_ret_{lb}": r["exp_ret"], f"p_up_{lb}": r["p_up"],
                        f"spread_{lb}": r["spread"], f"excess_{lb}": excess})
            scores.append(excess / max(r["spread"], 1e-6))
            signs.append(np.sign(excess))
        row["score"] = float(np.mean(scores))
        row["consistent"] = len(set(signs)) == 1
        rows.append(row)
    rank = pd.DataFrame(rows).merge(uni[["symbol", "name", "market_cap", "next_earnings"]], on="symbol")
    rank = rank.sort_values("score", ascending=False).reset_index(drop=True)
    rank.to_csv(os.path.join(args.data_dir, "stage2_ranking.csv"), index=False)

    qqq = results["QQQ"][lookbacks[0]]
    print(f"\nQQQ ({args.horizon}d, ctx {lookbacks[0]}): median {qqq['exp_ret']:+.2%}, "
          f"P(up) {qqq['p_up']:.0%}, range {qqq['spread']:.2%}")
    short = rank[rank["consistent"] & (rank["score"] > 0)].head(args.shortlist)
    short.to_csv(os.path.join(args.data_dir, "shortlist.csv"), index=False)
    print(f"Shortlist: {len(short)} stocks (consistent, positive excess) out of {len(rank)}")
    print(short[["symbol", "name", "score", f"excess_{lookbacks[0]}", f"p_up_{lookbacks[0]}"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
