"""
최근 N개 거래 스냅샷 차트 (30분봉)
Usage: python chart_trades.py
출력: trades_snapshot.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
SYMBOL    = "BTCUSDT"
PARAMS    = dict(big_mult=2.4, cover_pct=0.7, rr_ratio=4.3, avg_len=13)
N_TRADES  = 100
COLS      = 10
BEFORE    = 5
AFTER_PAD = 3
OUTPUT    = "trades_snapshot.png"


def load_30m():
    cache = CACHE_DIR / f"{SYMBOL}_1m_2020_2026.parquet"
    if not cache.exists():
        raise FileNotFoundError(f"캐시 없음: {cache}\nbacktest_risk.py 먼저 실행하세요.")
    print("캐시 로드 중...", flush=True)
    df_1m = pd.read_parquet(cache)
    df = df_1m.set_index("ts").resample("30min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    ).dropna(subset=["open"]).reset_index()
    print(f"30분봉: {len(df):,}개", flush=True)
    return df


def _rolling_mean(arr, window):
    result = np.full(len(arr), np.nan)
    if len(arr) < window:
        return result
    cs = np.cumsum(arr)
    result[window - 1] = cs[window - 1] / window
    if len(arr) > window:
        result[window:] = (cs[window:] - cs[:len(arr) - window]) / window
    return result


def detect_signals(df, big_mult, cover_pct, avg_len):
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)
    body     = np.abs(close - open_)
    avg_body = _rolling_mean(body, int(avg_len))
    is_big   = body >= avg_body * big_mult
    is_bull  = close > open_
    is_bear  = close < open_
    prev_top = np.maximum(open_[:-1], close[:-1])
    prev_bot = np.minimum(open_[:-1], close[:-1])
    curr_top = np.maximum(open_[1:],  close[1:])
    curr_bot = np.minimum(open_[1:],  close[1:])
    overlap  = np.minimum(prev_top, curr_top) - np.maximum(prev_bot, curr_bot)
    cover    = np.where(body[:-1] > 0, overlap / body[:-1], 0.0)
    short_sig = np.zeros(n, dtype=bool)
    long_sig  = np.zeros(n, dtype=bool)
    short_sig[1:] = is_big[:-1] & is_bull[:-1] & is_bear[1:] & (cover >= cover_pct)
    long_sig[1:]  = is_big[:-1] & is_bear[:-1] & is_bull[1:] & (cover >= cover_pct)
    return short_sig, long_sig


def get_detailed_trades(df, params, n=100):
    short_sig, long_sig = detect_signals(
        df, params["big_mult"], params["cover_pct"], params["avg_len"])
    high   = df["high"].values
    low    = df["low"].values
    open_  = df["open"].values
    close  = df["close"].values
    ts     = df["ts"].values
    rr     = params["rr_ratio"]

    trades   = []
    exit_idx = -1

    for i in range(len(df) - 1):
        if i <= exit_idx:
            continue
        if not (short_sig[i] or long_sig[i]):
            continue

        if short_sig[i]:
            sl_price    = high[i]
            entry_price = open_[i + 1]
            sl_dist     = sl_price - entry_price
            side        = "short"
        else:
            sl_price    = low[i]
            entry_price = open_[i + 1]
            sl_dist     = entry_price - sl_price
            side        = "long"

        if sl_dist <= 0:
            continue

        tp_price = (entry_price - sl_dist * rr if side == "short"
                    else entry_price + sl_dist * rr)

        reason = "timeout"
        r      = 0.0
        j_exit = min(i + 1440, len(df) - 1)

        for j in range(i + 1, min(i + 1441, len(df))):
            if side == "short":
                hit_sl = high[j] >= sl_price
                hit_tp = low[j]  <= tp_price
            else:
                hit_sl = low[j]  <= sl_price
                hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                reason = "sl"; r = -1.0; j_exit = j; break
            elif hit_tp:
                reason = "tp"; r = rr;   j_exit = j; break
            elif hit_sl:
                reason = "sl"; r = -1.0; j_exit = j; break

        if reason == "timeout":
            r = ((entry_price - close[j_exit]) / sl_dist if side == "short"
                 else (close[j_exit] - entry_price) / sl_dist)

        exit_idx = j_exit
        trades.append({
            "entry_i": i + 1,
            "exit_i":  j_exit,
            "side":    side,
            "entry":   entry_price,
            "sl":      sl_price,
            "tp":      tp_price,
            "r":       r,
            "reason":  reason,
            "ts":      pd.Timestamp(ts[i + 1]).strftime("%m/%d %H:%M"),
        })

    return trades[-n:]


def _draw_candles(ax, sub_df):
    for idx in range(len(sub_df)):
        row = sub_df.iloc[idx]
        up  = row["close"] >= row["open"]
        col = "#26a69a" if up else "#ef5350"
        ax.plot([idx, idx], [row["low"], row["high"]], color=col, linewidth=0.5, zorder=1)
        bot = min(row["open"], row["close"])
        ht  = max(abs(row["close"] - row["open"]), row["close"] * 0.00004)
        ax.add_patch(plt.Rectangle(
            (idx - 0.32, bot), 0.64, ht,
            facecolor=col, edgecolor=col, linewidth=0.15, zorder=2
        ))


def plot_one(ax, df, trade):
    entry_i = trade["entry_i"]
    exit_i  = trade["exit_i"]
    start   = max(0, entry_i - BEFORE)
    end     = min(len(df), exit_i + AFTER_PAD + 1)
    sub     = df.iloc[start:end].reset_index(drop=True)

    ax.set_facecolor("#0d1117")
    _draw_candles(ax, sub)

    rel_entry = entry_i - start
    rel_exit  = exit_i  - start

    ax.axhline(trade["entry"], color="#42a5f5", linewidth=0.7, linestyle="--", alpha=0.9, zorder=3)
    ax.axhline(trade["tp"],    color="#66bb6a", linewidth=0.7, linestyle="--", alpha=0.9, zorder=3)
    ax.axhline(trade["sl"],    color="#ef5350", linewidth=0.7, linestyle="--", alpha=0.9, zorder=3)

    ax.axvline(rel_entry, color="#42a5f5", linewidth=0.5, alpha=0.5, zorder=4)
    exit_c = {"tp": "#66bb6a", "sl": "#ef5350", "timeout": "#9e9e9e"}[trade["reason"]]
    ax.axvline(rel_exit, color=exit_c, linewidth=0.9, alpha=0.9, zorder=4)

    r = trade["r"]
    r_str   = f"+{r:.1f}R" if r >= 0 else f"{r:.1f}R"
    title_c = "#66bb6a" if r >= 0 else "#ef5350"
    sym     = "▲" if trade["side"] == "long" else "▼"
    ax.set_title(f"{sym} {trade['ts']}  {r_str}",
                 fontsize=4.8, color=title_c, pad=1.2, fontweight="bold")

    yvals = [trade["sl"], trade["tp"], sub["low"].min(), sub["high"].max()]
    ymin, ymax = min(yvals), max(yvals)
    pad = (ymax - ymin) * 0.06
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2a2a2a")
        sp.set_linewidth(0.3)


def main():
    df = load_30m()
    print("거래 추출 중...", flush=True)
    trades = get_detailed_trades(df, PARAMS, n=N_TRADES)
    n = len(trades)
    print(f"총 {n}개 거래 추출", flush=True)

    rows = (n + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3.2, rows * 2.4))
    fig.patch.set_facecolor("#0d1117")

    wins    = sum(1 for t in trades if t["r"] > 0)
    losses  = sum(1 for t in trades if t["r"] < 0)
    timeout = sum(1 for t in trades if t["reason"] == "timeout")
    wr      = wins / n * 100 if n else 0

    plt.suptitle(
        f"최근 {n}개 거래 스냅샷 — BTCUSDT 30m  "
        f"(big={PARAMS['big_mult']} cover={PARAMS['cover_pct']} rr={PARAMS['rr_ratio']})    "
        f"{wins}승 {losses}패  WR {wr:.1f}%  timeout {timeout}건  "
        f"▲=LONG  ▼=SHORT  │  파란선=진입  초록=TP  빨강=SL",
        fontsize=8, color="#cccccc", y=0.998, fontfamily="DejaVu Sans"
    )

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax_i, ax in enumerate(axes_flat):
        if ax_i < n:
            plot_one(ax, df, trades[ax_i])
        else:
            ax.set_visible(False)

    plt.tight_layout(pad=0.25, rect=[0, 0, 1, 0.994])
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"저장 완료: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
