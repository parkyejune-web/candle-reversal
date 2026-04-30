"""
수수료 시나리오 비교: Gate.io(잘못된 코드) vs Gate.io(실제) vs MEXC
현재 라이브 파라미터 기준
"""
import numpy as np
import pandas as pd
from pathlib import Path

CACHE = Path("cache/BTCUSDT_1m_2024_2025.parquet")

MAX_BARS  = 1440

PARAM_SETS = {
    "TradingView (cover=0.5,rr=3.0,avg=20)": dict(big_mult=1.8, cover_pct=0.50, rr_ratio=3.0, avg_len=20, min_sl=0),
    "Live (cover=0.3,rr=3.5,avg=10)":        dict(big_mult=1.8, cover_pct=0.30, rr_ratio=3.5, avg_len=10, min_sl=0),
}

SCENARIOS = {
    "수수료 0%":             dict(maker=0.00000, taker=0.00000),
    "Gate.io (현재)":        dict(maker=0.00015, taker=0.00050),
    "MEXC (0%/0.01%)":       dict(maker=0.00000, taker=0.00010),
}


def detect_signals(df, big_mult, cover_pct, avg_len):
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)

    body     = np.abs(close - open_)
    avg_body = pd.Series(body).rolling(int(avg_len), min_periods=int(avg_len)).mean().values

    is_big  = body >= avg_body * big_mult
    is_bull = close > open_
    is_bear = close < open_

    prev_top = np.maximum(open_[:-1], close[:-1])
    prev_bot = np.minimum(open_[:-1], close[:-1])
    curr_top = np.maximum(open_[1:],  close[1:])
    curr_bot = np.minimum(open_[1:],  close[1:])
    prev_body = body[:-1]

    overlap = np.minimum(prev_top, curr_top) - np.maximum(prev_bot, curr_bot)
    cover   = np.where(prev_body > 0, overlap / prev_body, 0.0)

    short_sig = np.zeros(n, dtype=bool)
    long_sig  = np.zeros(n, dtype=bool)
    short_sig[1:] = is_big[:-1] & is_bull[:-1] & is_bear[1:] & (cover >= cover_pct)
    long_sig[1:]  = is_big[:-1] & is_bear[:-1] & is_bull[1:] & (cover >= cover_pct)
    return short_sig, long_sig


def run_backtest(df, short_sig, long_sig, rr_ratio, maker, taker,
                 min_sl_dist=0.0, max_bars=1440):
    high  = df["high"].values
    low   = df["low"].values
    open_ = df["open"].values
    close = df["close"].values
    n     = len(df)

    results_r = []
    exit_idx  = -1

    for i in range(n - 1):
        if i <= exit_idx:
            continue

        if short_sig[i]:
            side        = "short"
            sl_price    = high[i]
            entry_price = open_[i + 1]
            sl_dist     = sl_price - entry_price
        elif long_sig[i]:
            side        = "long"
            sl_price    = low[i]
            entry_price = open_[i + 1]
            sl_dist     = entry_price - sl_price
        else:
            continue

        if sl_dist <= 0 or sl_dist < min_sl_dist:
            continue

        tp_price = (entry_price - sl_dist * rr_ratio if side == "short"
                    else entry_price + sl_dist * rr_ratio)

        lev      = entry_price / sl_dist
        fee_win  = (maker + maker) * lev        # 진입maker + TP maker
        fee_loss = (maker - taker) * lev        # 진입maker - SL taker

        found = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if side == "short":
                hit_sl = high[j] >= sl_price
                hit_tp = low[j]  <= tp_price
            else:
                hit_sl = low[j]  <= sl_price
                hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                r = -1.0 + fee_loss
            elif hit_sl:
                r = -1.0 + fee_loss
            elif hit_tp:
                r = float(rr_ratio) + fee_win
            else:
                continue

            results_r.append(r)
            exit_idx = j
            found = True
            break

        if not found:
            j = min(i + max_bars, n - 1)
            r_raw = ((entry_price - close[j]) / sl_dist if side == "short"
                     else (close[j] - entry_price) / sl_dist)
            results_r.append(r_raw + fee_loss)
            exit_idx = j

    if not results_r:
        return None

    arr  = np.array(results_r)
    wins = (arr > 0).sum()
    cum  = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    mdd  = float((peak - cum).max())
    return {
        "trades":     len(arr),
        "win_rate":   wins / len(arr) * 100,
        "expectancy": float(arr.mean()),
        "total_r":    float(arr.sum()),
        "mdd_r":      mdd,
    }


if __name__ == "__main__":
    print("캐시 로드 중...")
    df_all = pd.read_parquet(CACHE)
    df_is  = df_all[df_all["ts"].dt.year == 2024].reset_index(drop=True)
    df_oos = df_all[df_all["ts"].dt.year == 2025].reset_index(drop=True)
    print(f"IS 2024: {len(df_is):,}봉  |  OOS 2025: {len(df_oos):,}봉\n")

    print(f"{'파라미터셋':<38} {'수수료':<18} {'기간':<8} {'거래':>6} {'승률':>7} {'기대값':>9} {'총R':>8}")
    print("-" * 100)

    for pname, p in PARAM_SETS.items():
        ss_is,  ls_is  = detect_signals(df_is,  p["big_mult"], p["cover_pct"], p["avg_len"])
        ss_oos, ls_oos = detect_signals(df_oos, p["big_mult"], p["cover_pct"], p["avg_len"])

        for fname, fees in SCENARIOS.items():
            for label, df, ss, ls in [
                ("IS 2024",  df_is,  ss_is,  ls_is),
                ("OOS 2025", df_oos, ss_oos, ls_oos),
            ]:
                r = run_backtest(df, ss, ls, p["rr_ratio"],
                                 fees["maker"], fees["taker"], p["min_sl"], MAX_BARS)
                if r:
                    print(f"{pname:<38} {fname:<18} {label:<8} {r['trades']:>6} "
                          f"{r['win_rate']:>6.1f}% {r['expectancy']:>9.4f}R "
                          f"{r['total_r']:>8.1f}R")
        print()
