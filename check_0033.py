"""
A3_0033 5m 진입 방식 비교
  A: next-bar open  (내 기존 백테스트)
  B: 신호봉 종가 지정가 + SL만 테이커  (실제 라이브 방식)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

CACHE = Path("cache/BTCUSDT_1m_2020_2025.parquet")
TAKER = 0.00010

def resample(df_1m, minutes):
    df = df_1m.set_index("ts")
    rs = df.resample(f"{minutes}min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()
    return rs

def summarize(arr_list):
    if not arr_list:
        return None
    arr  = np.array(arr_list)
    wins = (arr > 0).sum()
    cum  = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    mdd  = float((peak - cum).max())
    return dict(
        trades=len(arr),
        wr=round(wins/len(arr)*100,1),
        exp=round(float(arr.mean()),4),
        totalR=round(float(arr.sum()),1),
        mdd=round(mdd,1),
    )

def run(df, max_hold=48):
    bb = ta.bbands(df["close"], length=20, std=2)
    df = df.copy()
    df["bb_lower"] = bb.iloc[:, 0]

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    open_ = df["open"].values
    bb_l  = df["bb_lower"].values
    n     = len(df)

    raw_sl = np.where(close > 0, (close - low) / close, 0.001)
    sl_arr = np.clip(raw_sl, 0.001, 0.05)
    sig    = (close < bb_l) & ~np.isnan(bb_l)

    res_A = []   # next-bar open, 양방향 taker
    res_B = []   # 신호봉 close 지정가, SL만 taker

    exit_A = exit_B = -1

    for i in range(n - 1):
        sl_pct   = float(sl_arr[i])
        fee_both = 2 * TAKER / sl_pct
        fee_loss = TAKER / sl_pct

        # ── 방식 A ────────────────────────────────────────────────────
        if i > exit_A and sig[i]:
            entry_A  = open_[i + 1]
            sl_A     = entry_A * (1 - sl_pct)
            tp_A     = entry_A * (1 + sl_pct)
            found = False
            for j in range(i + 1, min(i + max_hold + 1, n)):
                h_sl = low[j]  <= sl_A
                h_tp = high[j] >= tp_A
                if h_sl and h_tp:   r = -1.0 - fee_both
                elif h_tp:          r =  1.0 - fee_both
                elif h_sl:          r = -1.0 - fee_both
                else: continue
                res_A.append(r); exit_A = j; found = True; break
            if not found:
                j = min(i + max_hold, n - 1)
                res_A.append((close[j]-entry_A)/(entry_A*sl_pct) - fee_both)
                exit_A = j

        # ── 방식 B ────────────────────────────────────────────────────
        if i > exit_B and sig[i]:
            entry_B  = close[i]
            sl_B     = low[i]
            tp_B     = 2 * close[i] - low[i]
            found = False
            for j in range(i + 1, min(i + max_hold + 1, n)):
                h_sl = low[j]  <= sl_B
                h_tp = high[j] >= tp_B
                if h_sl and h_tp:   r = -1.0 - fee_loss
                elif h_tp:          r =  1.0
                elif h_sl:          r = -1.0 - fee_loss
                else: continue
                res_B.append(r); exit_B = j; found = True; break
            if not found:
                j = min(i + max_hold, n - 1)
                res_B.append((close[j]-entry_B)/(entry_B*sl_pct) - fee_loss)
                exit_B = j

    return summarize(res_A), summarize(res_B)


if __name__ == "__main__":
    print("로드 중...")
    df_1m = pd.read_parquet(CACHE)
    df_5m = resample(df_1m, 5)

    df_is  = df_5m[df_5m["ts"].dt.year.isin([2020,2021,2022])].reset_index(drop=True)
    df_oos = df_5m[df_5m["ts"].dt.year.isin([2023,2024,2025])].reset_index(drop=True)

    print("\nA3_0033 5m — 진입 방식 비교 (수수료 0.01%)")
    print(f"{'기간':<12} {'방식':<30} {'거래':>6} {'승률':>7} {'기대값':>10} {'총R':>9} {'MDD':>8}")
    print("-"*80)

    for label, df in [("IS 2020-22", df_is), ("OOS 2023-25", df_oos)]:
        a, b = run(df)
        if a:
            print(f"{label:<12} {'A: next-bar open (양방향 taker)':<30} {a['trades']:>6} "
                  f"{a['wr']:>6.1f}% {a['exp']:>+10.4f}R {a['totalR']:>+9.1f}R {a['mdd']:>8.1f}R")
        if b:
            print(f"{label:<12} {'B: 신호봉close + SL만 taker':<30} {b['trades']:>6} "
                  f"{b['wr']:>6.1f}% {b['exp']:>+10.4f}R {b['totalR']:>+9.1f}R {b['mdd']:>8.1f}R")
        print()
