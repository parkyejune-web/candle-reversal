"""
라이브 코드 기반 백테스트 (a3-xxx + a3-bb-reversal)
- 진입: 신호봉 종가 지정가 (최대 3봉 내 체결 확인)
- TP:   지정가 메이커 → 수수료 0%
- SL:   시장가 테이커 → 수수료 0.01%
- IS: 2020-2022  OOS: 2023-2025
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

CACHE    = Path("cache/BTCUSDT_1m_2020_2025.parquet")
TAKER    = 0.00010   # 0.01% (MEXC taker)
MIN_SL   = 0.001
MAX_SL   = 0.05
MAX_HOLD = 48        # 봉 수
FILL_BARS = 3        # 지정가 체결 대기 최대 봉 수


def resample(df_1m, minutes):
    df = df_1m.set_index("ts")
    rs = df.resample(f"{minutes}min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"),
        volume=("volume","sum"),
    ).dropna().reset_index()
    return rs


def summarize(arr_list, label):
    if not arr_list:
        return dict(label=label, trades=0, wr=0.0, exp=0.0, totalR=0.0, mdd=0.0)
    arr  = np.array(arr_list)
    wins = (arr > 0).sum()
    cum  = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    mdd  = float((peak - cum).max())
    return dict(
        label=label,
        trades=len(arr),
        wr=round(float(wins)/len(arr)*100, 1),
        exp=round(float(arr.mean()), 4),
        totalR=round(float(arr.sum()), 1),
        mdd=round(mdd, 1),
    )


def run_live_bt(df, sig_mask, sl_arr, max_hold=MAX_HOLD, fill_bars=FILL_BARS,
                fixed_sl_pct=None):
    """
    sig_mask   : bool[n]  신호 발생 여부
    sl_arr     : float[n] sl_pct (dynamic) — fixed_sl_pct 지정 시 무시
    fixed_sl_pct: float or None  (bb-reversal 용 고정 SL)
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(df)

    results  = []
    exit_idx = -1

    for i in range(n - 1):
        if i <= exit_idx:
            continue
        if not sig_mask[i]:
            continue

        limit_px = close[i]                     # 지정가 = 신호봉 종가
        sl_pct   = fixed_sl_pct if fixed_sl_pct else float(sl_arr[i])
        sl_price = limit_px * (1 - sl_pct)
        tp_price = limit_px * (1 + sl_pct)
        fee_loss = TAKER / sl_pct               # 손실 시만 테이커 0.01%

        # ── 체결 확인 (최대 fill_bars봉) ──────────────────────────────
        fill_j = None
        for fj in range(i + 1, min(i + fill_bars + 1, n)):
            if low[fj] <= limit_px:
                fill_j = fj
                break

        if fill_j is None:
            continue                             # 미체결 → 스킵

        entry = limit_px                         # 지정가 체결

        # ── SL/TP 탐색 ─────────────────────────────────────────────────
        found = False
        for j in range(fill_j, min(fill_j + max_hold + 1, n)):
            hit_sl = low[j]  <= sl_price
            hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                r = -1.0 - fee_loss   # 동시 hit → 보수적으로 SL
            elif hit_tp:
                r =  1.0              # TP 메이커 0%
            elif hit_sl:
                r = -1.0 - fee_loss
            else:
                continue

            results.append(r)
            exit_idx = j
            found = True
            break

        if not found:
            j     = min(fill_j + max_hold, n - 1)
            r_raw = (close[j] - entry) / (entry * sl_pct)
            results.append(r_raw - fee_loss)
            exit_idx = j

    return results


def build_indicators(df):
    df = df.copy()
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:, 0]
        pc = [c for c in bb.columns if c.startswith("BBP_")]
        if pc:
            df["pctb"] = bb[pc[0]]
    mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    if mfi is not None:
        df["mfi"] = mfi
    r14 = ta.rsi(df["close"], length=14)
    if r14 is not None:
        df["rsi"] = r14
    r2 = ta.rsi(df["close"], length=2)
    if r2 is not None:
        df["rsi_2"] = r2
    sma = ta.sma(df["close"], length=200)
    if sma is not None:
        df["sma_200"] = sma
    st = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    if st is not None and len(st.columns) >= 2:
        df["slowk"] = st.iloc[:, 0]
        df["slowd"] = st.iloc[:, 1]
    return df


def get_sig(df, strategy_id):
    n     = len(df)
    sig   = np.zeros(n, dtype=bool)
    close = df["close"].values
    low   = df["low"].values
    raw   = np.where(close > 0, (close - low) / close, MIN_SL)
    sl_a  = np.clip(raw, MIN_SL, MAX_SL)

    def col(name):
        return df[name].values if name in df.columns else np.full(n, np.nan)

    if strategy_id == "A3_0033":
        v = col("bb_lower")
        sig = (close < v) & ~np.isnan(v)
    elif strategy_id == "A3_0039":
        v = col("mfi")
        sig = (v < 20) & ~np.isnan(v)
    elif strategy_id == "A3_0096":
        v = col("pctb")
        sig[1:] = (v[1:] < 0) & (v[1:] > v[:-1]) & ~np.isnan(v[1:])
    elif strategy_id == "A3_0012":
        s, r = col("sma_200"), col("rsi_2")
        sig = (close > s) & (r <= 10) & ~np.isnan(s) & ~np.isnan(r)
    elif strategy_id == "A3_0070":
        sk, sd = col("slowk"), col("slowd")
        sig = (sk < 20) & (sk > sd) & ~np.isnan(sk) & ~np.isnan(sd)
    elif strategy_id == "A3_0063":
        r, s = col("rsi"), col("sma_200")
        sig = (r < 40) & (close > s) & ~np.isnan(r) & ~np.isnan(s)

    return sig, sl_a


def get_sig_bb_reversal(df):
    n     = len(df)
    sig   = np.zeros(n, dtype=bool)
    close = df["close"].values
    v     = df["pctb"].values if "pctb" in df.columns else np.full(n, np.nan)
    sig[1:] = (v[1:] < 0) & (close[1:] > close[:-1]) & ~np.isnan(v[1:])
    return sig


def run_strategy(df_full, name, tf_min, sig_fn, fixed_sl=None):
    print(f"\n[ {name} {tf_min}m ]")
    df_tf  = resample(df_full, tf_min)
    df_tf  = build_indicators(df_tf)
    df_is  = df_tf[df_tf["ts"].dt.year.isin([2020,2021,2022])].reset_index(drop=True)
    df_oos = df_tf[df_tf["ts"].dt.year.isin([2023,2024,2025])].reset_index(drop=True)

    rows = []
    for label, df in [("IS  2020-22", df_is), ("OOS 2023-25", df_oos)]:
        if fixed_sl is not None:
            sig   = sig_fn(df)
            sl_a  = np.zeros(len(df))
            rlist = run_live_bt(df, sig, sl_a, fixed_sl_pct=fixed_sl)
        else:
            sig, sl_a = sig_fn(df)
            rlist = run_live_bt(df, sig, sl_a)
        r = summarize(rlist, label)
        rows.append(r)
        print(f"  {label}: {r['trades']:>5}건  WR={r['wr']:>5.1f}%  "
              f"E={r['exp']:>+.4f}R  총={r['totalR']:>+8.1f}R  MDD={r['mdd']:.1f}R")
    return rows


# ── 메인 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("로드 중...")
    df_1m = pd.read_parquet(CACHE)
    print(f"  {len(df_1m):,}봉  (2020-2025)")

    all_rows = []

    # a3-bb-reversal (15m, 고정 SL 0.8%)
    rows = run_strategy(df_1m, "a3-bb-reversal", 15,
                        get_sig_bb_reversal, fixed_sl=0.008)
    for r in rows:
        all_rows.append({"전략": "a3-bb-reversal 15m", **r})

    # a3-xxx (dynamic SL)
    CONFIGS = [
        ("A3_0033", [5, 15]),
        ("A3_0012", [5, 15]),
        ("A3_0039", [5]),
        ("A3_0063", [5]),
        ("A3_0070", [5]),
        ("A3_0096", [5]),
    ]
    for sid, tfs in CONFIGS:
        for tf in tfs:
            fn = lambda df, s=sid: get_sig(df, s)
            rows = run_strategy(df_1m, sid, tf, fn)
            for r in rows:
                all_rows.append({"전략": f"{sid} {tf}m", **r})

    # ── 전체 요약 ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'전략':<22} {'기간':<12} {'거래':>6} {'승률':>7} {'기대값':>10} {'총R':>9} {'MDD':>8}")
    print("="*80)
    for r in all_rows:
        print(f"{r['전략']:<22} {r['label']:<12} {r['trades']:>6} "
              f"{r['wr']:>6.1f}% {r['exp']:>+10.4f}R {r['totalR']:>+9.1f}R {r['mdd']:>8.1f}R")

    pd.DataFrame(all_rows).to_csv("backtest_live_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: backtest_live_results.csv")
