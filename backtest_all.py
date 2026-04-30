"""
전체 라이브 전략 백테스트 — 수수료 0.01% taker 통일
IS: 2020-2022  OOS: 2023-2025  심볼: BTCUSDT

전략 목록:
  candle-reversal  (1m/3m/5m) — 기존 결과 CSV 로드
  a3-bb-reversal   (15m)      — %B<0 AND close>prev_close, SL/TP 0.8%
  A3_0012 (5m/15m)            — RSI-2<=10 AND close>SMA200
  A3_0033 (5m/15m)            — close < BB_lower
  A3_0039 (5m)                — MFI < 20
  A3_0063 (5m)                — RSI14<40 AND close>SMA200
  A3_0070 (5m)                — slowk<20 AND slowk>slowd
  A3_0096 (5m)                — pctB<0 AND pctB>prev_pctB
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

CACHE = Path("cache/BTCUSDT_1m_2020_2025.parquet")
TAKER = 0.00010   # 0.01%


# ── 유틸 ─────────────────────────────────────────────────────────────
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
        return {k: None for k in ("label","trades","wr","exp","totalR","mdd")}
    arr  = np.array(arr_list)
    wins = (arr > 0).sum()
    cum  = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    mdd  = float((peak - cum).max())
    return {
        "label":   label,
        "trades":  len(arr),
        "wr":      round(wins / len(arr) * 100, 1),
        "exp":     round(float(arr.mean()), 4),
        "totalR":  round(float(arr.sum()), 1),
        "mdd":     round(mdd, 1),
    }


def run_bt(df, signal_mask, sl_arr, max_hold):
    """
    signal_mask : bool array, len=n  (True = signal on bar i → enter next bar)
    sl_arr      : float array, len=n (sl_pct for each bar)
    max_hold    : bars
    """
    high  = df["high"].values
    low   = df["low"].values
    open_ = df["open"].values
    close = df["close"].values
    n     = len(df)

    results = []
    exit_idx = -1

    for i in range(n - 1):
        if i <= exit_idx:
            continue
        if not signal_mask[i]:
            continue

        sl_pct   = float(sl_arr[i])
        if sl_pct <= 0:
            continue

        entry    = open_[i + 1]
        sl_price = entry * (1 - sl_pct)
        tp_price = entry * (1 + sl_pct)
        fee_r    = 2.0 * TAKER / sl_pct

        found = False
        for j in range(i + 1, min(i + max_hold + 1, n)):
            hit_sl = low[j]  <= sl_price
            hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                r = -1.0 - fee_r
            elif hit_tp:
                r =  1.0 - fee_r
            elif hit_sl:
                r = -1.0 - fee_r
            else:
                continue

            results.append(r)
            exit_idx = j
            found = True
            break

        if not found:
            j     = min(i + max_hold, n - 1)
            r_raw = (close[j] - entry) / (entry * sl_pct)
            results.append(r_raw - fee_r)
            exit_idx = j

    return results


def run_bt_maker_entry(df, signal_mask, sl_arr, max_hold):
    """
    진입: 신호봉 종가 지정가 (메이커 0%)
    TP:  지정가 메이커 (0%)  → 이기면 수수료 0
    SL:  시장가 테이커 (0.01%) → 질 때만 수수료
    SL 가격 = 신호봉 저가 (exactly)
    TP 가격 = 2×종가 - 저가
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(df)

    results = []
    exit_idx = -1

    for i in range(n - 1):
        if i <= exit_idx:
            continue
        if not signal_mask[i]:
            continue

        sl_pct = float(sl_arr[i])
        if sl_pct <= 0:
            continue

        entry    = close[i]          # 신호봉 종가 지정가
        sl_price = low[i]            # = entry × (1 - sl_pct)
        tp_price = 2 * close[i] - low[i]   # = entry × (1 + sl_pct)

        # 이길 때 수수료 0, 질 때만 테이커 0.01%
        fee_loss = TAKER / sl_pct    # 손실 시만 적용

        found = False
        for j in range(i + 1, min(i + max_hold + 1, n)):
            hit_sl = low[j]  <= sl_price
            hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                r = -1.0 - fee_loss
            elif hit_tp:
                r =  1.0              # 수수료 0
            elif hit_sl:
                r = -1.0 - fee_loss
            else:
                continue

            results.append(r)
            exit_idx = j
            found = True
            break

        if not found:
            j     = min(i + max_hold, n - 1)
            r_raw = (close[j] - entry) / (entry * sl_pct)
            results.append(r_raw - fee_loss)
            exit_idx = j

    return results


# ── 지표 빌드 ─────────────────────────────────────────────────────────
def build_indicators(df):
    df = df.copy()
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:, 0]
        pctb_cols = [c for c in bb.columns if c.startswith("BBP_")]
        if pctb_cols:
            df["pctb"] = bb[pctb_cols[0]]
    mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
    if mfi is not None:
        df["mfi"] = mfi
    rsi14 = ta.rsi(df["close"], length=14)
    if rsi14 is not None:
        df["rsi"] = rsi14
    rsi2 = ta.rsi(df["close"], length=2)
    if rsi2 is not None:
        df["rsi_2"] = rsi2
    sma200 = ta.sma(df["close"], length=200)
    if sma200 is not None:
        df["sma_200"] = sma200
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    if stoch is not None and len(stoch.columns) >= 2:
        df["slowk"] = stoch.iloc[:, 0]
        df["slowd"] = stoch.iloc[:, 1]
    return df


def get_signals_a3(df, strategy_id):
    """각 전략의 시그널 마스크와 sl_pct 배열 반환."""
    n = len(df)
    sig  = np.zeros(n, dtype=bool)
    sl_a = np.zeros(n, dtype=float)

    close = df["close"].values
    low   = df["low"].values

    # sl_pct = clip((close - low) / close, 0.001, 0.05)
    raw_sl = np.where(close > 0, (close - low) / close, 0.001)
    sl_arr = np.clip(raw_sl, 0.001, 0.05)

    if strategy_id == "A3_0033":
        if "bb_lower" not in df.columns:
            return sig, sl_a
        mask = (close < df["bb_lower"].values) & ~np.isnan(df["bb_lower"].values)
        sig  = mask
        sl_a = sl_arr

    elif strategy_id == "A3_0039":
        if "mfi" not in df.columns:
            return sig, sl_a
        mask = (df["mfi"].values < 20) & ~np.isnan(df["mfi"].values)
        sig  = mask
        sl_a = sl_arr

    elif strategy_id == "A3_0096":
        if "pctb" not in df.columns:
            return sig, sl_a
        pctb = df["pctb"].values
        sig[1:]  = (pctb[1:] < 0) & (pctb[1:] > pctb[:-1]) & ~np.isnan(pctb[1:])
        sl_a = sl_arr

    elif strategy_id == "A3_0012":
        if "sma_200" not in df.columns or "rsi_2" not in df.columns:
            return sig, sl_a
        sma = df["sma_200"].values
        r2  = df["rsi_2"].values
        mask = (close > sma) & (r2 <= 10) & ~np.isnan(sma) & ~np.isnan(r2)
        sig  = mask
        sl_a = sl_arr

    elif strategy_id == "A3_0070":
        if "slowk" not in df.columns or "slowd" not in df.columns:
            return sig, sl_a
        sk = df["slowk"].values
        sd = df["slowd"].values
        mask = (sk < 20) & (sk > sd) & ~np.isnan(sk) & ~np.isnan(sd)
        sig  = mask
        sl_a = sl_arr

    elif strategy_id == "A3_0063":
        if "rsi" not in df.columns or "sma_200" not in df.columns:
            return sig, sl_a
        rsi = df["rsi"].values
        sma = df["sma_200"].values
        mask = (rsi < 40) & (close > sma) & ~np.isnan(rsi) & ~np.isnan(sma)
        sig  = mask
        sl_a = sl_arr

    return sig, sl_a


def get_signals_bb_reversal(df):
    """%B < 0 AND close > prev_close — 고정 SL 0.8%."""
    n = len(df)
    sig  = np.zeros(n, dtype=bool)
    sl_a = np.full(n, 0.008)  # 고정 0.8%

    if "pctb" not in df.columns:
        return sig, sl_a

    pctb  = df["pctb"].values
    close = df["close"].values

    sig[1:] = (pctb[1:] < 0) & (close[1:] > close[:-1]) & ~np.isnan(pctb[1:])
    return sig, sl_a


# ── 분할 및 실행 ─────────────────────────────────────────────────────
def run_strategy(df_full, strategy_id, tf_min, get_sig_fn, max_hold_bars):
    df_tf  = resample(df_full, tf_min) if tf_min > 1 else df_full.copy()
    df_tf  = build_indicators(df_tf)

    df_is  = df_tf[df_tf["ts"].dt.year.isin([2020,2021,2022])].reset_index(drop=True)
    df_oos = df_tf[df_tf["ts"].dt.year.isin([2023,2024,2025])].reset_index(drop=True)

    rows = []
    for label, df in [("IS 2020-22", df_is), ("OOS 2023-25", df_oos)]:
        sig, sl_a = get_sig_fn(df)
        res_list  = run_bt(df, sig, sl_a, max_hold_bars)
        rows.append(summarize(res_list, label))

    return rows


# ── 메인 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("캐시 로드 중...")
    df_1m = pd.read_parquet(CACHE)
    print(f"  {len(df_1m):,}봉\n")

    all_rows = []

    # ── 1. candle-reversal - 기존 MEXC 결과 읽기 ──────────────────────
    print("[ candle-reversal - 기존 MEXC 0.01% 결과 ]")
    for tf in ("1m", "3m", "5m"):
        fp = Path(f"oos_{tf}_mexc.csv")
        if not fp.exists():
            print(f"  {tf}: 결과 없음")
            continue
        oos = pd.read_csv(fp)
        if oos.empty:
            continue
        best = oos.sort_values("OOS_exp", ascending=False).iloc[0]
        all_rows.append({
            "전략": f"candle-reversal {tf}",
            "기간": "OOS 2023-25",
            "거래": int(best["OOS_trades"]),
            "승률": f"{best['OOS_wr']:.1f}%",
            "기대값": f"{best['OOS_exp']:+.4f}R",
            "총R": f"{best['OOS_totalR']:+.1f}R",
            "MDD": f"{best['OOS_mdd_r']:.1f}R",
        })
    print()

    # ── 2. a3-bb-reversal (15m) ────────────────────────────────────────
    print("[ a3-bb-reversal — 15m, %B<0 AND close>prev, SL/TP 0.8% ]")
    rows = run_strategy(
        df_1m, "A3_BB_Reversal", 15,
        get_signals_bb_reversal, 48,
    )
    for r in rows:
        print(f"  {r['label']}: {r['trades']}건  WR={r['wr']}%  "
              f"E={r['exp']:+.4f}R  총={r['totalR']:+.1f}R  MDD={r['mdd']:.1f}R")
        all_rows.append({
            "전략": "a3-bb-reversal 15m",
            "기간": r["label"],
            "거래": r["trades"],
            "승률": f"{r['wr']}%",
            "기대값": f"{r['exp']:+.4f}R",
            "총R": f"{r['totalR']:+.1f}R",
            "MDD": f"{r['mdd']:.1f}R",
        })
    print()

    # ── 3. A3_xxx 전략들 ──────────────────────────────────────────────
    STRATEGIES = [
        ("A3_0012", [5, 15], 48),
        ("A3_0033", [5, 15], 48),
        ("A3_0039", [5],     48),
        ("A3_0063", [5],     48),
        ("A3_0070", [5],     48),
        ("A3_0096", [5],     48),
    ]

    for strat_id, tfs, max_hold in STRATEGIES:
        for tf in tfs:
            label_str = f"{strat_id} {tf}m"
            print(f"[ {label_str} ]")
            get_sig = lambda df, s=strat_id: get_signals_a3(df, s)
            rows = run_strategy(df_1m, strat_id, tf, get_sig, max_hold)
            for r in rows:
                print(f"  {r['label']}: {r['trades']}건  WR={r['wr']}%  "
                      f"E={r['exp']:+.4f}R  총={r['totalR']:+.1f}R  MDD={r['mdd']:.1f}R")
                all_rows.append({
                    "전략": label_str,
                    "기간": r["label"],
                    "거래": r["trades"],
                    "승률": f"{r['wr']}%",
                    "기대값": f"{r['exp']:+.4f}R",
                    "총R": f"{r['totalR']:+.1f}R",
                    "MDD": f"{r['mdd']:.1f}R",
                })
            print()

    # ── 전체 요약 ─────────────────────────────────────────────────────
    print("\n" + "="*85)
    print(f"{'전략':<24} {'기간':<14} {'거래':>6} {'승률':>7} {'기대값':>10} {'총R':>9} {'MDD':>8}")
    print("="*85)
    for r in all_rows:
        print(f"{r['전략']:<24} {r['기간']:<14} {r['거래']:>6} {r['승률']:>7} "
              f"{r['기대값']:>10} {r['총R']:>9} {r['MDD']:>8}")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv("backtest_all_results.csv", index=False, encoding="utf-8-sig")
    print("\n결과 저장: backtest_all_results.csv")
