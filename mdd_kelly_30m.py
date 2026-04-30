"""
30분봉 전략 OOS MDD 정확 계산 + Kelly 비율 산출
Usage: python mdd_kelly_30m.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
SYMBOL    = "BTCUSDT"

PARAMS = dict(big_mult=2.4, cover_pct=0.7, rr_ratio=4.3, avg_len=13, taker_fee=0.0001)

IS_START  = pd.Timestamp("2020-01-01", tz="UTC")
IS_END    = pd.Timestamp("2023-01-01", tz="UTC")
OOS_START = pd.Timestamp("2023-01-01", tz="UTC")
OOS_END   = pd.Timestamp("2026-04-01", tz="UTC")


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


@njit(cache=True)
def _backtest_r(high, low, open_, close, short_sig, long_sig,
                rr_ratio, taker_fee, max_bars=1440):
    n = len(high)
    res_r = np.empty(n, dtype=np.float64)
    cnt = 0
    exit_idx = -1

    for i in range(n - 1):
        if i <= exit_idx:
            continue
        if not (short_sig[i] or long_sig[i]):
            continue

        if short_sig[i]:
            sl_price    = high[i]
            entry_price = open_[i + 1]
            sl_dist     = sl_price - entry_price
            side        = -1
        else:
            sl_price    = low[i]
            entry_price = open_[i + 1]
            sl_dist     = entry_price - sl_price
            side        = 1

        if sl_dist <= 0.0:
            continue

        tp_price  = (entry_price - sl_dist * rr_ratio if side == -1
                     else entry_price + sl_dist * rr_ratio)
        lev       = entry_price / sl_dist
        fee_entry = taker_fee * lev
        fee_sl    = taker_fee * lev

        found = False
        j_end = min(i + max_bars + 1, n)
        for j in range(i + 1, j_end):
            if side == -1:
                hit_sl = high[j] >= sl_price
                hit_tp = low[j]  <= tp_price
            else:
                hit_sl = low[j]  <= sl_price
                hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                r = -1.0 - fee_entry - fee_sl
            elif hit_sl:
                r = -1.0 - fee_entry - fee_sl
            elif hit_tp:
                r = rr_ratio - fee_entry
            else:
                continue

            res_r[cnt] = r
            cnt += 1
            exit_idx = j
            found = True
            break

        if not found:
            j = min(i + max_bars, n - 1)
            r_raw = ((entry_price - close[j]) / sl_dist if side == -1
                     else (close[j] - entry_price) / sl_dist)
            res_r[cnt] = r_raw - fee_entry - fee_sl
            cnt += 1
            exit_idx = j

    return res_r[:cnt]


def calc_stats(r_arr, label):
    if len(r_arr) == 0:
        print(f"{label}: 거래 없음")
        return {}

    n      = len(r_arr)
    wins   = int((r_arr > 0).sum())
    losses = int((r_arr < 0).sum())
    wr     = wins / n * 100
    exp    = float(r_arr.mean())
    total  = float(r_arr.sum())

    # MDD (R 기준)
    cum  = np.cumsum(r_arr)
    peak = np.maximum.accumulate(cum)
    dd   = peak - cum
    mdd  = float(dd.max())

    # MDD 발생 위치
    mdd_end   = int(np.argmax(dd))
    mdd_start = int(np.argmax(cum[:mdd_end + 1])) if mdd_end > 0 else 0

    # 최대 연속 손실
    max_consec_loss = 0
    cur_consec = 0
    for r in r_arr:
        if r < 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  거래수       : {n:,}건")
    print(f"  승/패        : {wins}승 {losses}패")
    print(f"  승률         : {wr:.2f}%")
    print(f"  기대값       : {exp:+.4f}R")
    print(f"  총 수익      : {total:+.1f}R")
    print(f"  MDD          : {mdd:.1f}R  (거래 #{mdd_start}→#{mdd_end})")
    print(f"  최대연속손실  : {max_consec_loss}연패")

    return dict(n=n, wins=wins, wr=wr, exp=exp, total=total, mdd=mdd,
                max_consec=max_consec_loss)


def kelly_analysis(stats, rr_ratio, label):
    wr  = stats["wr"] / 100
    q   = 1 - wr
    mdd = stats["mdd"]

    full_kelly = (wr * rr_ratio - q) / rr_ratio * 100
    half_kelly = full_kelly / 2

    print(f"\n{'='*50}")
    print(f"  Kelly 분석 - {label}")
    print(f"{'='*50}")
    print(f"  WR={wr*100:.2f}%  RR={rr_ratio}  MDD={mdd:.1f}R")
    print(f"  Full Kelly   : {full_kelly:.2f}%")
    print(f"  Half Kelly   : {half_kelly:.2f}%")
    print()
    print(f"  * MDD={mdd:.0f}R 기준 리스크별 복리 최대낙폭 추정:")
    print(f"  {'리스크':>8} | {'단순MDD':>9} | {'복리MDD':>9} | {'판정':>12}")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*12}")

    for pct in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, half_kelly, full_kelly]:
        if pct <= 0:
            continue
        p = pct / 100
        simple_mdd = mdd * pct
        compound_mdd = (1 - (1 - p) ** mdd) * 100
        if compound_mdd < 20:
            verdict = "[OK] 매우 안전"
        elif compound_mdd < 35:
            verdict = "[OK] 안전"
        elif compound_mdd < 50:
            verdict = "[!]  주의"
        elif compound_mdd < 70:
            verdict = "[X]  위험"
        else:
            verdict = "[!!] 파탄위험"
        marker = " <--" if abs(pct - half_kelly) < 0.05 or abs(pct - full_kelly) < 0.05 else ""
        print(f"  {pct:>7.2f}% | {simple_mdd:>8.1f}% | {compound_mdd:>8.1f}% | {verdict}{marker}")

    print()
    target_mdd_pct = 25.0
    safe_pct = (1 - (1 - target_mdd_pct / 100) ** (1 / mdd)) * 100
    print(f"  -> 복리MDD 25% 이하 목표 리스크 상한: {safe_pct:.2f}%")
    print(f"  -> 복리MDD 40% 이하 목표 리스크 상한: "
          f"{(1-(1-0.40)**(1/mdd))*100:.2f}%")
    print()
    print(f"  ★ 추천 리스크: {safe_pct:.1f}% (복리MDD 25% 이내)")


def main():
    p = PARAMS
    df = load_30m()

    df_is  = df[(df["ts"] >= IS_START)  & (df["ts"] < IS_END)].reset_index(drop=True)
    df_oos = df[(df["ts"] >= OOS_START) & (df["ts"] < OOS_END)].reset_index(drop=True)

    print(f"\n파라미터: big_mult={p['big_mult']}  cover_pct={p['cover_pct']}  "
          f"rr_ratio={p['rr_ratio']}  avg_len={p['avg_len']}  fee={p['taker_fee']*100:.3f}%")

    for label, df_seg in [("IS  2020-2022", df_is), ("OOS 2023-2026", df_oos)]:
        short_sig, long_sig = detect_signals(df_seg, p["big_mult"], p["cover_pct"], p["avg_len"])
        r_arr = _backtest_r(
            df_seg["high"].values, df_seg["low"].values,
            df_seg["open"].values, df_seg["close"].values,
            short_sig, long_sig,
            p["rr_ratio"], p["taker_fee"]
        )
        stats = calc_stats(r_arr, label)
        if stats:
            kelly_analysis(stats, p["rr_ratio"], label)


if __name__ == "__main__":
    main()
