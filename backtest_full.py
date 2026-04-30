"""
MEXC 0.01%/side 시장가 수수료 | 장대봉+반대봉 역추세 전략
데이터: BTCUSDT 1m 2020~2025  TF: 1m / 3m / 5m
IS: 2020~2022  |  OOS: 2023~2025
Kelly 비중 최적화 포함
실행: python backtest_full.py
"""
import io, os, time as time_mod, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR  = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
SYMBOL     = "BTCUSDT"
MIN_TRADES = 30

TAKER = 0.00010   # MEXC 0.01% per side (진입 + 청산 각각)

PARAM_GRID = {
    "big_mult":   [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
    "cover_pct":  [0.3, 0.4, 0.5, 0.6],
    "rr_ratio":   [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
    "avg_len":    [5, 10, 15, 20],
    "min_sl_pct": [0.0, 0.0005, 0.001, 0.002, 0.003],
}
# 6×4×6×4×5 = 2,880 조합

IS_YEARS  = [2020, 2021, 2022]
OOS_YEARS = [2023, 2024, 2025]
ALL_YEARS = IS_YEARS + OOS_YEARS
TF_MINUTES = [1, 3, 5]


# ─────────────────────────── DATA ────────────────────────────────────────────
_BASE = "https://data.binance.vision/data/spot/monthly/klines"


def _fetch_month(symbol, year, month):
    fname = f"{symbol}-1m-{year}-{month:02d}"
    r = requests.get(f"{_BASE}/{symbol}/1m/{fname}.zip", timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(
                f, header=None,
                names=["ts","open","high","low","close","volume",
                       "close_ts","qvol","ntrades","tb","tq","ign"],
                usecols=["ts","open","high","low","close","volume"])
    for c in ["ts","open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    ts_vals = df["ts"].astype("int64").values
    unit = "us" if ts_vals.max() > 1e13 else "ms"
    df["ts"] = pd.to_datetime(ts_vals, unit=unit, utc=True)
    return df


def load_data(symbol, start_year, end_year):
    cache = CACHE_DIR / f"{symbol}_1m_{start_year}_{end_year}.parquet"
    if cache.exists():
        print(f"캐시 로드: {cache}")
        df = pd.read_parquet(cache)
        print(f"  {len(df):,}봉")
        return df

    today  = pd.Timestamp.now()
    months = [(y, m) for y in range(start_year, end_year + 1)
              for m in range(1, 13)
              if pd.Timestamp(year=y, month=m, day=1) < today]

    print(f"다운로드: {symbol} 1m {start_year}~{end_year} ({len(months)}개월)")
    frames = []
    for i, (y, m) in enumerate(months, 1):
        print(f"  [{i}/{len(months)}] {y}-{m:02d}...", end=" ", flush=True)
        try:
            frames.append(_fetch_month(symbol, y, m))
            print(f"{len(frames[-1]):,}봉")
        except Exception as e:
            print(f"SKIP ({e})")

    if not frames:
        raise RuntimeError("데이터 없음")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache)
    print(f"저장: {cache}  ({len(df):,}봉)")
    return df


def resample_tf(df_1m, minutes):
    if minutes == 1:
        return df_1m.copy()
    rs = (df_1m.set_index("ts")
          .resample(f"{minutes}min")
          .agg(open=("open","first"), high=("high","max"),
               low=("low","min"),    close=("close","last"),
               volume=("volume","sum"))
          .dropna()
          .reset_index())
    return rs


# ─────────────────────────── SIGNALS ─────────────────────────────────────────
def detect_signals(df, big_mult, cover_pct, avg_len):
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)

    body     = np.abs(close - open_)
    avg_body = (pd.Series(body)
                .rolling(int(avg_len), min_periods=int(avg_len))
                .mean().values)

    is_big  = body >= avg_body * big_mult
    is_bull = close > open_
    is_bear = close < open_

    prev_top      = np.maximum(open_[:-1], close[:-1])
    prev_bot      = np.minimum(open_[:-1], close[:-1])
    curr_top      = np.maximum(open_[1:],  close[1:])
    curr_bot      = np.minimum(open_[1:],  close[1:])
    prev_body_arr = body[:-1]

    overlap = np.minimum(prev_top, curr_top) - np.maximum(prev_bot, curr_bot)
    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(prev_body_arr > 0, overlap / prev_body_arr, 0.0)

    short_sig = np.zeros(n, dtype=bool)
    long_sig  = np.zeros(n, dtype=bool)
    short_sig[1:] = is_big[:-1] & is_bull[:-1] & is_bear[1:] & (cover >= cover_pct)
    long_sig[1:]  = is_big[:-1] & is_bear[:-1] & is_bull[1:] & (cover >= cover_pct)
    return short_sig, long_sig


# ─────────────────────────── BACKTEST ────────────────────────────────────────
def run_backtest(df, short_sig, long_sig, rr_ratio,
                 min_sl_pct=0.0, max_bars=1440, keep_arr=False):
    """
    수수료: 진입 TAKER(0.01%) + 청산 TAKER(0.01%) = 왕복 0.02%
    fee (R단위) = 2 × TAKER × lev  (승패 동일 적용)
    """
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

        if sl_dist <= 0:
            continue
        if min_sl_pct > 0 and (sl_dist / entry_price) < min_sl_pct:
            continue

        tp_price = (entry_price - sl_dist * rr_ratio if side == "short"
                    else entry_price + sl_dist * rr_ratio)

        lev = entry_price / sl_dist
        fee = 2.0 * TAKER * lev   # 진입 0.01% + 청산 0.01%

        found = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if side == "short":
                hit_sl = high[j] >= sl_price
                hit_tp = low[j]  <= tp_price
            else:
                hit_sl = low[j]  <= sl_price
                hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                r = -1.0 - fee
            elif hit_sl:
                r = -1.0 - fee
            elif hit_tp:
                r = float(rr_ratio) - fee
            else:
                continue

            results_r.append(r)
            exit_idx = j
            found = True
            break

        if not found:
            j     = min(i + max_bars, n - 1)
            r_raw = ((entry_price - close[j]) / sl_dist if side == "short"
                     else (close[j] - entry_price) / sl_dist)
            results_r.append(r_raw - fee)
            exit_idx = j

    if not results_r:
        return None

    arr  = np.array(results_r)
    wins = (arr > 0).sum()
    cum  = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    mdd  = float((peak - cum).max())
    out  = {
        "trades":     len(arr),
        "win_rate":   wins / len(arr) * 100,
        "expectancy": float(arr.mean()),
        "total_r":    float(arr.sum()),
        "mdd_r":      mdd,
    }
    if keep_arr:
        out["arr"] = arr
    return out


# ─────────────────────────── KELLY ───────────────────────────────────────────
def kelly_optimize(arr):
    """기하 평균 최대화 → 최적 Kelly 비중."""
    worst = float(arr.min())
    f_max = min(0.25, 0.99 / abs(worst)) if worst < 0 else 0.25

    records = []
    for f in np.linspace(0.001, f_max, 300):
        port = 1.0 + f * arr
        if np.any(port <= 0):
            break
        log_g  = float(np.log(port).mean())
        cum    = np.cumprod(port)
        peak   = np.maximum.accumulate(cum)
        mdd    = float(((peak - cum) / peak).max() * 100)
        records.append({
            "frac_pct": round(f * 100, 3),
            "log_g":    log_g,
            "final_x":  float(cum[-1]),
            "mdd_pct":  mdd,
        })

    if not records:
        return None, pd.DataFrame()

    df_k     = pd.DataFrame(records)
    best_idx = df_k["log_g"].idxmax()
    best_f   = float(df_k.loc[best_idx, "frac_pct"])
    return best_f, df_k


# ─────────────────────────── GRID SEARCH ─────────────────────────────────────
def _run_combo_worker(args):
    df, params, max_bars = args
    ss, ls = detect_signals(df, params["big_mult"],
                            params["cover_pct"], params["avg_len"])
    res = run_backtest(df, ss, ls, params["rr_ratio"],
                       params["min_sl_pct"], max_bars, keep_arr=False)
    if res and res["trades"] >= MIN_TRADES:
        return {**params, **res}
    return None


def grid_search(df, max_bars, label=""):
    keys   = list(PARAM_GRID.keys())
    combos = list(product(*PARAM_GRID.values()))
    total  = len(combos)
    print(f"\n{label} 그리드 서치: {total}개 조합  max_bars={max_bars}")

    args    = [(df, dict(zip(keys, c)), max_bars) for c in combos]
    records = []
    t0      = time_mod.time()
    workers = min(os.cpu_count() or 4, 8)

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_run_combo_worker, a): i for i, a in enumerate(args)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result()
            except Exception as e:
                print(f"  경고: {e}")
                continue
            if res:
                records.append(res)
            if done % 500 == 0:
                el  = time_mod.time() - t0
                eta = el / done * (total - done)
                print(f"  {done}/{total}  {el:.0f}s  ETA {eta:.0f}s  유효:{len(records)}")

    elapsed = time_mod.time() - t0
    print(f"완료: {elapsed:.1f}s  유효: {len(records)}개")
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("expectancy", ascending=False)


# ─────────────────────────── PER-YEAR STATS ──────────────────────────────────
def year_breakdown(df_tf, params, max_bars, years):
    rows = []
    for yr in years:
        sub = df_tf[df_tf["ts"].dt.year == yr].reset_index(drop=True)
        if len(sub) < 200:
            rows.append(dict(year=yr, trades=0, wr=0.0,
                             exp=0.0, total_r=0.0, mdd_r=0.0))
            continue
        ss, ls = detect_signals(sub, params["big_mult"],
                                 params["cover_pct"], params["avg_len"])
        res = run_backtest(sub, ss, ls, params["rr_ratio"],
                           params["min_sl_pct"], max_bars)
        if res:
            rows.append(dict(year=yr, trades=res["trades"],
                             wr=round(res["win_rate"], 1),
                             exp=round(res["expectancy"], 4),
                             total_r=round(res["total_r"], 1),
                             mdd_r=round(res["mdd_r"], 1)))
        else:
            rows.append(dict(year=yr, trades=0, wr=0.0,
                             exp=0.0, total_r=0.0, mdd_r=0.0))
    return pd.DataFrame(rows)


# ─────────────────────────── MAIN ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("  MEXC 0.01%/side 수수료 | 장대봉+반대봉 역추세 전략 완전 검증")
    print(f"  IS: {IS_YEARS}  →  OOS: {OOS_YEARS}")
    print("  수수료: 진입 0.01% + 청산 0.01% = 왕복 0.02% (모두 시장가)")
    print("=" * 72)
    print()

    df_1m = load_data(SYMBOL, 2020, 2025)

    for tf_min in TF_MINUTES:
        print(f"\n{'#'*72}")
        print(f"  ▶ {tf_min}분봉 분석")
        print(f"{'#'*72}")

        df_tf  = resample_tf(df_1m, tf_min)
        df_is  = df_tf[df_tf["ts"].dt.year.isin(IS_YEARS)].reset_index(drop=True)
        df_oos = df_tf[df_tf["ts"].dt.year.isin(OOS_YEARS)].reset_index(drop=True)
        max_bars = 1440 // tf_min

        print(f"IS  ({IS_YEARS[0]}~{IS_YEARS[-1]}): {len(df_is):,}봉")
        print(f"OOS ({OOS_YEARS[0]}~{OOS_YEARS[-1]}): {len(df_oos):,}봉")

        # ── 1. IS 그리드 서치 ─────────────────────────────────────────────
        lbl   = f"[{tf_min}m IS {IS_YEARS[0]}-{IS_YEARS[-1]}]"
        is_df = grid_search(df_is, max_bars, label=lbl)
        if is_df.empty:
            print("유효 조합 없음 — 스킵")
            continue

        is_df.to_csv(f"is_{tf_min}m_mexc.csv", index=False)

        SCOLS = ["big_mult","cover_pct","rr_ratio","avg_len","min_sl_pct",
                 "trades","win_rate","expectancy","total_r","mdd_r"]
        print(f"\n── IS {IS_YEARS[0]}~{IS_YEARS[-1]} 상위 10개 ({tf_min}m) ──")
        print(is_df[SCOLS].head(10).to_string(index=False, float_format="%.4f"))

        # ── 2. OOS 검증 (IS 상위 5) ──────────────────────────────────────
        print(f"\n── OOS 검증 (IS 상위 5 파라미터) ──")
        oos_records = []
        for _, row in is_df.head(5).iterrows():
            p = {k: row[k] for k in PARAM_GRID}
            ss_oos, ls_oos = detect_signals(
                df_oos, p["big_mult"], p["cover_pct"], p["avg_len"])
            res_oos = run_backtest(
                df_oos, ss_oos, ls_oos, p["rr_ratio"], p["min_sl_pct"], max_bars)
            if res_oos:
                oos_records.append({
                    **p,
                    "IS_exp":     round(row["expectancy"], 4),
                    "OOS_trades": res_oos["trades"],
                    "OOS_wr":     round(res_oos["win_rate"], 1),
                    "OOS_exp":    round(res_oos["expectancy"], 4),
                    "OOS_totalR": round(res_oos["total_r"], 1),
                    "OOS_mdd_r":  round(res_oos["mdd_r"], 1),
                })

        if oos_records:
            oos_val = (pd.DataFrame(oos_records)
                       .sort_values("OOS_exp", ascending=False))
            print(oos_val.to_string(index=False))
            oos_val.to_csv(f"oos_{tf_min}m_mexc.csv", index=False)

        # ── 3. 연도별 성과 (IS Best 파라미터) ────────────────────────────
        best_p = {k: is_df.iloc[0][k] for k in PARAM_GRID}
        tag = (f"big={best_p['big_mult']} cover={best_p['cover_pct']} "
               f"rr={best_p['rr_ratio']} avg={int(best_p['avg_len'])} "
               f"msl%={best_p['min_sl_pct']*100:.2f}%")
        print(f"\n── 연도별 성과 (IS Best: {tag}) ──")
        yr_df = year_breakdown(df_tf, best_p, max_bars, ALL_YEARS)
        print(yr_df.to_string(index=False))

        # ── 4. Kelly 비중 최적화 ──────────────────────────────────────────
        print(f"\n── Kelly 비중 최적화 ({tf_min}m) ──")

        ss_is, ls_is = detect_signals(
            df_is, best_p["big_mult"], best_p["cover_pct"], best_p["avg_len"])
        res_is_full = run_backtest(
            df_is, ss_is, ls_is, best_p["rr_ratio"],
            best_p["min_sl_pct"], max_bars, keep_arr=True)

        if not res_is_full or res_is_full["trades"] < 20:
            print("IS 거래 부족 — 스킵")
            continue

        arr_is   = res_is_full["arr"]
        best_f, kelly_df = kelly_optimize(arr_is)
        kelly_df.to_csv(f"kelly_{tf_min}m.csv", index=False)

        if best_f is None:
            print("Kelly 최적화 실패")
            continue

        print(f"\n  IS 기댓값: {res_is_full['expectancy']:+.4f}R  "
              f"거래수: {res_is_full['trades']}  MDD(R): {res_is_full['mdd_r']:.1f}")
        print()
        print(f"  {'비중':>6} {'거래당복리':>12} {'최종배수(IS)':>13} {'MDD%':>8}")
        print(f"  {'-'*45}")

        show_f = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]
        for sf in show_f:
            match = kelly_df[kelly_df["frac_pct"].between(sf * 0.8, sf * 1.2)]
            if len(match) == 0:
                match = kelly_df[(kelly_df["frac_pct"] - sf).abs() < 0.5]
            if len(match) == 0:
                continue
            r = match.sort_values("frac_pct").iloc[0]
            g_pct = (np.exp(r["log_g"]) - 1) * 100
            tag   = " ◀ 최적" if abs(r["frac_pct"] - best_f) < 0.4 else ""
            print(f"  {r['frac_pct']:>5.1f}%  {g_pct:>+11.4f}%  "
                  f"{r['final_x']:>12.2f}x  {r['mdd_pct']:>7.1f}%{tag}")

        print(f"\n  ★ IS 기준 최적 Kelly 비중: {best_f:.2f}%")

        # OOS에 동일 Kelly 적용
        ss_oos2, ls_oos2 = detect_signals(
            df_oos, best_p["big_mult"], best_p["cover_pct"], best_p["avg_len"])
        res_oos2 = run_backtest(
            df_oos, ss_oos2, ls_oos2, best_p["rr_ratio"],
            best_p["min_sl_pct"], max_bars, keep_arr=True)

        if res_oos2 and "arr" in res_oos2:
            arr_oos = res_oos2["arr"]
            f = best_f / 100.0

            def _compound_stats(arr, f):
                cum  = np.cumprod(1.0 + f * arr)
                peak = np.maximum.accumulate(cum)
                mdd  = float(((peak - cum) / peak).max() * 100)
                return float(cum[-1]), mdd

            fin_is,  mdd_is  = _compound_stats(arr_is,  f)
            fin_oos, mdd_oos = _compound_stats(arr_oos, f)

            print(f"\n  복리 시뮬레이션 (Kelly {best_f:.1f}% / 거래당 리스크):")
            print(f"  IS  {IS_YEARS[0]}~{IS_YEARS[-1]}:  "
                  f"{(fin_is-1)*100:>+,.1f}%  MDD {mdd_is:.1f}%  "
                  f"({res_is_full['trades']}건)")
            print(f"  OOS {OOS_YEARS[0]}~{OOS_YEARS[-1]}: "
                  f"{(fin_oos-1)*100:>+,.1f}%  MDD {mdd_oos:.1f}%  "
                  f"({res_oos2['trades']}건)")

            # OOS 연도별 Kelly 복리
            print(f"\n  OOS 연도별 Kelly {best_f:.1f}% 복리:")
            for yr in OOS_YEARS:
                sub = df_tf[df_tf["ts"].dt.year == yr].reset_index(drop=True)
                if len(sub) < 200:
                    continue
                ss_y, ls_y = detect_signals(sub, best_p["big_mult"],
                                             best_p["cover_pct"], best_p["avg_len"])
                r_y = run_backtest(sub, ss_y, ls_y, best_p["rr_ratio"],
                                   best_p["min_sl_pct"], max_bars, keep_arr=True)
                if r_y and "arr" in r_y:
                    fin_y, mdd_y = _compound_stats(r_y["arr"], f)
                    print(f"    {yr}: {(fin_y-1)*100:>+,.1f}%  MDD {mdd_y:.1f}%  "
                          f"({r_y['trades']}건  exp={r_y['expectancy']:+.4f}R)")

    print("\n\n완료. 저장된 파일:")
    for tf in TF_MINUTES:
        print(f"  is_{tf}m_mexc.csv  /  oos_{tf}m_mexc.csv  /  kelly_{tf}m.csv")
