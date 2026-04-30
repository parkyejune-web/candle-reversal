"""
장대봉 + 반대봉 역추세 전략 — 3분/5분봉 멀티 타임프레임 백테스트
IS: 2024  OOS: 2025  심볼: BTCUSDT
수수료: 진입·TP=지정가maker(+0.015%), SL=시장가taker(-0.050%)
실행: python backtest_tf.py
"""
import io, time as time_mod, zipfile, os
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

MAKER_REBATE = 0.00015
TAKER_COST   = 0.00050

PARAM_GRID = {
    "big_mult":    [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
    "cover_pct":   [0.3, 0.4, 0.5, 0.6, 0.7],
    "rr_ratio":    [1.5, 2.0, 2.5, 3.0, 3.5],
    "avg_len":     [10, 15, 20, 30],
    "min_sl_dist": [0, 50, 100, 150, 200, 300],
}
# 조합 수: 6×5×5×4×6 = 3,600


# ── 데이터 ────────────────────────────────────────────────────────────
_BASE = "https://data.binance.vision/data/spot/monthly/klines"

def _fetch_month(symbol, year, month):
    fname = f"{symbol}-1m-{year}-{month:02d}"
    url   = f"{_BASE}/{symbol}/1m/{fname}.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None,
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

def load_1m(symbol, start_year, end_year):
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

    print(f"Binance 다운로드: {symbol} 1m {start_year}~{end_year} ({len(months)}개월)")
    frames = []
    for i, (y, m) in enumerate(months, 1):
        print(f"  [{i}/{len(months)}] {y}-{m:02d}...", end=" ", flush=True)
        try:
            df_m = _fetch_month(symbol, y, m)
            frames.append(df_m)
            print(f"{len(df_m):,}봉")
        except Exception as e:
            print(f"SKIP ({e})")

    if not frames:
        raise RuntimeError("데이터 없음")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache)
    print(f"저장: {cache} ({len(df):,}봉)")
    return df


def resample(df_1m, minutes):
    """1분봉 → N분봉 리샘플링."""
    df = df_1m.set_index("ts")
    rule = f"{minutes}min"
    rs = df.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna().reset_index().rename(columns={"ts": "ts"})
    return rs


# ── 신호 감지 ─────────────────────────────────────────────────────────
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
    prev_body_arr = body[:-1]

    overlap = np.minimum(prev_top, curr_top) - np.maximum(prev_bot, curr_bot)
    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.where(prev_body_arr > 0, overlap / prev_body_arr, 0.0)

    short_sig = np.zeros(n, dtype=bool)
    long_sig  = np.zeros(n, dtype=bool)
    short_sig[1:] = is_big[:-1] & is_bull[:-1] & is_bear[1:] & (cover >= cover_pct)
    long_sig[1:]  = is_big[:-1] & is_bear[:-1] & is_bull[1:] & (cover >= cover_pct)
    return short_sig, long_sig


# ── 백테스트 ─────────────────────────────────────────────────────────
def run_backtest(df, short_sig, long_sig, rr_ratio,
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

        if sl_dist <= 0:
            continue
        if sl_dist < min_sl_dist:
            continue

        tp_price = (entry_price - sl_dist * rr_ratio if side == "short"
                    else entry_price + sl_dist * rr_ratio)

        lev      = entry_price / sl_dist
        fee_win  = (MAKER_REBATE + MAKER_REBATE) * lev
        fee_loss = (MAKER_REBATE - TAKER_COST)   * lev

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
        return {"trades": 0, "win_rate": 0.0, "expectancy": 0.0,
                "total_r": 0.0, "mdd_r": 0.0}

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


# ── 그리드 서치 ───────────────────────────────────────────────────────
def _run_combo(args):
    df, params, max_bars = args
    short_sig, long_sig = detect_signals(
        df, params["big_mult"], params["cover_pct"], params["avg_len"])
    res = run_backtest(df, short_sig, long_sig,
                       params["rr_ratio"], params["min_sl_dist"], max_bars)
    if res["trades"] >= MIN_TRADES:
        return {**params, **res}
    return None

def grid_search(df, param_grid, max_bars, label=""):
    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    total  = len(combos)
    print(f"{label} 파라미터 조합: {total}개  max_bars={max_bars}")

    args = [(df, dict(zip(keys, c)), max_bars) for c in combos]
    records = []
    t0 = time_mod.time()

    workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_run_combo, a): i for i, a in enumerate(args)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [경고] {e}")
                continue
            if res:
                records.append(res)
            if done % 600 == 0:
                elapsed = time_mod.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  {done}/{total}  경과 {elapsed:.0f}s  ETA {eta:.0f}s")

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("expectancy", ascending=False)


def validate_oos(df_oos, is_top, n=10, max_bars=1440):
    records = []
    for _, row in is_top.head(n).iterrows():
        short_sig, long_sig = detect_signals(
            df_oos, row["big_mult"], row["cover_pct"], row["avg_len"])
        res = run_backtest(df_oos, short_sig, long_sig,
                           row["rr_ratio"], row["min_sl_dist"], max_bars)
        records.append({
            "big_mult":    row["big_mult"],
            "cover_pct":   row["cover_pct"],
            "rr_ratio":    row["rr_ratio"],
            "avg_len":     int(row["avg_len"]),
            "min_sl_dist": int(row["min_sl_dist"]),
            "IS_exp":      round(row["expectancy"], 3),
            "IS_trades":   int(row["trades"]),
            "IS_wr":       round(row["win_rate"], 1),
            "IS_mdd_r":    round(row["mdd_r"], 1),
            "OOS_trades":  res["trades"],
            "OOS_wr":      round(res["win_rate"], 1),
            "OOS_exp":     round(res["expectancy"], 3),
            "OOS_totalR":  round(res["total_r"], 1),
            "OOS_mdd_r":   round(res["mdd_r"], 1),
        })
    return pd.DataFrame(records).sort_values("OOS_exp", ascending=False)


# ── 메인 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("수수료: 진입·TP=지정가maker(+0.015%), SL=시장가taker(-0.050%)")
    print()

    df_1m = load_1m(SYMBOL, 2024, 2025)

    for tf_min in [3, 5]:
        print(f"\n{'='*65}")
        print(f"  타임프레임: {tf_min}분봉")
        print(f"{'='*65}")

        df_tf  = resample(df_1m, tf_min)
        df_is  = df_tf[df_tf["ts"].dt.year == 2024].reset_index(drop=True)
        df_oos = df_tf[df_tf["ts"].dt.year == 2025].reset_index(drop=True)

        # max_bars: 24시간을 해당 TF 봉 수로
        max_bars = 1440 // tf_min

        print(f"IS  2024: {len(df_is):,}봉")
        print(f"OOS 2025: {len(df_oos):,}봉")
        print()

        label = f"[{tf_min}m IS 2024]"
        t0    = time_mod.time()
        is_df = grid_search(df_is, PARAM_GRID, max_bars, label=label)
        elapsed = time_mod.time() - t0
        print(f"완료: {elapsed:.1f}초  유효 조합: {len(is_df)}개")

        if is_df.empty:
            print("유효 조합 없음")
            continue

        print(f"\n=== {tf_min}m IS 2024 상위 15개 ===")
        print(is_df.head(15).to_string(index=False))
        is_df.to_csv(f"is_{tf_min}m.csv", index=False)

        print(f"\n=== {tf_min}m OOS 2025 검증 (IS 상위 20) ===")
        oos_df = validate_oos(df_oos, is_df, n=20, max_bars=max_bars)
        print(oos_df.to_string(index=False))
        oos_df.to_csv(f"oos_{tf_min}m.csv", index=False)

        survivors = oos_df[oos_df["OOS_exp"] > 0]
        print(f"\nOOS 양수: {len(survivors)}/20")
        if len(survivors):
            best = survivors.iloc[0]
            print(f"  최고: big={best['big_mult']} cover={best['cover_pct']}"
                  f" rr={best['rr_ratio']} avg={best['avg_len']}"
                  f" msl={best['min_sl_dist']}"
                  f"  OOS exp={best['OOS_exp']}R  WR={best['OOS_wr']}%"
                  f"  {best['OOS_trades']}건/년")

    print("\n완료. 결과: is_3m.csv / oos_3m.csv / is_5m.csv / oos_5m.csv")
