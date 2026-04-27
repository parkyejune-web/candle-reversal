"""
장대봉 + 반대봉 역추세 전략 — 파라미터 최적화 + OOS 검증
IS: 2024  OOS: 2025  심볼: BTCUSDT 1분봉
실행: python backtest.py
"""
import io
import time as time_mod
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from itertools import product

CACHE_DIR  = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
SYMBOL     = "BTCUSDT"
MIN_TRADES = 30  # 최소 거래 수 (필터)

# ── 파라미터 그리드 ───────────────────────────────────────────────────
PARAM_GRID = {
    "big_mult":  [1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
    "cover_pct": [0.3, 0.4, 0.5, 0.6, 0.7],
    "rr_ratio":  [1.5, 2.0, 2.5, 3.0, 3.5],
    "avg_len":   [10, 15, 20, 30],
}


# ── 데이터 다운로드 (Binance bulk zip) ───────────────────────────────
_BASE = "https://data.binance.vision/data/spot/monthly/klines"

def _fetch_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    fname = f"{symbol}-1m-{year}-{month:02d}"
    url   = f"{_BASE}/{symbol}/1m/{fname}.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, header=None,
                             names=["ts","open","high","low","close","volume",
                                    "close_ts","qvol","ntrades","tb","tq","ign"],
                             usecols=["ts","open","high","low","close","volume"])
    for c in ["ts","open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    ts_vals = df["ts"].astype("int64").values
    # 바이낸스 2025+는 마이크로초(μs) 타임스탬프 사용 (2024까지는 ms)
    unit = "us" if ts_vals.max() > 1e13 else "ms"
    df["ts"] = pd.to_datetime(ts_vals, unit=unit, utc=True)
    return df

def download_1m(symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    cache = CACHE_DIR / f"{symbol}_1m_{start_year}_{end_year}.parquet"
    if cache.exists():
        print(f"캐시 로드: {cache}")
        df = pd.read_parquet(cache)
        print(f"  {len(df):,}봉")
        return df

    import calendar
    today = pd.Timestamp.now()
    months = [(y, m) for y in range(start_year, end_year + 1)
              for m in range(1, 13)
              if pd.Timestamp(year=y, month=m, day=1) < today]

    print(f"Binance bulk 다운로드: {symbol} 1m {start_year}~{end_year} ({len(months)}개월)")
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
        raise RuntimeError(f"다운로드 실패: {symbol} {start_year}~{end_year} 데이터 없음")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache)
    print(f"저장: {cache} ({len(df):,}봉)")
    return df


# ── 신호 감지 (벡터화) ────────────────────────────────────────────────
def detect_signals(df: pd.DataFrame, big_mult: float,
                   cover_pct: float, avg_len: int):
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

    prev_top  = np.maximum(open_[:-1], close[:-1])
    prev_bot  = np.minimum(open_[:-1], close[:-1])
    curr_top  = np.maximum(open_[1:],  close[1:])
    curr_bot  = np.minimum(open_[1:],  close[1:])
    prev_body = body[:-1]

    overlap = np.minimum(prev_top, curr_top) - np.maximum(prev_bot, curr_bot)
    cover   = np.where(prev_body > 0, overlap / prev_body, 0.0)

    short_sig = np.zeros(n, dtype=bool)
    long_sig  = np.zeros(n, dtype=bool)
    short_sig[1:] = is_big[:-1] & is_bull[:-1] & is_bear[1:] & (cover >= cover_pct)
    long_sig[1:]  = is_big[:-1] & is_bear[:-1] & is_bull[1:] & (cover >= cover_pct)

    return short_sig, long_sig


# ── 백테스트 ──────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, short_sig, long_sig,
                 rr_ratio: float, max_bars: int = 1440) -> dict:
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

        tp_price = (entry_price - sl_dist * rr_ratio if side == "short"
                    else entry_price + sl_dist * rr_ratio)

        found = False
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if side == "short":
                hit_sl = high[j] >= sl_price
                hit_tp = low[j]  <= tp_price
            else:
                hit_sl = low[j]  <= sl_price
                hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:
                r = -1.0          # 보수: SL 먼저
            elif hit_sl:
                r = -1.0
            elif hit_tp:
                r = float(rr_ratio)
            else:
                continue

            results_r.append(r)
            exit_idx = j
            found = True
            break

        if not found:
            # 최대 보유 초과 → 종가 청산
            j = min(i + max_bars, n - 1)
            exit_price = close[j]
            r = ((entry_price - exit_price) / sl_dist if side == "short"
                 else (exit_price - entry_price) / sl_dist)
            results_r.append(r)
            exit_idx = j

    if not results_r:
        return {"trades": 0, "win_rate": 0.0, "expectancy": 0.0, "total_r": 0.0, "mdd_r": 0.0}

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


# ── 그리드 서치 (멀티프로세싱) ───────────────────────────────────────
def _run_combo(args):
    df, params = args
    short_sig, long_sig = detect_signals(
        df, params["big_mult"], params["cover_pct"], params["avg_len"])
    res = run_backtest(df, short_sig, long_sig, params["rr_ratio"])
    if res["trades"] >= MIN_TRADES:
        return {**params, **res}
    return None

def grid_search(df: pd.DataFrame, param_grid: dict) -> pd.DataFrame:
    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    total  = len(combos)
    print(f"파라미터 조합: {total}개")

    args = [(df, dict(zip(keys, c))) for c in combos]
    records = []
    t0 = time_mod.time()

    import os
    workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_run_combo, a): i for i, a in enumerate(args)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [경고] 조합 오류: {e}")
                continue
            if res:
                records.append(res)
            if done % 100 == 0:
                elapsed = time_mod.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  {done}/{total}  경과 {elapsed:.0f}s  ETA {eta:.0f}s")

    if not records:
        return pd.DataFrame()
    result_df = pd.DataFrame(records).sort_values("expectancy", ascending=False)
    return result_df


# ── OOS 검증 ─────────────────────────────────────────────────────────
def validate_oos(df_oos: pd.DataFrame, is_top: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    records = []
    for _, row in is_top.head(n).iterrows():
        short_sig, long_sig = detect_signals(
            df_oos, row["big_mult"], row["cover_pct"], row["avg_len"])
        res = run_backtest(df_oos, short_sig, long_sig, row["rr_ratio"])
        records.append({
            "big_mult":   row["big_mult"],
            "cover_pct":  row["cover_pct"],
            "rr_ratio":   row["rr_ratio"],
            "avg_len":    int(row["avg_len"]),
            "IS_trades":  int(row["trades"]),
            "IS_wr":      round(row["win_rate"], 1),
            "IS_exp":     round(row["expectancy"], 3),
            "IS_mdd_r":   round(row["mdd_r"], 1),
            "OOS_trades": res["trades"],
            "OOS_wr":     round(res["win_rate"], 1),
            "OOS_exp":    round(res["expectancy"], 3),
            "OOS_totalR": round(res["total_r"], 1),
            "OOS_mdd_r":  round(res["mdd_r"], 1),
        })
    return pd.DataFrame(records).sort_values("OOS_exp", ascending=False)


# ── 메인 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 데이터
    df_all = download_1m(SYMBOL, 2024, 2025)
    df_is  = df_all[df_all["ts"].dt.year == 2024].reset_index(drop=True)
    df_oos = df_all[df_all["ts"].dt.year == 2025].reset_index(drop=True)
    print(f"\nIS  2024: {len(df_is):,}봉")
    print(f"OOS 2025: {len(df_oos):,}봉\n")

    # IS 그리드 서치
    print("=" * 50)
    print("[IS 2024] 그리드 서치 시작...")
    print("=" * 50)
    t0       = time_mod.time()
    is_df    = grid_search(df_is, PARAM_GRID)
    elapsed  = time_mod.time() - t0
    print(f"\n완료: {elapsed:.1f}초  유효 조합: {len(is_df)}개\n")

    if is_df.empty:
        print("IS 유효 조합 없음 — MIN_TRADES 기준 완화 필요")
        exit(1)

    print("=== IS 2024 상위 10개 (expectancy 기준) ===")
    print(is_df.head(10).to_string(index=False))
    is_df.to_csv("is_results.csv", index=False)

    if len(df_oos) == 0:
        print("OOS 데이터 없음 — 2025 다운로드 확인 필요")
        exit(1)

    # OOS 검증
    print("\n" + "=" * 50)
    print("[OOS 2025] 상위 10개 검증...")
    print("=" * 50)
    oos_df = validate_oos(df_oos, is_df, n=10)

    print("\n=== OOS 2025 결과 ===")
    print(oos_df.to_string(index=False))
    oos_df.to_csv("oos_results.csv", index=False)

    print("\n결과 저장: is_results.csv / oos_results.csv")

    # 최종 추천
    survivors = oos_df[oos_df["OOS_exp"] > 0]
    if len(survivors) > 0:
        best = survivors.iloc[0]
        print(f"\n★ OOS 양수 조합 {len(survivors)}개")
        print(f"  최고: big_mult={best['big_mult']} cover={best['cover_pct']}"
              f" rr={best['rr_ratio']} avg={best['avg_len']}"
              f" → OOS exp={best['OOS_exp']}R / WR={best['OOS_wr']}%")
    else:
        print("\n✗ OOS 양수 조합 없음 — 전략 재검토 필요")
