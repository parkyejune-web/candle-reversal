"""
장대봉 + 반대봉 역추세 전략 — 멀티 타임프레임 파라미터 최적화 (고속 버전)
타임프레임: 15분 / 30분 / 1시간
IS: 2020~2022  OOS: 2023~2026-03
수수료: 진입·손절=시장가(파라미터), 익절=지정가(0%)
최적화:
  - numpy cumsum rolling (pandas rolling 대체)
  - 신호 계산 143,397 → 3,591회 (cover_ratio/is_big 사전 계산 후 재사용)
  - ProcessPoolExecutor 멀티프로세싱 (avg_len 단위 병렬)
실행: python backtest_params.py
"""

import io
import os
import time as time_mod
import warnings
import zipfile
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from numba import njit

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
SYMBOL = "BTCUSDT"
MIN_TRADES = 20

# ── 파라미터 그리드 ───────────────────────────────────────────────────
BIG_MULT_LIST  = [round(x, 1) for x in np.arange(1.2, 3.01, 0.1)]   # 19개
COVER_PCT_LIST = [round(x, 1) for x in np.arange(0.1, 0.91, 0.1)]   # 9개
RR_RATIO_LIST  = [round(x, 1) for x in np.arange(1.0, 5.01, 0.1)]   # 41개
AVG_LEN_LIST   = list(range(10, 31))                                   # 21개
TAKER_FEE_LIST = [0.0001, 0.00015, 0.0002]                            # 3개
# 총 조합: 19×9×41×21×3 = 443,709개

IS_START  = pd.Timestamp("2020-01-01", tz="UTC")
IS_END    = pd.Timestamp("2023-01-01", tz="UTC")
OOS_START = pd.Timestamp("2023-01-01", tz="UTC")
OOS_END   = pd.Timestamp("2026-04-01", tz="UTC")

TIMEFRAMES   = ["15m", "30m", "1h"]
RESAMPLE_MAP = {"15m": "15min", "30m": "30min", "1h": "1h"}

_BASE = "https://data.binance.vision/data/spot/monthly/klines"


# ── 데이터 다운로드 ───────────────────────────────────────────────────
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


def download_1m(symbol):
    cache = CACHE_DIR / f"{symbol}_1m_2020_2026.parquet"
    if cache.exists():
        print(f"  캐시 로드: {cache}", flush=True)
        return pd.read_parquet(cache)

    cutoff = pd.Timestamp("2026-04-01")
    months = [(y, m) for y in range(2020, 2027)
              for m in range(1, 13)
              if pd.Timestamp(year=y, month=m, day=1) < cutoff]

    print(f"  Binance 다운로드: {symbol} 1분봉 ({len(months)}개월)", flush=True)
    frames = []
    for i, (y, m) in enumerate(months, 1):
        print(f"    [{i}/{len(months)}] {y}-{m:02d}...", end=" ", flush=True)
        try:
            df_m = _fetch_month(symbol, y, m)
            frames.append(df_m)
            print(f"{len(df_m):,}봉", flush=True)
        except Exception as e:
            print(f"건너뜀 ({e})", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_parquet(cache)
    print(f"  저장완료: {cache} ({len(df):,}봉)", flush=True)
    return df


def resample_ohlcv(df_1m, tf):
    rule = RESAMPLE_MAP[tf]
    df = df_1m.set_index("ts")
    agg = df.resample(rule, label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum")
    ).dropna(subset=["open"]).reset_index()
    return agg


# ── numpy rolling mean (pandas rolling 대체, 동일 결과) ──────────────
def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    if len(arr) < window:
        return result
    cs = np.cumsum(arr)
    result[window - 1] = cs[window - 1] / window
    if len(arr) > window:
        result[window:] = (cs[window:] - cs[:len(arr) - window]) / window
    return result


# ── Numba 백테스트 루프 ────────────────────────────────────────────────
@njit(cache=True)
def _backtest_numba(high, low, open_, close,
                    short_sig, long_sig,
                    rr_ratio: float, taker_fee: float,
                    max_bars: int = 1440):
    n = len(high)
    results = np.empty(n, dtype=np.float64)
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

        tp_price = (entry_price - sl_dist * rr_ratio if side == -1
                    else entry_price + sl_dist * rr_ratio)

        lev        = entry_price / sl_dist
        fee_entry  = taker_fee * lev
        fee_sl     = taker_fee * lev

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
                r = rr_ratio - fee_entry   # 익절 = 지정가(수수료 0)
            else:
                continue

            results[cnt] = r
            cnt += 1
            exit_idx = j
            found = True
            break

        if not found:
            j = min(i + max_bars, n - 1)
            r_raw = ((entry_price - close[j]) / sl_dist if side == -1
                     else (close[j] - entry_price) / sl_dist)
            results[cnt] = r_raw - fee_entry - fee_sl
            cnt += 1
            exit_idx = j

    return results[:cnt]


@njit(cache=True)
def _backtest_monthly_numba(high, low, open_, close, ts_ym,
                             short_sig, long_sig,
                             rr_ratio: float, taker_fee: float,
                             max_bars: int = 1440):
    n = len(high)
    r_arr  = np.empty(n, dtype=np.float64)
    ym_arr = np.empty(n, dtype=np.int64)
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

        tp_price = (entry_price - sl_dist * rr_ratio if side == -1
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

            r_arr[cnt]  = r
            ym_arr[cnt] = ts_ym[i + 1]
            cnt += 1
            exit_idx = j
            found = True
            break

        if not found:
            j = min(i + max_bars, n - 1)
            r_raw = ((entry_price - close[j]) / sl_dist if side == -1
                     else (close[j] - entry_price) / sl_dist)
            r_arr[cnt]  = r_raw - fee_entry - fee_sl
            ym_arr[cnt] = ts_ym[i + 1]
            cnt += 1
            exit_idx = j

    return r_arr[:cnt], ym_arr[:cnt]


# ── 멀티프로세싱 워커 (avg_len 단위) ──────────────────────────────────
def _worker(args):
    al, high, low, open_, close, min_trades = args

    n    = len(close)
    body = np.abs(close - open_)

    # avg_body: numpy rolling (pandas 대체)
    avg_body = _rolling_mean(body, al)

    # 파라미터 독립 배열 사전 계산
    is_bull = close > open_
    is_bear = close < open_
    prev_top   = np.maximum(open_[:-1], close[:-1])
    prev_bot   = np.minimum(open_[:-1], close[:-1])
    curr_top   = np.maximum(open_[1:],  close[1:])
    curr_bot   = np.minimum(open_[1:],  close[1:])
    prev_body  = body[:-1]
    overlap    = np.minimum(prev_top, curr_top) - np.maximum(prev_bot, curr_bot)
    cover_ratio = np.where(prev_body > 0, overlap / prev_body, 0.0)

    records = []
    for bm in BIG_MULT_LIST:
        is_big = body >= avg_body * bm
        # short/long 베이스 (cover_pct 무관)
        short_base = is_big[:-1] & is_bull[:-1] & is_bear[1:]
        long_base  = is_big[:-1] & is_bear[:-1] & is_bull[1:]

        for cp in COVER_PCT_LIST:
            cover_mask = cover_ratio >= cp
            short_sig  = np.zeros(n, dtype=np.bool_)
            long_sig   = np.zeros(n, dtype=np.bool_)
            short_sig[1:] = short_base & cover_mask
            long_sig[1:]  = long_base  & cover_mask

            for rr in RR_RATIO_LIST:
                for fee in TAKER_FEE_LIST:
                    arr = _backtest_numba(high, low, open_, close,
                                         short_sig, long_sig, rr, fee)
                    if len(arr) < min_trades:
                        continue
                    wins = (arr > 0).sum()
                    cum  = np.cumsum(arr)
                    peak = np.maximum.accumulate(cum)
                    mdd  = float((peak - cum).max())
                    records.append({
                        "장대봉배수": bm,
                        "덮음비율":   cp,
                        "손익비":     rr,
                        "평균기간":   al,
                        "수수료":     fee,
                        "거래수":     len(arr),
                        "승률":       round(wins / len(arr) * 100, 1),
                        "기대치":     round(float(arr.mean()), 4),
                        "총수익R":    round(float(arr.sum()), 2),
                        "MDD_R":      round(mdd, 2),
                    })
    return records


# ── 그리드 서치 (멀티프로세싱) ───────────────────────────────────────
def grid_search(df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    high  = df["high"].values
    low   = df["low"].values
    open_ = df["open"].values
    close = df["close"].values

    total_combos = len(BIG_MULT_LIST)*len(COVER_PCT_LIST)*len(RR_RATIO_LIST)*len(TAKER_FEE_LIST)
    print(f"\n  [{tf_label}] IS 그리드 서치: "
          f"avg_len {len(AVG_LEN_LIST)}개 × {total_combos:,}조합 = "
          f"{len(AVG_LEN_LIST)*total_combos:,}건", flush=True)

    args_list = [(al, high, low, open_, close, MIN_TRADES) for al in AVG_LEN_LIST]
    workers   = min(os.cpu_count() or 4, len(AVG_LEN_LIST))
    records   = []
    t0        = time_mod.time()

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_worker, a): a[0] for a in args_list}
        done = 0
        for fut in as_completed(futures):
            al = futures[fut]
            done += 1
            try:
                res = fut.result()
                records.extend(res)
            except Exception as e:
                print(f"  [경고] avg_len={al} 오류: {e}", flush=True)
            elapsed = time_mod.time() - t0
            eta     = elapsed / done * (len(AVG_LEN_LIST) - done) if done < len(AVG_LEN_LIST) else 0
            print(f"  [{tf_label}] avg_len={al} 완료 ({done}/{len(AVG_LEN_LIST)})"
                  f"  경과 {elapsed:.0f}초  남은시간 {eta:.0f}초", flush=True)

    elapsed = time_mod.time() - t0
    print(f"  [{tf_label}] 완료: {elapsed:.1f}초  유효조합: {len(records):,}개", flush=True)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("기대치", ascending=False).reset_index(drop=True)


# ── 월별 성과 집계 ────────────────────────────────────────────────────
def monthly_stats(df: pd.DataFrame, short_sig, long_sig,
                  rr_ratio: float, taker_fee: float) -> pd.DataFrame:
    ts_ym = (df["ts"].dt.year * 100 + df["ts"].dt.month).values.astype(np.int64)
    r_arr, ym_arr = _backtest_monthly_numba(
        df["high"].values, df["low"].values,
        df["open"].values, df["close"].values,
        ts_ym, short_sig, long_sig, rr_ratio, taker_fee
    )
    if len(r_arr) == 0:
        return pd.DataFrame()

    rows = []
    for ym in sorted(set(ym_arr)):
        mask = ym_arr == ym
        arr  = r_arr[mask]
        wins = (arr > 0).sum()
        rows.append({
            "연월":    str(ym),
            "거래수":  len(arr),
            "승률%":   round(wins / len(arr) * 100, 1),
            "월수익R": round(float(arr.sum()), 3),
        })
    monthly = pd.DataFrame(rows)
    cum, cum_list = 0.0, []
    for v in monthly["월수익R"]:
        cum += v
        cum_list.append(round(cum, 3))
    monthly["누적수익R"] = cum_list
    return monthly


# ── 신호 감지 (원본 로직 그대로, 월별 분석용) ────────────────────────
def _make_signals(df: pd.DataFrame, bm, cp, al):
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)

    body     = np.abs(close - open_)
    avg_body = _rolling_mean(body, int(al))

    is_big  = body >= avg_body * bm
    is_bull = close > open_
    is_bear = close < open_

    prev_top  = np.maximum(open_[:-1], close[:-1])
    prev_bot  = np.minimum(open_[:-1], close[:-1])
    curr_top  = np.maximum(open_[1:],  close[1:])
    curr_bot  = np.minimum(open_[1:],  close[1:])
    prev_body = body[:-1]

    overlap = np.minimum(prev_top, curr_top) - np.maximum(prev_bot, curr_bot)
    cover   = np.where(prev_body > 0, overlap / prev_body, 0.0)

    short_sig = np.zeros(n, dtype=np.bool_)
    long_sig  = np.zeros(n, dtype=np.bool_)
    short_sig[1:] = is_big[:-1] & is_bull[:-1] & is_bear[1:] & (cover >= cp)
    long_sig[1:]  = is_big[:-1] & is_bear[:-1] & is_bull[1:] & (cover >= cp)
    return short_sig, long_sig


# ── 상위 파라미터 IS+OOS 월별 분석 ────────────────────────────────────
def analyze_top(df_is, df_oos, is_df, tf_label, top_n=3):
    print(f"\n  [{tf_label}] 상위 {top_n}개 파라미터 IS+OOS 월별 분석", flush=True)
    all_rows = []
    for rank, (_, row) in enumerate(is_df.head(top_n).iterrows(), 1):
        bm, cp, rr = row["장대봉배수"], row["덮음비율"], row["손익비"]
        al, fee    = int(row["평균기간"]), row["수수료"]
        label = (f"{rank}등 [배수={bm} 덮음={cp} 손익비={rr}"
                 f" 평균{al}봉 수수료{fee*100:.3f}%]")
        print(f"\n    ─ {label}", flush=True)
        print(f"      IS 기대치: {row['기대치']}R  승률: {row['승률']}%  거래수: {row['거래수']}", flush=True)

        ss_is, ls_is   = _make_signals(df_is, bm, cp, al)
        ss_oos, ls_oos = _make_signals(df_oos, bm, cp, al)

        mon_is  = monthly_stats(df_is,  ss_is,  ls_is,  rr, fee)
        mon_oos = monthly_stats(df_oos, ss_oos, ls_oos, rr, fee)

        arr_oos = _backtest_numba(
            df_oos["high"].values, df_oos["low"].values,
            df_oos["open"].values, df_oos["close"].values,
            ss_oos, ls_oos, rr, fee)
        if len(arr_oos) > 0:
            oos_wr  = round((arr_oos > 0).sum() / len(arr_oos) * 100, 1)
            oos_exp = round(float(arr_oos.mean()), 4)
            oos_tot = round(float(arr_oos.sum()), 2)
        else:
            oos_wr = oos_exp = oos_tot = 0

        print(f"      OOS 기대치: {oos_exp}R  승률: {oos_wr}%"
              f"  거래수: {len(arr_oos)}  총수익R: {oos_tot}", flush=True)

        if not mon_is.empty:
            print(f"\n      [IS 월별 성적]", flush=True)
            print(mon_is.to_string(index=False), flush=True)
        if not mon_oos.empty:
            print(f"\n      [OOS 월별 성적]", flush=True)
            print(mon_oos.to_string(index=False), flush=True)

        for _, mr in mon_is.iterrows():
            all_rows.append({"타임프레임": tf_label, "구간": "IS",
                              "순위": rank, "파라미터": label, **mr.to_dict()})
        for _, mr in mon_oos.iterrows():
            all_rows.append({"타임프레임": tf_label, "구간": "OOS",
                              "순위": rank, "파라미터": label, **mr.to_dict()})
    return all_rows


# ── 메인 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65, flush=True)
    print("장대봉 역추세 전략 파라미터 최적화 (고속 버전)", flush=True)
    print(f"IS: 2020~2022  OOS: 2023~2026-03", flush=True)
    total = len(BIG_MULT_LIST)*len(COVER_PCT_LIST)*len(RR_RATIO_LIST)*len(AVG_LEN_LIST)*len(TAKER_FEE_LIST)
    print(f"파라미터 총 조합: {total:,}개 × 3 타임프레임", flush=True)
    print(f"CPU: {os.cpu_count()}코어 활용 (avg_len 단위 병렬)", flush=True)
    print("=" * 65, flush=True)

    # ── Numba 워밍업 ──────────────────────────────────────────────────
    print("\nNumba 컴파일 중 (최초 1회)...", flush=True)
    _h = np.array([1.0, 1.0, 1.0]); _l = np.array([0.9, 0.9, 0.9])
    _o = np.array([0.95, 0.95, 0.95]); _c = np.array([1.0, 0.92, 1.0])
    _ss = np.array([False, True, False]); _ls = np.array([False, False, False])
    _ym = np.array([202001, 202001, 202001], dtype=np.int64)
    _backtest_numba(_h, _l, _o, _c, _ss, _ls, 2.0, 0.001)
    _backtest_monthly_numba(_h, _l, _o, _c, _ym, _ss, _ls, 2.0, 0.001)
    print("컴파일 완료\n", flush=True)

    # ── 1분봉 로드 ────────────────────────────────────────────────────
    print("1분봉 데이터 로드...", flush=True)
    df_1m = download_1m(SYMBOL)
    print(f"  총 {len(df_1m):,}봉\n", flush=True)

    all_monthly_rows = []

    for tf in TIMEFRAMES:
        print("=" * 65, flush=True)
        print(f"[{tf} 타임프레임]", flush=True)
        print("=" * 65, flush=True)

        df_tf  = resample_ohlcv(df_1m, tf)
        df_is  = df_tf[(df_tf["ts"] >= IS_START) & (df_tf["ts"] < IS_END)].reset_index(drop=True)
        df_oos = df_tf[(df_tf["ts"] >= OOS_START) & (df_tf["ts"] < OOS_END)].reset_index(drop=True)
        print(f"  IS  (2020~2022): {len(df_is):,}봉", flush=True)
        print(f"  OOS (2023~2026-03): {len(df_oos):,}봉", flush=True)

        if len(df_is) == 0 or len(df_oos) == 0:
            print("  데이터 부족, 건너뜀", flush=True)
            continue

        # IS 그리드 서치
        is_df = grid_search(df_is, tf)
        if is_df.empty:
            print(f"  [{tf}] IS 유효 조합 없음", flush=True)
            continue

        print(f"\n  [{tf}] IS 기대치 상위 10개", flush=True)
        print(is_df.head(10).to_string(index=False), flush=True)
        is_df.to_csv(f"is_결과_{tf}.csv", index=False, encoding="utf-8-sig")

        # 상위 3개 월별 분석
        rows = analyze_top(df_is, df_oos, is_df, tf, top_n=3)
        all_monthly_rows.extend(rows)

    # ── 전체 저장 ─────────────────────────────────────────────────────
    if all_monthly_rows:
        df_mon = pd.DataFrame(all_monthly_rows)
        df_mon.to_csv("월별_성적_전체.csv", index=False, encoding="utf-8-sig")

    print("\n\n결과 파일:", flush=True)
    for tf in TIMEFRAMES:
        print(f"  is_결과_{tf}.csv", flush=True)
    print("  월별_성적_전체.csv", flush=True)
    print("\n완료", flush=True)
