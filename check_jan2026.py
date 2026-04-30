"""
Jan 2026 30분봉 신호+거래 상세 출력
Pine vs Python 비교용
"""
import io, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
SYMBOL = "BTCUSDT"
BIG_MULT  = 2.4
COVER_PCT = 0.7
RR_RATIO  = 4.3
AVG_LEN   = 13
MAX_BARS  = 1440

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

def load_1m():
    cache = CACHE_DIR / f"{SYMBOL}_1m_2020_2026.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raise FileNotFoundError("캐시 없음. backtest_risk.py 먼저 실행하세요.")

def resample(df_1m, rule="30min"):
    df = df_1m.set_index("ts")
    return df.resample(rule, label="left", closed="left").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last")
    ).dropna(subset=["open"]).reset_index()

def _rolling_mean(arr, window):
    result = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    result[window - 1] = cs[window - 1] / window
    if len(arr) > window:
        result[window:] = (cs[window:] - cs[:len(arr) - window]) / window
    return result

def detect_signals(df):
    close = df["close"].values
    open_ = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    n     = len(df)
    body  = np.abs(close - open_)
    avg_body = _rolling_mean(body, AVG_LEN)
    is_big  = body >= avg_body * BIG_MULT
    is_bull = close > open_
    is_bear = close < open_
    prev_top = np.maximum(open_[:-1], close[:-1])
    prev_bot = np.minimum(open_[:-1], close[:-1])
    curr_top = np.maximum(open_[1:],  close[1:])
    curr_bot = np.minimum(open_[1:],  close[1:])
    prev_body = body[:-1]
    overlap  = np.minimum(prev_top, curr_top) - np.maximum(prev_bot, curr_bot)
    cover    = np.where(prev_body > 0, overlap / prev_body, 0.0)
    short_sig = np.zeros(n, dtype=bool)
    long_sig  = np.zeros(n, dtype=bool)
    short_sig[1:] = is_big[:-1] & is_bull[:-1] & is_bear[1:] & (cover >= COVER_PCT)
    long_sig[1:]  = is_big[:-1] & is_bear[:-1] & is_bull[1:] & (cover >= COVER_PCT)
    return short_sig, long_sig, body, avg_body, cover

if __name__ == "__main__":
    print("캐시 로드 중...", flush=True)
    df_1m = load_1m()

    # OOS 전체 (2023-01 ~ 2026-04) 30분봉
    oos_start = pd.Timestamp("2023-01-01", tz="UTC")
    oos_end   = pd.Timestamp("2026-04-01", tz="UTC")
    df_1m_oos = df_1m[(df_1m["ts"] >= oos_start) & (df_1m["ts"] < oos_end)]
    df30 = resample(df_1m_oos)
    df30 = df30.reset_index(drop=True)

    close = df30["close"].values
    open_ = df30["open"].values
    high  = df30["high"].values
    low   = df30["low"].values
    ts    = df30["ts"].values

    short_sig, long_sig, body, avg_body, cover = detect_signals(df30)

    n = len(df30)
    exit_idx = -1

    jan_start = pd.Timestamp("2026-01-01")
    jan_end   = pd.Timestamp("2026-02-01")

    print("\n===== 2026년 1월 신호 (신호봉) =====")
    for i in range(n):
        t = pd.Timestamp(ts[i]).tz_localize(None)
        if t < jan_start or t >= jan_end:
            continue
        if short_sig[i]:
            print(f"  SHORT 신호 @ {t}  close={close[i]:.1f}  high={high[i]:.1f}  body={body[i]:.1f}  avg_body={avg_body[i-1]:.1f}  cover={cover[i-1]:.3f}")
        if long_sig[i]:
            print(f"  LONG  신호 @ {t}  close={close[i]:.1f}  low={low[i]:.1f}  body={body[i]:.1f}  avg_body={avg_body[i-1]:.1f}  cover={cover[i-1]:.3f}")

    print("\n===== 2026년 1월 거래 (진입/청산 포함) =====")
    exit_idx = -1
    for i in range(n - 1):
        if i <= exit_idx:
            continue
        if not (short_sig[i] or long_sig[i]):
            continue
        t_sig = pd.Timestamp(ts[i]).tz_localize(None)
        t_entry = pd.Timestamp(ts[i + 1]).tz_localize(None)

        # 1월 신호만
        if t_sig < jan_start or t_sig >= jan_end:
            continue

        if short_sig[i]:
            sl_price    = high[i]
            entry_price = open_[i + 1]
            sl_dist     = sl_price - entry_price
            side        = -1
            side_str    = "SHORT"
        else:
            sl_price    = low[i]
            entry_price = open_[i + 1]
            sl_dist     = entry_price - sl_price
            side        = 1
            side_str    = "LONG"

        if sl_dist <= 0.0:
            print(f"  [{side_str}] 신호 {t_sig} → sl_dist<=0 스킵 (entry={entry_price:.1f} sl={sl_price:.1f})")
            continue

        tp_price = (entry_price - sl_dist * RR_RATIO if side == -1
                    else entry_price + sl_dist * RR_RATIO)

        print(f"\n  [{side_str}] 신호:{t_sig}  진입:{t_entry}  entry={entry_price:.2f}  sl={sl_price:.2f}  tp={tp_price:.2f}  sl_dist={sl_dist:.2f}")

        found = False
        j_end = min(i + MAX_BARS + 1, n)
        for j in range(i + 1, j_end):
            if side == -1:
                hit_sl = high[j] >= sl_price
                hit_tp = low[j]  <= tp_price
            else:
                hit_sl = low[j]  <= sl_price
                hit_tp = high[j] >= tp_price

            if hit_sl or hit_tp:
                t_exit = pd.Timestamp(ts[j]).tz_localize(None)
                result = "TP" if (hit_tp and not hit_sl) else "SL" if (hit_sl and not hit_tp) else "SL(둘다)"
                pnl = RR_RATIO if (hit_tp and not hit_sl) else -1.0
                print(f"         청산:{t_exit}  결과={result}  R={pnl:.1f}  bars={j-i}")
                exit_idx = j
                found = True
                break

        if not found:
            j = min(i + MAX_BARS, n - 1)
            t_exit = pd.Timestamp(ts[j]).tz_localize(None)
            r_raw = ((entry_price - close[j]) / sl_dist if side == -1
                     else (close[j] - entry_price) / sl_dist)
            print(f"         청산:{t_exit}  결과=TIMEOUT  R={r_raw:.2f}  bars={j-i}")
            exit_idx = j

    print("\n완료")
