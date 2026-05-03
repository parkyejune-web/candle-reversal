"""
Candle-Reversal Live Trader — Gate.io USDT-PERP
전략: BTC_USDT 1m  장대봉+반대봉 역추세
params: big_mult=1.8, cover=0.3, rr=3.5, avg_len=10
SL: 신호봉 고가(short) / 저가(long)
TP: entry ± sl_dist × 3.5
포지션: 고정 $100 / 레버리지 1x
"""
import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gate_api
from gate_api import ApiClient, Configuration
from gate_api.api import FuturesApi
from gate_api.models import (
    FuturesOrder,
    FuturesPriceTriggeredOrder,
    FuturesPriceTrigger,
    FuturesInitialOrder,
)
from dotenv import load_dotenv

import telegram_bot as tg

load_dotenv()

# ── 전략 파라미터 ──────────────────────────────────────────────────────
BIG_MULT     = 1.8
COVER_PCT    = 0.3
RR_RATIO     = 3.5
AVG_LEN      = 10
RISK_USDT    = 100.0          # 고정 베팅 $100
LEVERAGE     = 1              # 고정 1배율

CONTRACT     = "BTC_USDT"
SETTLE       = "usdt"
QUANTO       = 0.0001         # 1 contract = 0.0001 BTC (Gate.io BTC_USDT)
INTERVAL     = "1m"
KLINE_LIMIT  = 60
POLL_SEC     = 30
MAX_HOLD_SEC = 86_400         # 24시간 최대 보유

DEMO         = os.environ.get("DEMO", "true").lower() != "false"
GATE_HOST    = os.environ.get("GATE_HOST", "https://api.gateio.ws/api/v4")

# ── 로깅 ──────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))
logging.Formatter.converter = lambda *args: datetime.now(KST).timetuple()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s KST [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CandleReversal")


# ── Gate.io 클라이언트 ────────────────────────────────────────────────
def make_api() -> FuturesApi:
    cfg = Configuration(
        host=GATE_HOST,
        key=os.environ["GATE_API_KEY"],
        secret=os.environ["GATE_API_SECRET"],
    )
    return FuturesApi(ApiClient(cfg))


# ── 캔들 데이터 ───────────────────────────────────────────────────────
def fetch_klines(api: FuturesApi) -> pd.DataFrame:
    """완성된 1m 봉 (마지막 진행중 봉 제외)."""
    candles = api.list_futures_candlesticks(
        settle=SETTLE, contract=CONTRACT,
        interval=INTERVAL, limit=KLINE_LIMIT + 1,
    )
    rows = [{"ts": c.t, "open": float(c.o or 0), "high": float(c.h or 0),
              "low": float(c.l or 0), "close": float(c.c or 0)}
            for c in candles]
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)


# ── 신호 감지 ─────────────────────────────────────────────────────────
def detect_signal(df: pd.DataFrame) -> Optional[str]:
    """마지막 완성 봉 쌍 기준 신호 반환 ('long'/'short'/None)."""
    if len(df) < AVG_LEN + 2:
        return None

    close = df["close"].values
    open_ = df["open"].values
    body  = np.abs(close - open_)
    avg_body = pd.Series(body).rolling(AVG_LEN, min_periods=AVG_LEN).mean().values

    i = len(df) - 1   # 신호봉 (반대봉)
    j = i - 1         # 장대봉

    if np.isnan(avg_body[j]) or body[j] < avg_body[j] * BIG_MULT:
        return None

    # 커버(겹침) 계산
    prev_top = max(open_[j], close[j])
    prev_bot = min(open_[j], close[j])
    curr_top = max(open_[i], close[i])
    curr_bot = min(open_[i], close[i])
    overlap  = min(prev_top, curr_top) - max(prev_bot, curr_bot)
    cover    = overlap / body[j] if body[j] > 0 else 0.0

    if cover < COVER_PCT:
        return None

    if close[j] > open_[j] and close[i] < open_[i]:
        return "short"
    if close[j] < open_[j] and close[i] > open_[i]:
        return "long"
    return None


# ── 차트 생성 ─────────────────────────────────────────────────────────
def _draw_candles(ax, sub_df: pd.DataFrame) -> None:
    for idx in range(len(sub_df)):
        row = sub_df.iloc[idx]
        up  = row["close"] >= row["open"]
        col = "#26a69a" if up else "#ef5350"
        ax.plot([idx, idx], [row["low"], row["high"]], color=col, linewidth=0.6, zorder=1)
        bot = min(row["open"], row["close"])
        ht  = max(abs(row["close"] - row["open"]), row["close"] * 0.00004)
        ax.add_patch(plt.Rectangle(
            (idx - 0.35, bot), 0.7, ht,
            facecolor=col, edgecolor=col, linewidth=0.2, zorder=2
        ))


def make_trade_chart(df: pd.DataFrame, side: str,
                     entry: float, sl: float, tp: float,
                     entry_ts_str: str) -> bytes:
    """진입 직후 1m 차트 (PNG bytes 반환)."""
    n   = min(60, len(df))
    sub = df.iloc[-n:].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    _draw_candles(ax, sub)

    ax.axhline(entry, color="#42a5f5", linewidth=1.0, linestyle="--", alpha=0.95, label=f"Entry {entry:,.1f}", zorder=3)
    ax.axhline(tp,    color="#66bb6a", linewidth=1.0, linestyle="--", alpha=0.95, label=f"TP    {tp:,.1f}", zorder=3)
    ax.axhline(sl,    color="#ef5350", linewidth=1.0, linestyle="--", alpha=0.95, label=f"SL    {sl:,.1f}", zorder=3)
    ax.axvline(len(sub) - 1, color="#42a5f5", linewidth=0.8, alpha=0.6, zorder=4)

    sym   = "▲ LONG" if side == "long" else "▼ SHORT"
    color = "#66bb6a" if side == "long" else "#ef5350"
    ax.set_title(
        f"{sym}  BTC_USDT 1m  {entry_ts_str}  |  RR {RR_RATIO}x",
        fontsize=9, color=color, pad=4, fontweight="bold"
    )

    yvals = [sl, tp, sub["low"].min(), sub["high"].max()]
    ymin, ymax = min(yvals), max(yvals)
    pad = (ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_xticks([])
    ax.legend(loc="upper left", fontsize=7, facecolor="#1a1a2e",
              edgecolor="#333", labelcolor="white", framealpha=0.85)
    for sp in ax.spines.values():
        sp.set_color("#2a2a2a"); sp.set_linewidth(0.4)
    ax.tick_params(colors="#888", labelsize=7)
    ax.yaxis.set_tick_params(labelcolor="#aaa")

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_exit_chart(df: pd.DataFrame, pos, status: str) -> bytes:
    """청산 시 전체 거래 구간 차트 (진입→청산)."""
    entry_ts_unix = int(pos.entry_ts.timestamp())
    ts_vals  = df["ts"].values
    entry_idx = int(np.searchsorted(ts_vals, entry_ts_unix))
    entry_idx = min(entry_idx, len(df) - 1)
    start = max(0, entry_idx - 8)
    sub   = df.iloc[start:].reset_index(drop=True)

    rel_e = entry_idx - start
    rel_x = len(sub) - 1

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    _draw_candles(ax, sub)

    ax.axhline(pos.entry_price, color="#42a5f5", lw=1.0, ls="--", alpha=0.9, label=f"Entry {pos.entry_price:,.1f}")
    if pos.tp:
        ax.axhline(pos.tp, color="#66bb6a", lw=1.0, ls="--", alpha=0.9, label=f"TP {pos.tp:,.1f}")
    if pos.sl:
        ax.axhline(pos.sl, color="#ef5350", lw=1.0, ls="--", alpha=0.9, label=f"SL {pos.sl:,.1f}")
    ax.axvline(rel_e, color="#42a5f5", lw=0.8, alpha=0.6)
    exit_col = {"WIN": "#66bb6a", "LOSS": "#ef5350", "EVEN": "#9e9e9e"}.get(status, "#9e9e9e")
    ax.axvline(rel_x, color=exit_col, lw=1.2, alpha=0.9)

    sym = "▲ LONG" if pos.side == "long" else "▼ SHORT"
    ax.set_title(
        f"{sym}  BTC_USDT 1m  |  {status}",
        fontsize=9, color=exit_col, pad=4, fontweight="bold"
    )
    yvals = [pos.entry_price, sub["low"].min(), sub["high"].max()]
    if pos.sl: yvals.append(pos.sl)
    if pos.tp: yvals.append(pos.tp)
    pad = (max(yvals) - min(yvals)) * 0.08
    ax.set_ylim(min(yvals) - pad, max(yvals) + pad)
    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_xticks([])
    ax.legend(loc="upper left", fontsize=7, facecolor="#1a1a2e",
              edgecolor="#333", labelcolor="white", framealpha=0.85)
    for sp in ax.spines.values():
        sp.set_color("#2a2a2a"); sp.set_linewidth(0.4)
    ax.tick_params(colors="#888", labelsize=7)
    ax.yaxis.set_tick_params(labelcolor="#aaa")

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── 유틸 ──────────────────────────────────────────────────────────────
def get_balance(api: FuturesApi) -> float:
    return float(api.list_futures_accounts(settle=SETTLE).total)


def get_position_size(api: FuturesApi) -> int:
    """현재 포지션 크기 (양수=롱, 음수=숏, 0=없음)."""
    try:
        pos = api.get_position(settle=SETTLE, contract=CONTRACT)
        return int(pos.size)
    except Exception:
        return 0


def get_last_price(api: FuturesApi) -> float:
    tickers = api.list_futures_tickers(settle=SETTLE, contract=CONTRACT)
    return float(tickers[0].last)


def calc_entry(last_price: float) -> tuple:
    """고정 $100 / 1배율 — 계약수 반환."""
    contracts = max(1, int(RISK_USDT / (QUANTO * last_price)))
    return contracts, LEVERAGE


# ── 포지션 ────────────────────────────────────────────────────────────
class Position:
    def __init__(self, side: str, entry: float, sl: float, tp: float,
                 contracts: int, entry_ts: datetime,
                 sl_id: Optional[int], tp_id: Optional[int]):
        self.side        = side
        self.entry_price = entry
        self.sl          = sl
        self.tp          = tp
        self.contracts   = contracts
        self.entry_ts    = entry_ts
        self.sl_id       = sl_id
        self.tp_id       = tp_id


# ── 트레이더 ──────────────────────────────────────────────────────────
LAST_SIG_FILE = ".last_signal_ts"
STATS_FILE    = ".stats.json"


class CandleReversalTrader:
    def __init__(self):
        self.api      = make_api()
        self.pos: Optional[Position] = None
        self.df: Optional[pd.DataFrame] = None
        self._last_sig_ts: Optional[int] = self._load_sig_ts()
        stats = self._load_stats()
        self._wins         = stats["wins"]
        self._losses       = stats["losses"]
        self._total_profit = stats["total_profit"]
        self._total_loss   = stats["total_loss"]
        self._daily_date: Optional[str] = None
        self._recover_position()

    def _load_stats(self) -> dict:
        """Gate.io 청산 이력에서 승/패 + PnL 집계 — 재배포해도 통계 유지."""
        try:
            closes = self.api.list_position_close(
                settle=SETTLE, contract=CONTRACT, limit=1000)
            wins         = sum(1 for c in closes if float(c.pnl or 0) > 0)
            losses       = sum(1 for c in closes if float(c.pnl or 0) < 0)
            total_profit = sum(float(c.pnl or 0) for c in closes if float(c.pnl or 0) > 0)
            total_loss   = sum(float(c.pnl or 0) for c in closes if float(c.pnl or 0) < 0)
            logger.info(f"통계 복원 (Gate.io): {wins}승 {losses}패  "
                        f"수익 ${total_profit:.2f}  손실 ${total_loss:.2f}")
            return {"wins": wins, "losses": losses,
                    "total_profit": total_profit, "total_loss": total_loss}
        except Exception as e:
            logger.warning(f"통계 API 조회 실패, 파일 폴백: {e}")
        try:
            with open(STATS_FILE) as f:
                d = json.load(f)
                return {
                    "wins":         int(d.get("wins", 0)),
                    "losses":       int(d.get("losses", 0)),
                    "total_profit": float(d.get("total_profit", 0.0)),
                    "total_loss":   float(d.get("total_loss", 0.0)),
                }
        except Exception:
            return {"wins": 0, "losses": 0, "total_profit": 0.0, "total_loss": 0.0}

    def _save_stats(self) -> None:
        try:
            with open(STATS_FILE, "w") as f:
                json.dump({
                    "wins":         self._wins,
                    "losses":       self._losses,
                    "total_profit": self._total_profit,
                    "total_loss":   self._total_loss,
                }, f)
        except Exception as e:
            logger.warning(f"stats 저장 실패: {e}")

    def _load_sig_ts(self) -> Optional[int]:
        try:
            with open(LAST_SIG_FILE) as f:
                v = f.read().strip()
                return int(v) if v else None
        except FileNotFoundError:
            return None

    def _save_sig_ts(self, ts: int) -> None:
        try:
            with open(LAST_SIG_FILE, "w") as f:
                f.write(str(ts))
        except Exception as e:
            logger.warning(f"sig_ts 저장 실패: {e}")

    def _cancel_all_trigger_orders(self) -> None:
        """포지션 없을 때 잔류 트리거 주문 전체 취소."""
        try:
            open_orders = self.api.list_price_triggered_orders(
                settle=SETTLE, status="open", contract=CONTRACT)
            if not open_orders:
                return
            for o in open_orders:
                try:
                    self.api.cancel_price_triggered_order(
                        settle=SETTLE, order_id=str(o.id))
                    logger.info(f"잔류 트리거 주문 취소: {o.id}")
                except Exception as e:
                    logger.debug(f"취소 실패(이미 체결?): {o.id}  {e}")
            logger.warning(f"잔류 트리거 주문 {len(open_orders)}개 취소 완료")
        except Exception as e:
            logger.warning(f"트리거 주문 조회 실패: {e}")

    def _recover_position(self) -> None:
        """재시작 후 오픈 포지션 복원 — 없으면 잔류 트리거 주문 청소."""
        try:
            size = get_position_size(self.api)
        except Exception as e:
            logger.warning(f"포지션 복원 조회 실패: {e}")
            return

        if size == 0:
            self._cancel_all_trigger_orders()
            return

        try:
            pos_data    = self.api.get_position(settle=SETTLE, contract=CONTRACT)
            entry_price = float(pos_data.entry_price or 0)
        except Exception as e:
            logger.warning(f"포지션 상세 조회 실패: {e}")
            return

        side      = "long" if size > 0 else "short"
        contracts = abs(size)

        # 오픈 트리거 주문에서 sl_id / tp_id 찾기
        sl_id = tp_id = None
        try:
            open_orders = self.api.list_price_triggered_orders(
                settle=SETTLE, status="open", contract=CONTRACT)
            for o in open_orders:
                order_price = float(getattr(o.initial, "price", "0") or "0")
                if order_price == 0.0:
                    sl_id = o.id   # market(IOC) → SL
                else:
                    tp_id = o.id   # limit(GTC) → TP
        except Exception as e:
            logger.warning(f"트리거 주문 조회 실패: {e}")

        self.pos = Position(
            side=side, entry=entry_price, sl=0.0, tp=0.0,
            contracts=contracts, entry_ts=datetime.now(timezone.utc),
            sl_id=sl_id, tp_id=tp_id,
        )
        logger.warning(
            f"[복원] {side.upper()} {contracts}계약 @ {entry_price:.1f}  "
            f"sl_id={sl_id} tp_id={tp_id}"
        )
        tg.send_error(
            f"[복원] 재시작 전 포지션 발견\n"
            f"{side.upper()} {contracts}계약 @ ${entry_price:,.1f}\n"
            f"SL_id={sl_id}  TP_id={tp_id}"
        )

    def _stats(self) -> dict:
        t = self._wins + self._losses
        return {
            "wins":         self._wins,
            "losses":       self._losses,
            "winrate":      self._wins / t * 100 if t else 0.0,
            "total_profit": self._total_profit,
            "total_loss":   self._total_loss,
            "cum_pnl":      self._total_profit + self._total_loss,
        }

    # ── 진입 ─────────────────────────────────────────────────────────
    def try_enter(self, df: pd.DataFrame) -> None:
        if self.pos:
            return

        signal = detect_signal(df)
        if not signal:
            return

        # 중복 방지: 같은 신호봉 재진입 차단
        sig_ts = int(df.iloc[-1]["ts"])
        if sig_ts == self._last_sig_ts:
            return

        # Gate 포지션 재확인 (4중 방어)
        try:
            if get_position_size(self.api) != 0:
                logger.info("포지션 있음 — 스킵")
                return
        except Exception as e:
            logger.warning(f"포지션 조회 실패: {e}")
            return

        # 현재가
        try:
            last_price = get_last_price(self.api)
        except Exception as e:
            logger.warning(f"시세 조회 실패: {e}")
            return

        sig_bar  = df.iloc[-1]
        sl_price = float(sig_bar["high"] if signal == "short" else sig_bar["low"])
        sl_dist  = abs(last_price - sl_price)
        if sl_dist <= 0:
            logger.warning("SL 거리 0 — 스킵")
            return

        tp_price  = (last_price - sl_dist * RR_RATIO if signal == "short"
                     else last_price + sl_dist * RR_RATIO)
        contracts, leverage = calc_entry(last_price)

        # 마진 사전 검증 — 1x이므로 필요마진 = 노셔널
        required_margin = contracts * QUANTO * last_price
        try:
            available = get_balance(self.api)
        except Exception as e:
            logger.warning(f"잔고 조회 실패: {e}")
            return
        if available < required_margin * 1.1:
            logger.warning(
                f"마진 부족 — 진입 취소 "
                f"(필요 ${required_margin:.1f} × 1.1 = ${required_margin*1.1:.1f}, "
                f"가용 ${available:.1f})"
            )
            return

        # 레버리지 1x 설정
        try:
            self.api.update_position_leverage(
                settle=SETTLE, contract=CONTRACT, leverage=str(leverage))
        except Exception as e:
            logger.warning(f"레버리지 설정 실패: {e}")

        logger.info(
            f"신호: {signal.upper()} | last≈{last_price:.1f} "
            f"sl={sl_price:.1f} tp={tp_price:.1f} | "
            f"contracts={contracts} lev=1x bet=$100 margin≈${required_margin:.1f}"
        )

        self._last_sig_ts = sig_ts
        self._save_sig_ts(sig_ts)

        # 시장가 진입 (양수=롱, 음수=숏)
        entry_size = -contracts if signal == "short" else contracts
        try:
            self.api.create_futures_order(
                settle=SETTLE,
                futures_order=FuturesOrder(
                    contract=CONTRACT, size=entry_size,
                    price="0", tif="ioc",
                    text="t-cr-entry",
                )
            )
        except Exception as e:
            body = getattr(e, 'body', None) or ''
            status = getattr(e, 'status', '?')
            logger.error(f"주문 실패({status}): {body or e}")
            tg.send_error(f"주문실패({status})\n{str(body)[:250] or str(e)[:250]}")
            return

        # 체결 확인 (3초 대기)
        time.sleep(3)
        try:
            actual_size = get_position_size(self.api)
            if actual_size == 0:
                logger.warning("체결 미확인 — 포기")
                return
        except Exception as e:
            logger.warning(f"체결 확인 실패: {e}")
            return

        try:
            entry_price = get_last_price(self.api)
        except Exception:
            entry_price = last_price

        # SL/TP 재계산 (실제 진입가 기준)
        sl_dist_actual = abs(entry_price - sl_price)
        if sl_dist_actual > 0:
            tp_price = (entry_price - sl_dist_actual * RR_RATIO if signal == "short"
                        else entry_price + sl_dist_actual * RR_RATIO)

        actual_contracts = abs(actual_size)
        sl_id, tp_id = self._place_sltp(signal, sl_price, tp_price, actual_contracts)

        self.pos = Position(
            side=signal, entry=entry_price, sl=sl_price, tp=tp_price,
            contracts=actual_contracts, entry_ts=datetime.now(timezone.utc),
            sl_id=sl_id, tp_id=tp_id,
        )
        try:
            balance = get_balance(self.api)
        except Exception:
            balance = 0.0
        sl_dist_actual = abs(entry_price - sl_price)
        risk_usdt = actual_contracts * QUANTO * sl_dist_actual

        entry_ts_str = datetime.now(KST).strftime("%m/%d %H:%M")
        img_bytes = None
        try:
            img_bytes = make_trade_chart(df, signal, entry_price, sl_price, tp_price, entry_ts_str)
        except Exception as e:
            logger.warning(f"진입 차트 생성 실패: {e}")

        tg.send_entry(
            side=signal, entry=entry_price, sl=sl_price, tp=tp_price,
            risk_usdt=risk_usdt, tp_usdt=risk_usdt * RR_RATIO,
            balance=balance, stats=self._stats(),
            image_bytes=img_bytes,
        )

    def _place_sltp(self, side: str, sl_price: float, tp_price: float,
                    contracts: int) -> tuple:
        # Long: SL rule=2(<=), TP rule=1(>=)
        # Short: SL rule=1(>=), TP rule=2(<=)
        sl_rule    = 2 if side == "long" else 1
        tp_rule    = 1 if side == "long" else 2
        close_size = -contracts if side == "long" else contracts

        sl_id = tp_id = None
        for label, price, rule in [("SL", sl_price, sl_rule), ("TP", tp_price, tp_rule)]:
            # SL: market(IOC) — speed over fill quality
            # TP: limit(GTC) — fill at tp_price or better, no slippage
            order_price = "0"            if label == "SL" else f"{price:.1f}"
            order_tif   = "ioc"          if label == "SL" else "gtc"
            try:
                result = self.api.create_price_triggered_order(
                    settle=SETTLE,
                    futures_price_triggered_order=FuturesPriceTriggeredOrder(
                        initial=FuturesInitialOrder(
                            contract=CONTRACT,
                            size=close_size,
                            price=order_price,
                            tif=order_tif,
                            reduce_only=True,
                            text=f"t-cr-{label.lower()}",
                        ),
                        trigger=FuturesPriceTrigger(
                            strategy_type=0,
                            price_type=0,  # 0=last price
                            price=f"{price:.1f}",
                            rule=rule,
                            expiration=86400,
                        ),
                    )
                )
                if label == "SL":
                    sl_id = result.id
                else:
                    tp_id = result.id
                logger.info(f"{label} 주문: {price:.1f}  id={result.id}")
            except Exception as e:
                logger.error(f"{label} 주문 실패: {e}")
                tg.send_error(f"{label} 주문 실패: {e}")

        return sl_id, tp_id

    # ── 포지션 모니터링 ───────────────────────────────────────────────
    def check_position(self) -> None:
        if not self.pos:
            return

        try:
            size = get_position_size(self.api)
        except Exception as e:
            logger.warning(f"포지션 체크 실패: {e}")
            return

        if size == 0:
            self._cancel_price_orders()
            self._record_close()
            return

        elapsed = (datetime.now(timezone.utc) - self.pos.entry_ts).total_seconds()
        if elapsed >= MAX_HOLD_SEC:
            logger.info(f"최대 보유 초과 ({elapsed / 3600:.1f}h) — 강제 청산")
            self._force_close()

    def _cancel_price_orders(self) -> None:
        if not self.pos:
            return
        for oid in [self.pos.sl_id, self.pos.tp_id]:
            if oid is None:
                continue
            try:
                self.api.cancel_price_triggered_order(
                    settle=SETTLE, order_id=str(oid))
                logger.info(f"트리거 주문 취소: {oid}")
            except Exception as e:
                logger.debug(f"취소 실패 (이미 체결?): {oid}  {e}")

    def _record_close(self) -> None:
        pos = self.pos
        time.sleep(2)

        pnl = None
        try:
            closes = self.api.list_position_close(
                settle=SETTLE, contract=CONTRACT, limit=1)
            if closes:
                pnl = float(closes[0].pnl)
        except Exception as e:
            logger.debug(f"position_close 조회 실패: {e}")

        # 조회 실패 시 SL/TP 거리로 추정
        if pnl is None:
            try:
                cur      = get_last_price(self.api)
                dist_sl  = abs(cur - pos.sl)
                dist_tp  = abs(cur - pos.tp)
                pnl      = 1.0 if dist_tp < dist_sl else -1.0
            except Exception:
                pnl = 0.0

        if pos.sl != 0.0:
            sl_dist   = abs(pos.entry_price - pos.sl)
            risk_usdt = pos.contracts * QUANTO * sl_dist
        else:
            risk_usdt = RISK_USDT
        r_unit = pnl / risk_usdt if risk_usdt > 0 else 0.0

        if pnl > 0:
            self._wins += 1
            self._total_profit += pnl
            status = "WIN"
        elif pnl < 0:
            self._losses += 1
            self._total_loss += pnl
            status = "LOSS"
        else:
            status = "EVEN"
        self._save_stats()

        try:
            balance = get_balance(self.api)
        except Exception:
            balance = 0.0

        hold_secs = (datetime.now(timezone.utc) - pos.entry_ts).total_seconds()
        logger.info(f"청산: {status}  R={r_unit:+.2f}  잔고=${balance:.2f}  보유={hold_secs/60:.1f}m")

        exit_img = None
        if self.df is not None:
            try:
                exit_img = make_exit_chart(self.df, pos, status)
            except Exception as e:
                logger.warning(f"청산 차트 생성 실패: {e}")

        tg.send_exit(status=status, side=pos.side, entry=pos.entry_price,
                     pnl_usdt=pnl, r_unit=r_unit, balance=balance, stats=self._stats(),
                     image_bytes=exit_img, hold_seconds=hold_secs)
        self.pos = None

    def _force_close(self) -> None:
        pos = self.pos
        self._cancel_price_orders()

        close_size = -pos.contracts if pos.side == "long" else pos.contracts
        try:
            self.api.create_futures_order(
                settle=SETTLE,
                futures_order=FuturesOrder(
                    contract=CONTRACT, size=close_size,
                    price="0", tif="ioc",
                    reduce_only=True,
                    text="t-cr-force",
                )
            )
            logger.info("강제 청산 주문 제출")
        except Exception as e:
            logger.error(f"강제 청산 실패: {e}")
            tg.send_error(f"강제 청산 실패: {e}")
            return

        time.sleep(3)
        self._record_close()

    # ── 일일 보고서 ──────────────────────────────────────────────────
    def maybe_daily_report(self) -> None:
        now  = datetime.now(KST)
        date = now.strftime("%Y-%m-%d")
        if now.hour == 9 and self._daily_date != date:
            tg.send_daily_report(wins=self._wins, losses=self._losses)
            self._daily_date = date

    # ── 메인 루프 ────────────────────────────────────────────────────
    def run(self) -> None:
        logger.info(
            f"Candle-Reversal 시작 | {'DEMO' if DEMO else 'LIVE'} | "
            f"big={BIG_MULT} cover={COVER_PCT} rr={RR_RATIO} avg={AVG_LEN} | "
            f"고정 ${RISK_USDT:.0f} / {LEVERAGE}x"
        )
        tg.send_startup(demo=DEMO, risk_pct=0.0)

        while True:
            try:
                df = fetch_klines(self.api)
                self.df = df
                self.check_position()
                self.try_enter(df)
                self.maybe_daily_report()
            except KeyboardInterrupt:
                logger.info("종료")
                tg.send_shutdown()
                break
            except Exception as e:
                body = getattr(e, 'body', None) or ''
                status = getattr(e, 'status', '')
                prefix = f"({status}) " if status else ""
                logger.error(f"루프 에러: {e}", exc_info=True)
                tg.send_error(f"{prefix}{str(body)[:250] or str(e)[:250]}")

            time.sleep(POLL_SEC)


if __name__ == "__main__":
    CandleReversalTrader().run()
