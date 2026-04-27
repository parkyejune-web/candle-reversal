"""
Candle-Reversal Live Trader — Gate.io USDT-PERP
전략: BTC_USDT 1m  장대봉+반대봉 역추세
params: big_mult=1.8, cover=0.3, rr=3.5, avg_len=10
SL: 신호봉 고가(short) / 저가(long)
TP: entry ± sl_dist × 3.5
리스크: 잔고의 5%/트레이드 (50 USDT @ 1000 USDT), 레버리지 10x
"""
import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
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
RISK_PCT     = 0.05           # 잔고의 5% (1000 USDT → 손실 50 USDT/트레이드)
LEVERAGE     = 10             # 레버리지 (진입 마진 = 포지션/10)

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
    api = FuturesApi(ApiClient(cfg))
    try:
        api.update_position_leverage(
            settle=SETTLE,
            contract=CONTRACT,
            leverage=str(LEVERAGE),
            cross_leverage_limit="0",
        )
        logger.info(f"레버리지 {LEVERAGE}x 설정 완료")
    except Exception as e:
        logger.warning(f"레버리지 설정 실패: {e}")
    return api


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


# ── 유틸 ──────────────────────────────────────────────────────────────
def get_balance(api: FuturesApi) -> float:
    return float(api.list_futures_accounts(settle=SETTLE).available)


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


def calc_contracts(balance: float, sl_dist: float) -> int:
    """0.5% 리스크 기반 계약수 (최소 1)."""
    risk_usdt = balance * RISK_PCT
    return max(1, int(risk_usdt / (sl_dist * QUANTO)))


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


class CandleReversalTrader:
    def __init__(self):
        self.api      = make_api()
        self.pos: Optional[Position] = None
        self._last_sig_ts: Optional[int] = self._load_sig_ts()
        self._wins    = 0
        self._losses  = 0
        self._daily_date: Optional[str] = None

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

    def _stats(self) -> dict:
        t = self._wins + self._losses
        return {"wins": self._wins, "losses": self._losses,
                "winrate": self._wins / t * 100 if t else 0.0}

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

        # 잔고 + 현재가
        try:
            balance    = get_balance(self.api)
            last_price = get_last_price(self.api)
        except Exception as e:
            logger.warning(f"잔고/시세 조회 실패: {e}")
            return

        sig_bar  = df.iloc[-1]
        sl_price = float(sig_bar["high"] if signal == "short" else sig_bar["low"])
        sl_dist  = abs(last_price - sl_price)
        if sl_dist <= 0:
            logger.warning("SL 거리 0 — 스킵")
            return

        tp_price  = (last_price - sl_dist * RR_RATIO if signal == "short"
                     else last_price + sl_dist * RR_RATIO)
        contracts = calc_contracts(balance, sl_dist)

        logger.info(
            f"신호: {signal.upper()} | last≈{last_price:.1f} "
            f"sl={sl_price:.1f} tp={tp_price:.1f} | "
            f"contracts={contracts} risk≈${balance * RISK_PCT:.2f}"
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
            logger.error(f"주문 실패: {e}")
            tg.send_error(str(e)[:300])
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
        tg.send_entry(
            side=signal, entry=entry_price, sl=sl_price, tp=tp_price,
            contracts=actual_contracts, risk_pct=RISK_PCT * 100,
            balance=balance, stats=self._stats(),
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
            try:
                result = self.api.create_price_triggered_order(
                    settle=SETTLE,
                    futures_price_triggered_order=FuturesPriceTriggeredOrder(
                        initial=FuturesInitialOrder(
                            contract=CONTRACT,
                            size=close_size,
                            price="0",
                            tif="ioc",
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

        sl_dist   = abs(pos.entry_price - pos.sl)
        risk_usdt = pos.contracts * QUANTO * sl_dist
        r_unit    = pnl / risk_usdt if risk_usdt > 0 else 0.0

        if pnl > 0:
            self._wins += 1
            status = "WIN"
        elif pnl < 0:
            self._losses += 1
            status = "LOSS"
        else:
            status = "EVEN"

        logger.info(f"청산: {status}  R={r_unit:+.2f}")
        tg.send_exit(status=status, side=pos.side, entry=pos.entry_price,
                     r_unit=r_unit, stats=self._stats())
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
            f"risk={RISK_PCT * 100:.1f}%"
        )
        tg.send_startup(demo=DEMO, risk_pct=RISK_PCT * 100)

        while True:
            try:
                df = fetch_klines(self.api)
                self.check_position()
                self.try_enter(df)
                self.maybe_daily_report()
            except KeyboardInterrupt:
                logger.info("종료")
                tg.send_shutdown()
                break
            except Exception as e:
                logger.error(f"루프 에러: {e}", exc_info=True)
                tg.send_error(str(e)[:300])

            time.sleep(POLL_SEC)


if __name__ == "__main__":
    CandleReversalTrader().run()
