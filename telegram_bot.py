"""Telegram 알림 — Candle-Reversal Bot (Gate.io)."""
import os
import logging
import requests
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("CandleReversal.TG")

KST = timezone(timedelta(hours=9))

_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def _send(text: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"TG 전송 실패: {e}")


def _now_kst() -> str:
    return datetime.now(KST).strftime("%m/%d %H:%M")


def send_startup(demo: bool, risk_pct: float) -> None:
    mode = "📋 DEMO" if demo else "🚀 LIVE"
    _send(
        f"{mode} | Candle-Reversal 시작\n"
        f"▸ BTC_USDT 1m | 장대봉+반대봉 역추세\n"
        f"▸ 리스크 {risk_pct:.1f}%/트레이드\n"
        f"▸ big=1.8  cover=0.3  rr=3.5  avg=10\n"
        f"▸ {_now_kst()} KST"
    )


def send_entry(side: str, entry: float, sl: float, tp: float,
               risk_usdt: float, tp_usdt: float,
               balance: float, stats: dict) -> None:
    emoji  = "🔴 숏 (SHORT)" if side == "short" else "🟢 롱 (LONG)"
    rr     = tp_usdt / risk_usdt if risk_usdt else 0
    w, l   = stats["wins"], stats["losses"]
    wr     = stats["winrate"]
    _send(
        f"{emoji} 진입: BTC_USDT\n\n"
        f"진입가: <code>${entry:,.1f}</code>\n"
        f"손절가: <code>${sl:,.1f}</code>  (여기 오면 -${risk_usdt:.2f})\n"
        f"목표가: <code>${tp:,.1f}</code>  (여기 오면 +${tp_usdt:.2f}, RR {rr:.1f})\n\n"
        f"💰 잔고: ${balance:,.2f}\n"
        f"📊 누적: {w}승 {l}패  ({wr:.1f}%)\n"
        f"시각: {_now_kst()} KST"
    )


def send_exit(status: str, side: str, entry: float,
              pnl_usdt: float, r_unit: float, stats: dict) -> None:
    if status == "WIN":
        emoji = "✅ 익절"
    elif status == "LOSS":
        emoji = "💥 손절"
    else:
        emoji = "➖ 본전"
    direction = "숏" if side == "short" else "롱"
    sign      = "+" if pnl_usdt >= 0 else ""
    w, l      = stats["wins"], stats["losses"]
    wr        = stats["winrate"]
    _send(
        f"{emoji}: BTC_USDT {direction}\n\n"
        f"진입가: <code>${entry:,.1f}</code>\n"
        f"실현 손익: {sign}${pnl_usdt:.2f}  ({sign}{r_unit:.2f}R)\n\n"
        f"📊 누적: {w}승 {l}패  ({wr:.1f}%)\n"
        f"시각: {_now_kst()} KST"
    )


def send_daily_report(wins: int, losses: int) -> None:
    t  = wins + losses
    wr = wins / t * 100 if t else 0.0
    _send(
        f"📊 일일 리포트\n"
        f"▸ 거래   {t}건  ({wins}승 {losses}패)\n"
        f"▸ 승률   {wr:.1f}%\n"
        f"▸ {_now_kst()} KST"
    )


def send_shutdown() -> None:
    _send(f"🛑 Candle-Reversal 종료  {_now_kst()} KST")


def send_error(msg: str) -> None:
    _send(f"⚠️ 오류\n<code>{msg}</code>\n▸ {_now_kst()} KST")
