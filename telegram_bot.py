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
               contracts: int, risk_pct: float, balance: float,
               stats: dict) -> None:
    emoji = "🔴 SHORT" if side == "short" else "🟢 LONG"
    wr    = stats["winrate"]
    w, l  = stats["wins"], stats["losses"]
    _send(
        f"{emoji} 진입\n"
        f"▸ Entry  <code>{entry:.1f}</code>\n"
        f"▸ SL     <code>{sl:.1f}</code>\n"
        f"▸ TP     <code>{tp:.1f}</code>\n"
        f"▸ 계약수 {contracts}  리스크 {risk_pct:.1f}%\n"
        f"▸ 잔고   ${balance:,.0f}\n"
        f"▸ 성과   {w}승 {l}패  WR {wr:.1f}%\n"
        f"▸ {_now_kst()} KST"
    )


def send_exit(status: str, side: str, entry: float,
              r_unit: float, stats: dict) -> None:
    if status == "WIN":
        emoji = "✅ WIN"
    elif status == "LOSS":
        emoji = "❌ LOSS"
    else:
        emoji = "➖ EVEN"
    dir_str = "SHORT" if side == "short" else "LONG"
    wr = stats["winrate"]
    w, l = stats["wins"], stats["losses"]
    _send(
        f"{emoji}  {dir_str} {r_unit:+.2f}R\n"
        f"▸ Entry  <code>{entry:.1f}</code>\n"
        f"▸ 누적   {w}승 {l}패  WR {wr:.1f}%\n"
        f"▸ {_now_kst()} KST"
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
