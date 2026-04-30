"""
backtest_30m_summary.xlsx 생성
Usage: python make_excel.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

OUT = "backtest_30m_summary.xlsx"

BG_DARK   = "0D1117"
BG_HEADER = "161B22"
BG_ROW1   = "1C2128"
BG_ROW2   = "21262D"
FG_WHITE  = "E6EDF3"
FG_GRAY   = "8B949E"
FG_GREEN  = "3FB950"
FG_RED    = "F85149"
FG_BLUE   = "58A6FF"
FG_YELLOW = "D29922"

def hfill(hex_): return PatternFill("solid", fgColor=hex_)
def font(hex_, bold=False, sz=10): return Font(color=hex_, bold=bold, size=sz, name="Calibri")
def align(h="center", v="center"): return Alignment(horizontal=h, vertical=v, wrap_text=True)
def thin_border():
    s = Side(style="thin", color="30363D")
    return Border(left=s, right=s, top=s, bottom=s)

def write_header_row(ws, row, cols):
    for c, (val, width) in enumerate(cols, 1):
        cell = ws.cell(row=row, column=c, value=val)
        cell.fill      = hfill(BG_HEADER)
        cell.font      = font(FG_BLUE, bold=True, sz=9)
        cell.alignment = align()
        cell.border    = thin_border()
        ws.column_dimensions[get_column_letter(c)].width = width

def write_data_row(ws, row, vals, fg=None, bg=None, bold=False):
    bg = bg or (BG_ROW1 if row % 2 == 0 else BG_ROW2)
    for c, val in enumerate(vals, 1):
        f  = fg[c-1] if (fg and c-1 < len(fg)) else FG_WHITE
        cell = ws.cell(row=row, column=c, value=val)
        cell.fill      = hfill(bg)
        cell.font      = font(f, bold=bold, sz=9)
        cell.alignment = align()
        cell.border    = thin_border()

def set_ws_style(ws):
    ws.sheet_view.showGridLines = False

# Sheet 1: Strategy Overview
def sheet_overview(wb):
    ws = wb.create_sheet("Strategy Overview")
    set_ws_style(ws)

    ws.merge_cells("A1:D1")
    t = ws["A1"]
    t.value     = "Candle Reversal Strategy - BTCUSDT 30m"
    t.fill      = hfill(BG_DARK)
    t.font      = font(FG_BLUE, bold=True, sz=13)
    t.alignment = align()
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:D2")
    s = ws["A2"]
    s.value     = "Big Candle + Reversal Candle Mean-Reversion | Gate.io Futures"
    s.fill      = hfill(BG_DARK)
    s.font      = font(FG_GRAY, sz=9)
    s.alignment = align()

    ws.merge_cells("A3:D3")
    ph = ws["A3"]
    ph.value     = "Parameters"
    ph.fill      = hfill(BG_HEADER)
    ph.font      = font(FG_YELLOW, bold=True, sz=10)
    ph.alignment = align()

    write_header_row(ws, 4, [("Parameter",22),("Value",18),("Description",40),("Notes",18)])
    params = [
        ("big_mult",  "2.4",   "Signal candle >= avg_body x multiplier", "Optimized OOS"),
        ("cover_pct", "0.7",   "Reversal candle overlap ratio threshold", "Optimized OOS"),
        ("rr_ratio",  "4.3",   "Risk:Reward ratio",                       "Optimized OOS"),
        ("avg_len",   "13",    "Rolling window for avg body size",         "Optimized OOS"),
        ("taker_fee", "0.01%", "Gate.io standard 0.05% taker",            "Tolerance < 0.11%"),
    ]
    for i, row in enumerate(params, 5):
        write_data_row(ws, i, row, fg=[FG_BLUE, FG_GREEN, FG_GRAY, FG_GRAY])

    ws.merge_cells("A11:D11")
    pw = ws["A11"]
    pw.value     = "Backtest Periods"
    pw.fill      = hfill(BG_HEADER)
    pw.font      = font(FG_YELLOW, bold=True, sz=10)
    pw.alignment = align()

    write_header_row(ws, 12, [("Period",22),("Date Range",18),("Purpose",40),("Notes",18)])
    period_data = [
        ("IS  (In-Sample)",    "2020-01 ~ 2022-12", "Parameter optimization",    "Grid search"),
        ("OOS (Out-of-Sample)","2023-01 ~ 2026-04", "Live performance proxy",     "Main evaluation"),
    ]
    for i, row in enumerate(period_data, 13):
        write_data_row(ws, i, row)

    ws.merge_cells("A16:D16")
    perf = ws["A16"]
    perf.value     = "OOS Performance Summary"
    perf.fill      = hfill(BG_HEADER)
    perf.font      = font(FG_YELLOW, bold=True, sz=10)
    perf.alignment = align()

    write_header_row(ws, 17, [("Metric",22),("IS Value",18),("OOS Value",40),("Target",18)])
    perf_data = [
        ("Win Rate",              "~30% est",    "26.5%",        ">25%"),
        ("Expectancy",            "+0.2R est",   "+0.192R",      ">0"),
        ("MDD (R)",               "~9R est",     "17.4R",        "<20R"),
        ("Max Consec Losses",     "~10 est",     "17 (confirmed)", "<20"),
        ("Full Kelly",            "--",           "17.6%",        "--"),
        ("Half Kelly",            "--",           "8.8%",         "--"),
        ("Recommended Risk",      "--",           "3.9%",         "Compound MDD <50%"),
        ("Monthly Return @ 3.9%", "--",           "~+10%",        "Estimated"),
    ]
    for i, row in enumerate(perf_data, 18):
        is_rec = row[0] == "Recommended Risk"
        bg = "1A3320" if is_rec else None
        write_data_row(ws, i, row, fg=[FG_WHITE, FG_WHITE, FG_GREEN, FG_GRAY], bg=bg)


# Sheet 2: Kelly & Risk
def sheet_kelly(wb):
    ws = wb.create_sheet("Kelly & Risk")
    set_ws_style(ws)

    ws.merge_cells("A1:E1")
    t = ws["A1"]
    t.value     = "Kelly Criterion & Risk Management"
    t.fill      = hfill(BG_DARK)
    t.font      = font(FG_BLUE, bold=True, sz=13)
    t.alignment = align()
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:E2")
    ws["A2"].value     = "f* = (WR x RR - (1-WR)) / RR  |  OOS: WR=26.5%  RR=4.3  MDD=17.4R"
    ws["A2"].fill      = hfill(BG_DARK)
    ws["A2"].font      = font(FG_GRAY, sz=9)
    ws["A2"].alignment = align()

    write_header_row(ws, 3, [
        ("Risk %", 12), ("Simple MDD %", 16), ("Compound MDD %", 18),
        ("Monthly Return", 18), ("Assessment", 26)
    ])

    wr  = 0.265; rr = 4.3; mdd_r = 17.4
    full_k = (wr * rr - (1 - wr)) / rr * 100
    half_k = full_k / 2

    risks = [0.3, 0.5, 0.75, 1.0, 1.3, 2.0, 3.0, 3.9, round(half_k,2), round(full_k,2)]
    seen = set()
    rows_data = []
    for p in risks:
        p = round(p, 2)
        if p in seen or p <= 0:
            continue
        seen.add(p)
        simple   = mdd_r * p
        compound = (1 - (1 - p/100)**mdd_r) * 100
        monthly  = ((1 + 0.192 * p / 100) ** 30 - 1) * 100
        if compound < 20:    verdict = "[OK] Very Safe"
        elif compound < 35:  verdict = "[OK] Safe"
        elif compound < 50:  verdict = "[!]  Caution"
        elif compound < 70:  verdict = "[X]  Risky"
        else:                verdict = "[!!] Danger"
        rows_data.append((p, simple, compound, monthly, verdict))

    for i, (p, s, c, m, v) in enumerate(rows_data, 4):
        is_opt = abs(p - 3.9) < 0.05
        is_hk  = abs(p - half_k) < 0.15
        is_fk  = abs(p - full_k) < 0.15
        if c < 35:    c_color = FG_GREEN
        elif c < 50:  c_color = FG_YELLOW
        else:         c_color = FG_RED
        bg = "1A3320" if is_opt else None
        tag = ""
        if is_opt:  tag += " <-- OPTIMAL"
        if is_hk and not is_opt: tag += " <-- Half Kelly"
        if is_fk and not is_opt: tag += " <-- Full Kelly"
        write_data_row(ws, i, [
            f"{p:.2f}%", f"{s:.1f}%", f"{c:.1f}%", f"+{m:.1f}%", v + tag
        ], fg=[FG_WHITE, FG_WHITE, c_color, FG_GREEN, FG_WHITE], bg=bg)
        if is_opt:
            for col in range(1, 6):
                ws.cell(row=i, column=col).font = font(
                    FG_GREEN if col != 3 else c_color, bold=True, sz=9)

    summary_row = len(rows_data) + 5
    ws.merge_cells(f"A{summary_row}:E{summary_row}")
    cell = ws[f"A{summary_row}"]
    cell.value     = (f"RECOMMENDED: 3.9% risk | Compound MDD ~50% | Monthly ~+10% | "
                      f"Full Kelly={full_k:.1f}% Half Kelly={half_k:.1f}%")
    cell.fill      = hfill("1A3320")
    cell.font      = font(FG_GREEN, bold=True, sz=10)
    cell.alignment = align()


# Sheet 3: Consecutive Losses
def sheet_consec(wb):
    ws = wb.create_sheet("Consecutive Losses")
    set_ws_style(ws)

    ws.merge_cells("A1:D1")
    t = ws["A1"]
    t.value     = "17-Consecutive-Loss Streak (OOS Confirmed)"
    t.fill      = hfill(BG_DARK)
    t.font      = font(FG_RED, bold=True, sz=13)
    t.alignment = align()
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:D2")
    ws["A2"].value     = "Period: 2024-03-08 ~ 2024-05-14 (67 days) | MDD 17.4R confirmed"
    ws["A2"].fill      = hfill(BG_DARK)
    ws["A2"].font      = font(FG_GRAY, sz=9)
    ws["A2"].alignment = align()

    write_header_row(ws, 3, [("Metric",26),("Value",20),("Notes",36),("Impact",20)])
    data = [
        ("Max Consecutive Losses",  "17",          "2024-03-08 to 2024-05-14",  "67 trading days"),
        ("MDD in R",                "17.4R",        "1R = 1x risk amount",       "Worst drawdown"),
        ("MDD at 1.0% risk",        "17.4%",        "Account drawdown",          "Recoverable"),
        ("MDD at 3.9% risk",        "~50%",         "Compound MDD formula",      "Hard to recover"),
        ("BTC Price Context",       "$62k-$71k",    "2024 March to May",         "Ranging/Trending up"),
        ("Signal Type Conflict",    "Mean-revert",  "Trend was upward",          "Explains losses"),
        ("Recovery needed",         "+100%",        "From -50% drawdown",        "At 3.9% risk"),
    ]
    for i, row in enumerate(data, 4):
        write_data_row(ws, i, row, fg=[FG_WHITE, FG_RED, FG_GRAY, FG_YELLOW])

    ws.merge_cells("A12:D12")
    ws["A12"].value     = "Risk Mitigation"
    ws["A12"].fill      = hfill(BG_HEADER)
    ws["A12"].font      = font(FG_YELLOW, bold=True, sz=10)
    ws["A12"].alignment = align()

    write_header_row(ws, 13, [("Strategy",26),("Implementation",20),("Effect",36),("Type",20)])
    mitigations = [
        ("Conservative start",  "Begin at 1.3% risk",        "Compound MDD ~20%",        "Gradual scaling"),
        ("Hard stop rule",      "Pause at -30% account",     "Force review before resume","Circuit breaker"),
        ("Kelly fraction",      "Use 3.9% (Half-ish Kelly)", "Balance growth vs ruin",   "Recommended"),
        ("Trend filter",        "Skip during strong trends",  "Fewer false reversals",    "Strategy mod"),
    ]
    for i, row in enumerate(mitigations, 14):
        write_data_row(ws, i, row)


# Sheet 4: Fee Tolerance
def sheet_fees(wb):
    ws = wb.create_sheet("Fee Tolerance")
    set_ws_style(ws)

    ws.merge_cells("A1:D1")
    t = ws["A1"]
    t.value     = "Fee Tolerance Analysis"
    t.fill      = hfill(BG_DARK)
    t.font      = font(FG_BLUE, bold=True, sz=13)
    t.alignment = align()
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:D2")
    ws["A2"].value     = "Natural leverage avg ~175x on 30m | Effective fee = taker_fee x leverage x 2 sides"
    ws["A2"].fill      = hfill(BG_DARK)
    ws["A2"].font      = font(FG_GRAY, sz=9)
    ws["A2"].alignment = align()

    write_header_row(ws, 3, [("Taker Fee",16),("Eff. Fee/Trade (R)",22),("OOS Expectancy",22),("Assessment",24)])

    base_exp = 0.386
    avg_lev  = 175.0
    fees = [0.00, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]
    for i, fee_pct in enumerate(fees, 4):
        fee_dec    = fee_pct / 100
        eff_fee    = fee_dec * avg_lev * 2
        expectancy = base_exp - eff_fee
        if expectancy > 0.2:    verdict = "[OK] Excellent"
        elif expectancy > 0:    verdict = "[OK] Profitable"
        elif expectancy > -0.1: verdict = "[!]  Marginal"
        else:                   verdict = "[X]  Unprofitable"
        color = FG_GREEN if expectancy > 0 else FG_RED
        is_gate = abs(fee_pct - 0.05) < 0.001
        bg = "1A3320" if is_gate else None
        write_data_row(ws, i, [
            f"{fee_pct:.3f}%",
            f"{eff_fee:.3f}R",
            f"{expectancy:+.3f}R",
            verdict + (" <-- Gate.io standard" if is_gate else ""),
        ], fg=[FG_WHITE, FG_WHITE, color, FG_WHITE], bg=bg)

    summary_row = len(fees) + 5
    ws.merge_cells(f"A{summary_row}:D{summary_row}")
    cell = ws[f"A{summary_row}"]
    cell.value     = "Gate.io 0.05% taker | OOS Expectancy: +0.234R | Break-even fee: ~0.11%"
    cell.fill      = hfill("1A3320")
    cell.font      = font(FG_GREEN, bold=True, sz=10)
    cell.alignment = align()

    exc_row = summary_row + 2
    ws.merge_cells(f"A{exc_row}:D{exc_row}")
    ws[f"A{exc_row}"].value     = "Exchange Fee Comparison"
    ws[f"A{exc_row}"].fill      = hfill(BG_HEADER)
    ws[f"A{exc_row}"].font      = font(FG_YELLOW, bold=True, sz=10)
    ws[f"A{exc_row}"].alignment = align()

    write_header_row(ws, exc_row+1, [("Exchange",16),("Taker Fee",22),("Status",22),("Notes",24)])
    exchanges = [
        ("Gate.io",  "0.050%", "Optimal",  "Default choice"),
        ("Binance",  "0.040%", "Better",   "Higher volume required"),
        ("Bybit",    "0.055%", "OK",        "Slightly higher"),
        ("OKX",      "0.050%", "OK",        "Same as Gate.io"),
        ("Bitget",   "0.060%", "OK",        "Within tolerance"),
        ("KuCoin",   "0.060%", "OK",        "Within tolerance"),
        ("Kraken",   "0.100%", "Marginal",  "At break-even limit"),
    ]
    for i, row in enumerate(exchanges, exc_row+2):
        fee_val = float(row[1].replace("%",""))
        fg = [FG_WHITE,
              FG_GREEN if fee_val <= 0.05 else FG_YELLOW,
              FG_GREEN if fee_val <= 0.06 else FG_YELLOW,
              FG_GRAY]
        write_data_row(ws, i, row, fg=fg)


def main():
    wb = Workbook()
    wb.remove(wb.active)

    sheet_overview(wb)
    sheet_kelly(wb)
    sheet_consec(wb)
    sheet_fees(wb)

    wb.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
