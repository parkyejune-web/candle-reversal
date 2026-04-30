# Pine Script v6 — CE10156 방지 핵심 규칙

## 절대 규칙 (어기면 CE10156 발생)

1. `//@version=6` 앞에 공백 절대 금지
2. `strategy()` 앞에 공백 절대 금지
3. 모든 최상위(top-level) 라인 들여쓰기 금지
4. 함수 인자 줄바꿈 금지 — strategy(), plotshape() 등 전부 한 줄
5. 복붙 시 앞 공백이 같이 들어가면 즉시 CE10156 발생

## 추가 금지 사항

- `strategy()` 에 `initial_capital`, `default_qty_type`, `commission_type` 등 v6 미지원 파라미터 금지
- `input.float(defval=2.4, ...)` — defval= 접두사 금지
- if 블록 내 지역변수 타입 어노테이션 금지: `float sl_dist = ...` → 오류
- 변수 정렬용 이중 공백 금지: `big_mult  =` → 오류

## 올바른 최소 strategy() 선언

```
strategy("Candle Reversal", overlay=true, calc_on_order_fills=true)
```

## 올바른 input 선언

```
big_mult = input.float(2.4, "Big Candle Mult", minval=1.0, step=0.1)
```

## if 블록 내 지역변수

```
if short_sig and no_pos
    sl_dist = math.max(high - close, 0.001)
```

## var 전역 지속 변수 (타입 어노테이션 OK)

```
var float saved_sl = na
```

---

# 전략 현황 (2026-04-29 기준)

## 채택 파라미터 (RESULTS.md 기준)

```
big_mult  = 1.8
cover_pct = 0.3
rr_ratio  = 3.5
avg_len   = 10
```

- IS 2024 / OOS 2025 그리드 서치 검증 완료
- OOS Expectancy: 0.192R, WR: 26.5%, MDD: 61R

## strategy_v6.pine 현재 상태

- 파일: `D:\candle-reversal\strategy_v6.pine`
- defaults: big_mult=2.4, cover_pct=0.7, rr_ratio=4.3, avg_len=13 (검증용)
- calc_on_order_fills=true 포함 (position_avg_price 즉시 사용을 위해 필수)
- process_orders_on_close 없음 → 다음봉 open 진입 (Python과 동일)
- TP: position_avg_price 기반 수동 close (strategy.exit 미사용)
- max_bars=1440 타임아웃 포함

## Pine 차트에서 1거래만 보이는 진짜 원인 (2026-04-29 확인)

**코드 버그 아님. 바 제한이 원인.**

| 플랜 | 30분봉 최대 로드 | 30분봉 기준 (~days) |
|------|-----------------|---------------------|
| Free | 5,000 bars | ~104일 (약 Jan 15까지) |
| Pro  | 10,000 bars | ~208일 |
| Premium | 20,000 bars | ~416일 |
| Expert | 25,000 bars | ~520일 |
| Elite | 30,000 bars | ~625일 |

2026-01-15 이전 봉은 Free 플랜에서 로드 불가 → Signal 1 (Jan-02), Signal 2 (Jan-06)은 범위 밖.
Signal 3 (Jan-22)만 보임 → 1거래. **정상 동작**.

**3거래 전부 보려면:** Strategy Properties → Deep Backtesting 활성화 → Strategy Report 탭 (차트 아님).

## TradingView 백테스팅 핵심 인사이트

### calc_on_order_fills=true (필수)
- `strategy.position_avg_price`는 진입봉 close까지 na 반환
- `calc_on_order_fills=true` 추가 시 주문 체결 직후 스크립트 재실행 → position_avg_price 즉시 사용 가능
- Python의 open[i+1] 진입가와 1:1 매칭 가능

### 주문 실행 타이밍
- 기본값: 다음봉 open (Python과 동일, 이것만 써야 함)
- process_orders_on_close=true: 당봉 close → TP 레벨 달라짐 → 절대 사용 금지

### Deep Backtesting
- 활성화 위치: Strategy Properties (렌치 아이콘) → Deep Backtesting
- 결과 위치: Strategy Report 탭 (차트에는 여전히 5000봉만 표시)
- 더 많은 역사 데이터로 백테스트 가능 (플랜별 한도 적용)

### strategy.exit vs strategy.close
- strategy.exit의 limit 파라미터가 v6에서 신뢰할 수 없음
- strategy.close로 수동 TP/SL 체크해야 정확함

## Python ↔ Pine 수식 대응

| 항목 | Python | Pine |
|------|--------|------|
| 진입가 | open[i+1] | strategy.position_avg_price |
| SL (LONG) | low[i] | sl_price (신호봉에서 저장) |
| sl_dist | open[i+1] - low[i] | position_avg_price - sl_price |
| TP | open[i+1] + sl_dist * rr | position_avg_price + sl_d * rr_ratio |

## 핵심 파일

- `backtest_risk.py` — Python 백테스트 (기준)
- `check_jan2026.py` — 1월 신호/거래 상세 출력 (디버그용)
- `strategy_v6.pine` — TradingView Pine Script
- `RESULTS.md` — 파라미터 최적화 결과
