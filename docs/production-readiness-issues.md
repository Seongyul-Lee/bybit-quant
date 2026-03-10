# 실전 투입 잔존 결함 목록

> 작성일: 2026-03-09
> 기준 브랜치: `preview`
> 이전 수정 세션에서 7건의 치명적 결함 중 5건을 해결했으며, 추가 분석으로 발견된 잔존 결함을 정리한다.

## 수정 가이드

- 각 이슈를 **번호 순서대로** 하나씩 수정한다
- 수정 후 `[x]`로 체크하고 수정 내역을 기록한다
- CRITICAL → HIGH → MEDIUM → LOW 순서로 처리한다

---

## CRITICAL

### C-1. atr_14가 raw OHLCV에 없어 변동성 체크/포지션 사이징 오작동

- [x] 수정 완료

**파일:** `main.py:218`, `main.py:242`

**문제:**
processor 제거 후 `df`는 raw OHLCV(timestamp, open, high, low, close, volume)만 포함한다.
전략의 `generate_signal(df)`는 내부에서 `df_feat = self.feature_engine.compute_all_features(df)` 또는 `_get_or_compute_*`로 피처를 계산하지만, 이 결과는 전략 내부 복사본에만 반영되고 **main.py의 `df` 원본에는 반영되지 않는다**.

```python
# main.py:218 — df에 atr_14가 없으므로 항상 0.0
current_vol = float(df["atr_14"].iloc[-1] / df["close"].iloc[-1]) if "atr_14" in df.columns else 0.0

# main.py:242 — df에 atr_14가 없으므로 항상 fallback 100
atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else 100
```

**영향:**
- `current_vol`이 항상 0.0 → `check_all`의 변동성 체크가 항상 스킵 → CircuitBreaker 변동성 보호 무력화
- `atr`가 항상 100 → 실제 ATR과 무관한 고정 포지션 사이즈 계산

**수정 방향:**
`generate_signal()` 호출 전에 main.py에서 직접 ATR을 계산하거나, 전략에서 계산된 피처 DataFrame을 반환받는 인터페이스를 추가한다.

간단한 방법: main.py에서 직접 ATR 14를 계산하는 헬퍼를 사용한다.

```python
# main.py에서 직접 ATR 계산
def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean()
    return float(atr_series.iloc[-1])
```

**검증:**
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/BTCUSDT_1h_features.parquet').tail(500)
# raw OHLCV만 추출
raw = df[['timestamp','open','high','low','close','volume']].copy()
print('atr_14 in raw:', 'atr_14' in raw.columns)  # False
print('atr_14 in processed:', 'atr_14' in df.columns)  # True
print('실제 ATR:', df['atr_14'].iloc[-1])
"
```

---

### C-2. 포지션 사이즈 단위 불일치 (BTC 수량 vs USD 금액)

- [x] 수정 완료

**파일:** `src/risk/manager.py:243-250` (`calculate_atr_position_size`)

**문제:**
```python
position_size = dollar_risk / atr          # 결과: 코인 수량 (예: 0.033 BTC)
max_size = portfolio_value * max_position_pct  # 결과: USD 금액 (예: $500)
return min(position_size, max_size)         # 수량과 금액을 비교 → 의미 없음
```

예시 (portfolio $10,000, ATR 300, BTC $90,000):
- `position_size` = $10,000 * 0.01 / 300 = 0.333 BTC
- `max_size` = $10,000 * 0.05 = $500
- `min(0.333, 500)` = 0.333 → 캡 미적용

C-1에서 atr=100 fallback이 적용되면:
- `position_size` = $10,000 * 0.01 / 100 = 1.0 BTC ($90,000 상당)
- `max_size` = $500
- `min(1.0, 500)` = 1.0 → **여전히 캡 미적용** (포트폴리오의 900%)

**영향:**
max_position_pct (5%) 상한이 사실상 무력화되어, 포트폴리오 대비 과대 포지션이 실행될 수 있다.

**수정 방향:**
max_size도 코인 수량으로 변환하여 비교하거나, position_size를 USD로 변환하여 비교한다.
현재 함수에는 `entry_price` 파라미터가 없으므로 시그니처 변경이 필요하다.

```python
def calculate_atr_position_size(
    self,
    portfolio_value: float,
    atr: float,
    entry_price: float,  # 추가
    risk_per_trade: Optional[float] = None,
) -> float:
    risk_pct = risk_per_trade or self.params["trade"]["risk_per_trade_pct"]
    dollar_risk = portfolio_value * risk_pct
    if atr <= 0 or entry_price <= 0:
        return 0.0
    position_size = dollar_risk / atr  # 코인 수량

    # max_size도 코인 수량으로 변환
    max_size_usd = portfolio_value * self.params["position"]["max_position_pct"]
    max_size_qty = max_size_usd / entry_price
    return min(position_size, max_size_qty)
```

**주의:** 이 함수의 호출부(`main.py:243`)도 함께 수정해야 한다.

**검증:**
```bash
python -m pytest tests/test_risk_manager.py -v -k "position_size"
```

---

### C-3. 심볼 파싱 하드코딩 (3글자 base만 지원)

- [x] 수정 완료

**파일:** `main.py:98`

**문제:**
```python
symbol = symbol_raw[:3] + "/" + symbol_raw[3:] + ":" + symbol_raw[3:]
```

| symbol_raw | 결과 | 정상 여부 |
|------------|------|-----------|
| `BTCUSDT` | `BTC/USDT:USDT` | O |
| `ETHUSDT` | `ETH/USDT:USDT` | O |
| `SOLUSDT` | `SOL/USDT:USDT` | O |
| `DOGEUSDT` | `DOG/EUSDT:EUSDT` | **X** |
| `SHIBUSDT` | `SHI/BUSDT:BUSDT` | **X** |
| `PEPEUSDT` | `PEP/EUSDT:EUSDT` | **X** |
| `1000PEPEUSDT` | `100/0PEPEUSDT:0PEPEUSDT` | **X** |

**영향:**
BTCUSDT, ETHUSDT, SOLUSDT 등 3글자 base에서만 정상 작동. 다른 심볼로 전략을 실행하면 ccxt가 알 수 없는 심볼 에러를 발생시킨다.

**수정 방향:**
quote 통화(USDT)를 기준으로 분리한다.

```python
def _convert_symbol(symbol_raw: str) -> str:
    """파일명용 심볼(BTCUSDT)을 ccxt 심볼(BTC/USDT:USDT)로 변환."""
    for quote in ("USDT", "USDC", "BTC"):
        if symbol_raw.endswith(quote):
            base = symbol_raw[:-len(quote)]
            return f"{base}/{quote}:{quote}"
    raise ValueError(f"알 수 없는 심볼 형식: {symbol_raw}")
```

**검증:**
```python
assert _convert_symbol("BTCUSDT") == "BTC/USDT:USDT"
assert _convert_symbol("DOGEUSDT") == "DOGE/USDT:USDT"
assert _convert_symbol("1000PEPEUSDT") == "1000PEPE/USDT:USDT"
```

---

## HIGH

### H-1. 재시작 시 prev_positions 미복원 → 청산 감지 누락

- [x] 수정 완료

**파일:** `main.py:132`

**문제:**
```python
prev_positions: dict = {}  # 항상 빈 딕셔너리로 시작
```

프로세스 재시작 후 첫 루프에서 `prev_positions`가 비어있으므로, 다운타임 중 SL/TP로 청산된 포지션을 감지할 수 없다. 결과적으로 해당 청산의 PnL이 `pnl_tracker`와 `circuit_breaker`에 반영되지 않는다.

`current_state.json`에는 `positions` 키로 마지막 포지션 스냅샷이 저장되어 있다.

**수정 방향:**
```python
# 저장된 상태에서 prev_positions 복원
prev_positions = saved_state.get("positions", {})
```

**검증:**
1. 포지션이 있는 상태에서 프로세스 종료
2. 외부에서 포지션 청산 (거래소 웹에서 수동 청산)
3. 프로세스 재시작 후 청산이 감지되는지 로그 확인

---

### H-2. fetch_my_trades PnL 중복 집계

- [x] 수정 완료

**파일:** `main.py:153-158`, `main.py:199-204`

**문제:**
```python
trades = exchange.fetch_my_trades(sym, limit=5)
closed_pnl = sum(
    float(t.get("info", {}).get("closedPnl", 0))
    for t in trades
    if float(t.get("info", {}).get("closedPnl", 0)) != 0
)
```

최근 5건의 모든 `closedPnl`을 합산하므로:
1. 이전 루프에서 이미 기록한 거래의 PnL이 **재차 합산**될 수 있다
2. 여러 부분 체결이 있으면 개별 체결마다 PnL이 중복 집계된다

**영향:**
`pnl_tracker`와 `circuit_breaker`에 실제보다 큰 손실이 기록되어, 일일/월간 손실 한도에 조기 도달하거나 CircuitBreaker가 부당하게 발동될 수 있다.

**수정 방향:**
마지막으로 처리한 거래 ID(`last_processed_trade_id`)를 추적하여, 이미 처리한 거래를 건너뛴다.

```python
# 상태에 last_trade_ids 추적 추가
last_trade_ids: set[str] = set()

# PnL 조회 시
trades = exchange.fetch_my_trades(sym, limit=5)
for t in trades:
    tid = t["id"]
    if tid in last_trade_ids:
        continue
    cpnl = float(t.get("info", {}).get("closedPnl", 0))
    if cpnl != 0:
        pnl_tracker.record_pnl(cpnl)
        circuit_breaker.record_trade(cpnl)
    last_trade_ids.add(tid)
```

`last_trade_ids`는 `current_state.json`에 직렬화하여 영속화한다.

---

### H-3. Graceful Shutdown 시 상태 미저장

- [x] 수정 완료

**파일:** `main.py:280-282`

**문제:**
```python
except KeyboardInterrupt:
    logger.info("사용자에 의해 종료")
    break
```

Ctrl+C로 종료 시 마지막 상태(pnl_tracker, circuit_breaker, last_processed_bar)를 저장하지 않는다. 루프 중간(포지션 동기화 후, 상태 저장 전)에 종료되면 데이터가 유실된다.

**수정 방향:**
```python
except KeyboardInterrupt:
    logger.info("사용자에 의해 종료")
    # 마지막 상태 저장
    try:
        executor._save_state(prev_positions, extra_state={
            "circuit_breaker": risk_manager.circuit_breaker.to_dict(),
            "pnl_tracker": pnl_tracker.to_dict(),
            "last_processed_bar": last_processed_bar,
        })
        logger.info("종료 전 상태 저장 완료")
    except Exception as e:
        logger.error(f"종료 시 상태 저장 실패: {e}")
    break
```

---

## MEDIUM

### M-1. close_position이 멱등성 체크에 의해 거부될 수 있음

- [x] 수정 완료

**파일:** `src/execution/executor.py:86-90`, `src/execution/executor.py:229`

**문제:**
`close_position()`은 내부에서 `execute()`를 호출한다. `execute()`에는 같은 symbol/side의 미체결(pending) 주문이 있으면 중복 주문을 방지하는 로직이 있다.

```python
# execute() 내부
for order_id, order in self.pending_orders.items():
    if order["symbol"] == symbol and order["side"] == side:
        logger.warning(f"중복 주문 방지: ...")
        return None
```

만약 이전에 같은 방향(예: sell)의 미체결 지정가 주문이 남아있는 상태에서 long 포지션을 청산(sell)하려 하면, 멱등성 체크에 의해 **청산이 거부**된다.

**영향:**
포지션 반전 시 기존 포지션 청산이 실패하면, 새 포지션만 진입되어 양방향 포지션이 생길 수 있다.

**수정 방향:**
`close_position()`에서는 멱등성 체크를 우회하거나, 기존 미체결 주문을 먼저 취소한다.

```python
def close_position(self, symbol, position, strategy_name=""):
    close_side = "sell" if position["side"] == "long" else "buy"
    # 같은 방향의 기존 미체결 주문 취소
    for oid, order in list(self.pending_orders.items()):
        if order["symbol"] == symbol and order["side"] == close_side:
            self.cancel(oid, symbol)
    return self.execute(
        symbol=symbol, side=close_side, amount=position["size"],
        order_type="market", strategy_name=strategy_name, signal_score=0,
    )
```

---

### M-2. CircuitBreaker 발동 시 텔레그램 알림 없음

- [x] 수정 완료

**파일:** `src/risk/manager.py:70-77`

**문제:**
```python
def trip(self, reason: str) -> None:
    self.is_tripped = True
    logger.critical(f"[Circuit Breaker 발동] 사유: {reason}")
    # 텔레그램 알림 없음
```

CircuitBreaker가 발동되면 모든 거래가 중단되는 중대 이벤트이지만, 로그에만 기록되고 운영자에게 즉시 알림이 가지 않는다.

**수정 방향:**
CircuitBreaker 자체에 notifier 의존성을 추가하는 것은 결합도가 높으므로, `check_all()`에서 CircuitBreaker 발동 시 알림을 보내는 방식이 낫다. 또는 main.py에서 `check_all` 실패 사유에 "Circuit Breaker"가 포함되면 별도 알림을 보낸다.

```python
# main.py — check_all 실패 처리부에 추가
if not ok:
    logger.warning(f"리스크 체크 실패: {reason}")
    notifier.send_sync(f"[경고] 리스크 체크 실패: {reason}")
    if "Circuit Breaker" in reason:
        notifier.send_sync(f"[긴급] Circuit Breaker 발동 — 수동 리셋 필요: {reason}")
```

현재 코드에 이미 `notifier.send_sync(f"[경고] 리스크 체크 실패: {reason}")`이 있지만, CircuitBreaker 발동은 "[경고]"가 아닌 "[긴급]" 수준이므로 별도 처리가 필요하다.

---

## LOW

### L-1. 미사용 import

- [x] 수정 완료

**파일:** `main.py:85-86`

**문제:**
```python
import ccxt       # 사용하지 않음
import pandas as pd  # 사용하지 않음
```

`ccxt`는 직접 사용하지 않고(collector와 executor가 내부적으로 사용), `pandas`도 `run_live()` 내에서 직접 사용하지 않는다.

**수정:** 두 import 라인 제거.

---

## 수정 완료 체크리스트

| 순번 | ID | 심각도 | 요약 | 상태 |
|------|----|--------|------|------|
| 1 | C-1 | CRITICAL | atr_14 미계산 → 변동성/사이징 오작동 | [x] |
| 2 | C-2 | CRITICAL | 포지션 사이즈 단위 불일치 (수량 vs 금액) | [x] |
| 3 | C-3 | CRITICAL | 심볼 파싱 3글자 base만 지원 | [x] |
| 4 | H-1 | HIGH | 재시작 시 prev_positions 미복원 | [x] |
| 5 | H-2 | HIGH | fetch_my_trades PnL 중복 집계 | [x] |
| 6 | H-3 | HIGH | Graceful Shutdown 상태 미저장 | [x] |
| 7 | M-1 | MEDIUM | close_position 멱등성 충돌 | [x] |
| 8 | M-2 | MEDIUM | CircuitBreaker 발동 텔레그램 미알림 | [x] |
| 9 | L-1 | LOW | 미사용 import (ccxt, pandas) | [x] |
