# AI Code Review 보고서

**리뷰 일자**: 2026-03-15
**리뷰 범위**: `preview` 브랜치 (30+ 커밋, main 대비) — 펀딩비 차익거래, 멀티 전략 포트폴리오, Hedge Mode 등 대규모 변경

---

## CRITICAL (즉시 수정 필요)

### 1. 주문 멱등성 실패 — 중복 주문 위험
**`src/execution/executor.py:108-111`**

`pending_orders`가 인메모리 딕셔너리만 확인. 거래소에 주문이 생성되었지만 네트워크 오류로 로컬 기록 실패 시 → 다음 호출에서 중복 주문 생성.

```
execute() → create_order() → 거래소 OK → 응답 유실 → pending_orders 미기록
→ 재호출 → 멱등성 체크 통과 → 중복 주문 발생
```

**수정**: `sync_positions()` 호출 후 거래소 실제 미체결 주문과 동기화 필수.

---

### 2. 차익거래 롤백 시 부분 체결 미처리
**`src/execution/arb_executor.py:99-111`**

현물 매수 실패 시 선물 숏 롤백에서 **원래 주문 수량**으로 청산 시도. 선물이 부분 체결(예: 0.5 BTC)되었으면 **실제 체결량(`filled`)**으로 청산해야 함.

```python
# 현재 (위험)
self.perp.execute(..., amount=amount)
# 수정
actual_filled = float(perp_order.get("filled", 0))
if actual_filled > 0:
    self.perp.execute(..., amount=actual_filled)
```

---

### 3. `check_portfolio()` 미호출 — MDD 한도 무력화
**`src/portfolio/risk.py:82-103`**

`PortfolioRiskManager.check_portfolio()`가 구현되어 있지만 `main.py`에서 **호출하지 않음**. 포트폴리오 MDD -10% 초과해도 거래 계속 가능.

---

### 4. 멀티타임프레임 피처의 미래정보 유출 가능성
**`strategies/_common/features.py:302, 315-342`**

`resample(rule, label='right', closed='right')` 후 `reindex(method='ffill')`로 1h 봉에 전파. 4h 봉 종료 시점의 데이터가 해당 구간 내 1h 봉에 노출될 수 있음.

**수정**: `label='left', closed='left'` 또는 `shift(1)` 적용.

---

### 5. 가상 포지션 ↔ 실제 주문 비동기
**`main.py` + `src/portfolio/virtual_position.py`**

가상 포지션을 **먼저 생성**하고 실제 주문을 실행. 주문 실패 시 가상 포지션 **롤백 없음** → 상태 불일치 → 이후 포지션 사이징/캡 계산 오류.

---

## HIGH (단기 수정 권장)

### 6. 무한 루프 예외 처리 — 복구 불가능 오류도 무시
**`main.py:1047-1048`**

`except Exception as e: logger.error(...)` 후 루프 재개. 포지션 동기화 실패, 상태 저장 오류 등 치명적 상황에서도 거래 계속.

**수정**: 네트워크 오류(재시도 가능) vs 프로그램 오류(종료 필요) 구분.

---

### 7. Slippage 계산 부호 반전
**`src/execution/arb_executor.py:125-127`**

```python
slippage_pct = (spot_fill - perp_fill) / spot_fill  # 부호 반대
```
차익거래(현물 매수 + 선물 숏)에서 현물을 더 비싸게 사면 **음수 슬리피지**여야 하는데 양수로 계산됨.

---

### 8. CSV Race Condition
**`src/execution/executor.py:247-250`**

`os.path.exists()` 체크와 CSV 쓰기 사이에 race condition. 동시 실행 시 헤더가 두 번 기록되어 CSV 손상.

---

### 9. 포지션 캡 계산 ZeroDivision
**`src/portfolio/manager.py:278-287`**

```python
scale = max(0, self.max_symbol_exposure - existing_pct) / new_pct
```
`new_pct`가 매우 작으면 비정상적 스케일링, 0이면 ZeroDivisionError.

---

### 10. Hedge Mode 전환 일부 실패 시 일괄 비활성화
**`main.py:493-564`**

BTC Hedge Mode 성공, ETH 실패 → `hedge_mode_enabled = False` → BTC도 One-Way로 강제. 심볼별 상태 추적 필요.

---

## MEDIUM

### 11. 펀딩비 차익거래 config 불일치
**`strategies/funding_arb/strategy.py:37-41` vs `config.yaml:7-9`**

config.yaml은 다중 심볼(`symbols` 배열) 정의, strategy.py는 단일 심볼(`symbol_spot` 기본값 BTC) 사용. ETH 차익거래가 BTC로 실행될 가능성.

### 12. SpotExecutor 미체결 주문 추적 부재
**`src/execution/spot_executor.py` 전체**

`OrderExecutor`와 달리 `pending_orders` 없음. 현물 주문 미체결 상태에서 재시작 시 추적 불가.

### 13. 네트워크 타임아웃 기본값 의존
**`main.py:701-703, 717, 793`**

`fetch_ohlcv_bulk()`, `fetch_balance()` 등에 명시적 타임아웃 없음. 네트워크 단절 시 무한 대기.

### 14. 상태 복원 일관성
**`main.py:612-625`**

여러 객체(`risk_manager`, `pnl_tracker`, `virtual_tracker` 등)의 상태를 개별 `to_dict()/from_dict()`로 관리. 부분 저장/복원 시 일관성 깨짐.

### 15. 포지션 노출도 근사치 문제
**`src/portfolio/manager.py:267-275`**

`size × entry_price`로 노출도 계산. 현재 시장가격 미반영으로 실제 위험도 과소/과대 평가.

---

## LOW / INFO

### 16. 동시 포지션 한도 `break` 로직
**`src/portfolio/manager.py:185-188`**

`break`로 이후 모든 전략 건너뜀. 신호 순서에 따라 할당 결과 불공평.

### 17. PnL 분배 부동소수점 누적 오차
매우 큰 금액에서 미세한 불일치 가능. 마지막 전략에 나머지 할당 패턴 권장.

### 18. CircuitBreaker 복구 조건 부재
발동 후 수동 리셋만 가능. 최소 대기시간/재평가 메커니즘 고려.

---

## 종합 점수

| 영역 | 점수 | 주요 이슈 |
|------|------|----------|
| **주문 실행 안전성** | 60/100 | 멱등성, 롤백, CSV race |
| **리스크 관리** | 70/100 | check_portfolio 미호출, 캡 계산 |
| **ML 파이프라인** | 85/100 | 멀티타임프레임 피처 누수 |
| **데이터 안전성** | 95/100 | Atomic write 적용 양호 |
| **포트폴리오 배분** | 80/100 | 가상포지션 롤백, 노출도 근사치 |
| **설정 일관성** | 75/100 | 펀딩비 전략 config 불일치 |

**종합: 77/100** — 견고한 아키텍처이나 실거래 안전성에 중요한 결함 다수 존재. CRITICAL 항목 5건 우선 수정 권장.

---

## 수정 우선순위 로드맵

### 즉시 (1주 이내)
1. 주문 멱등성: 거래소 미체결 주문 동기화 추가
2. 차익거래 롤백: 체결량(`filled`) 기반 청산
3. `check_portfolio()` 호출 추가
4. 가상 포지션 롤백 메커니즘 도입
5. 무한 루프 예외 분류 (네트워크 vs 프로그램 오류)

### 단기 (1개월)
6. 멀티타임프레임 피처 look-ahead 수정 + 재학습
7. Slippage 부호 수정
8. CSV atomic write 적용
9. 포지션 캡 ZeroDivision 방어
10. Hedge Mode 심볼별 상태 추적

### 중기 (분기)
11. 펀딩비 차익거래 config/코드 일관성
12. SpotExecutor 주문 추적
13. 네트워크 타임아웃 명시화
14. 중앙 상태 관리자 (StateManager) 도입
15. 포지션 노출도 현재가 반영
