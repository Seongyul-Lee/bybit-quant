---
name: verify-risk
description: 리스크 관리 규칙 준수 여부를 검증합니다. 리스크 파라미터 변경 또는 실행 흐름 수정 후 사용.
---

# 리스크 관리 규칙 검증

## Purpose

1. **risk_params.yaml ↔ CLAUDE.md 동기화 검증** — 전략별 리스크 파라미터 값이 CLAUDE.md에 문서화된 값과 일치하는지 확인
2. **portfolio.yaml ↔ CLAUDE.md 동기화 검증** — 포트폴리오 리스크 파라미터가 CLAUDE.md와 일치하는지 확인
3. **PortfolioRiskManager 검증** — 포트폴리오 레벨 리스크 관리가 올바르게 구현/사용되는지 확인
4. **PortfolioManager 자본 배분 한도 검증** — 자본 배분 한도가 portfolio.yaml과 일치하는지 확인
5. **CircuitBreaker 자동 리셋 금지** — CircuitBreaker.reset()을 자동으로 호출하는 코드가 없는지 확인
6. **check_all 우선 실행** — 주문 실행 전 RiskManager.check_all()이 반드시 호출되는지 확인
7. **레버리지 제한** — 설정된 최대 레버리지를 초과하는 코드가 없는지 확인

## When to Run

- `config/risk_params.yaml`을 수정한 후
- `config/portfolio.yaml`을 수정한 후
- `src/risk/manager.py`를 수정한 후
- `src/portfolio/risk.py`를 수정한 후
- `src/portfolio/manager.py`를 수정한 후
- `main.py`의 실거래 루프를 수정한 후
- CLAUDE.md의 리스크 파라미터 섹션을 수정한 후

## Related Files

| File | Purpose |
|------|---------|
| `config/risk_params.yaml` | 전략별 리스크 파라미터 설정 |
| `config/portfolio.yaml` | 포트폴리오 리스크 설정 (배분, 한도, MDD) |
| `src/risk/manager.py` | RiskManager + CircuitBreaker + PnLTracker 구현 |
| `src/portfolio/risk.py` | PortfolioRiskManager (2계층 리스크의 상위 레이어) |
| `src/portfolio/manager.py` | PortfolioManager (자본 배분 한도) |
| `src/execution/executor.py` | _save_state (extra_state로 CB/PnLTracker 저장) |
| `main.py` | 실거래 루프 (check_all 호출 위치, PnLTracker 사용) |
| `CLAUDE.md` | 프로젝트 지침 (리스크 파라미터 문서) |

## Workflow

### Step 1: risk_params.yaml ↔ CLAUDE.md 값 동기화

**파일:** `config/risk_params.yaml`, `CLAUDE.md`

**도구:** Read

두 파일을 읽고 다음 값이 일치하는지 대조합니다:

| 파라미터 | CLAUDE.md 문서 값 | risk_params.yaml 키 |
|----------|-------------------|---------------------|
| 일일 손실 한도 | 3% | `loss_limits.daily_loss_limit_pct: 0.03` |
| 연속 손실 | 5회 | `circuit_breaker.max_consecutive_losses: 5` |
| 레버리지 | 3배 | `position.max_leverage: 3` |
| 변동성 임계값 | 5% | `circuit_breaker.volatility_threshold: 0.05` |
| 기본 손절 | 2% | `trade.default_stop_loss_pct: 0.02` |
| 기본 익절 | 4% | `trade.default_take_profit_pct: 0.04` |
| 거래당 위험 | 1% | `trade.risk_per_trade_pct: 0.01` |

**참고:** 포트폴리오 관련 파라미터(전략당 포지션, 전체 노출, 동일 심볼, 포트폴리오 MDD)는 Step 2에서 portfolio.yaml로 검증합니다.

**PASS:** 모든 값이 CLAUDE.md에 문서화된 값과 일치
**FAIL:** 불일치하는 값이 존재

**수정:** risk_params.yaml 또는 CLAUDE.md 중 올바른 쪽을 업데이트 (사용자에게 확인)

### Step 2: portfolio.yaml ↔ CLAUDE.md 동기화 검증

**파일:** `config/portfolio.yaml`, `CLAUDE.md`

**도구:** Read

portfolio.yaml의 값과 CLAUDE.md의 리스크 파라미터 섹션을 대조합니다:

| 파라미터 | CLAUDE.md 문서 값 | portfolio.yaml 키 |
|----------|-------------------|-------------------|
| 전략당 포지션 | 20% | `portfolio.allocation.position_pct_per_strategy: 0.20` |
| 전체 노출 | 60% | `portfolio.limits.max_total_exposure: 0.60` |
| 동일 심볼 | 30% | `portfolio.limits.max_symbol_exposure: 0.30` |
| 포트폴리오 MDD | 10% | `portfolio.risk.max_portfolio_mdd: -0.10` |
| 일손실 | 3% | `portfolio.risk.max_daily_loss: -0.03` |

**PASS:** 모든 값이 CLAUDE.md에 문서화된 값과 일치
**FAIL:** 불일치하는 값이 존재

**수정:** portfolio.yaml 또는 CLAUDE.md 중 올바른 쪽을 업데이트 (사용자에게 확인)

### Step 3: PortfolioRiskManager 검증

**파일:** `src/portfolio/risk.py`, `config/portfolio.yaml`, `main.py`

**도구:** Read, Grep

```bash
# PortfolioRiskManager 클래스 확인
Grep pattern="class PortfolioRiskManager" path="src/portfolio/risk.py" output_mode="content"

# check_portfolio 메서드 확인
Grep pattern="def check_portfolio" path="src/portfolio/risk.py" output_mode="content"

# check_strategy_health 메서드 확인
Grep pattern="def check_strategy_health" path="src/portfolio/risk.py" output_mode="content"

# to_dict/from_dict 확인
Grep pattern="def (to_dict|from_dict)" path="src/portfolio/risk.py" output_mode="content"

# record_trade 확인
Grep pattern="def record_trade" path="src/portfolio/risk.py" output_mode="content"

# main.py에서 PortfolioRiskManager 사용 확인
Grep pattern="PortfolioRiskManager|portfolio_risk" path="main.py" output_mode="content"
```

**검증:**
1. `PortfolioRiskManager.__init__`의 기본값이 portfolio.yaml risk 섹션 값과 일치하는지:
   - `max_portfolio_mdd` 기본값 = `-0.10` (portfolio.yaml: `-0.10`)
   - `max_daily_loss` 기본값 = `-0.03` (portfolio.yaml: `-0.03`)
   - `strategy_disable_threshold` 기본값 = `0.8` (portfolio.yaml: `0.8`)
   - `strategy_disable_min_trades` 기본값 = `50` (portfolio.yaml: `50`)
2. `check_portfolio()`: MDD 한도 체크 로직이 존재하는지
3. `check_strategy_health()`: 전략별 PF 기반 비활성화 로직이 존재하는지
4. `to_dict()`/`from_dict()` 직렬화가 올바르게 구현되는지 (키 집합 일치)
5. `record_trade()`로 실전 성과를 추적하는지
6. main.py에서 PortfolioRiskManager를 생성하고 사용하는지

**PASS:** 기본값이 portfolio.yaml과 일치하고, 모든 메서드가 구현되어 있으며, main.py에서 사용됨
**FAIL:** 기본값 불일치, 메서드 미구현, 또는 main.py에서 미사용

### Step 4: PortfolioManager 자본 배분 한도 검증

**파일:** `src/portfolio/manager.py`, `config/portfolio.yaml`

**도구:** Read, Grep

```bash
# PortfolioManager 초기화 확인
Grep pattern="class PortfolioManager" path="src/portfolio/manager.py" output_mode="content" -A=20

# _apply_symbol_cap 메서드 확인
Grep pattern="def _apply_symbol_cap" path="src/portfolio/manager.py" output_mode="content"

# _apply_total_cap 메서드 확인
Grep pattern="def _apply_total_cap" path="src/portfolio/manager.py" output_mode="content"

# max_symbol_exposure 사용 확인
Grep pattern="max_symbol_exposure" path="src/portfolio/manager.py" output_mode="content"

# max_total_exposure 사용 확인
Grep pattern="max_total_exposure" path="src/portfolio/manager.py" output_mode="content"
```

**검증:**
1. `PortfolioManager.__init__`에서 portfolio.yaml의 limits 값을 올바르게 로드하는지:
   - `position_pct` ← `allocation.position_pct_per_strategy` (기본값 `0.20`)
   - `max_total_exposure` ← `limits.max_total_exposure` (기본값 `0.60`)
   - `max_symbol_exposure` ← `limits.max_symbol_exposure` (기본값 `0.30`)
   - `max_concurrent_positions` ← `limits.max_concurrent_positions` (기본값 `5`)
2. `_apply_symbol_cap()`이 `self.max_symbol_exposure`를 사용하여 동일 심볼 합산 노출을 제한하는지
3. `_apply_total_cap()`이 `self.max_total_exposure`를 사용하여 전체 노출을 제한하는지
4. 기본값이 portfolio.yaml의 실제 값과 일치하는지

**PASS:** 한도 값이 portfolio.yaml과 일치하고, cap 메서드가 올바르게 적용됨
**FAIL:** 기본값 불일치, 또는 cap 메서드에서 한도를 올바르게 사용하지 않음

### Step 5: CircuitBreaker 자동 리셋 금지 검증

**파일:** 전체 Python 소스

**도구:** Grep

```bash
# reset() 호출 탐색 (수동 리셋만 허용)
Grep pattern="\.reset\(\)" glob="*.py" path="src/"
Grep pattern="\.reset\(\)" glob="*.py" path="strategies/"
Grep pattern="circuit_breaker.*reset" glob="*.py" path="src/"
```

발견된 각 호출을 읽고, 자동 호출인지 수동 호출인지 판별합니다:
- **자동 호출 금지:** 타이머, 스케줄러, 조건부 자동 리셋 (예: `if time > X: cb.reset()`)
- **수동 허용:** 테스트 코드의 setUp/tearDown, CLI 명령, 명시적 수동 리셋 함수

```bash
# main.py에서 reset 호출 확인
Grep pattern="reset" path="main.py" output_mode="content"
```

**PASS:** 자동 리셋 코드가 없음
**FAIL:** 자동으로 reset()을 호출하는 코드가 존재

**수정:** 자동 리셋 코드를 제거하고, 필요 시 수동 리셋 CLI 명령으로 대체

### Step 6: check_all 우선 실행 검증

**파일:** `main.py`

**도구:** Read

`run_live()` 함수를 읽고, `executor.execute()` 호출 전에 반드시 `risk_manager.check_all()`이 호출되는지 확인합니다.

코드 흐름에서 다음을 확인:
1. `check_all()` 호출이 `execute()` 호출보다 앞에 있는지
2. `check_all()`이 False를 반환하면 `execute()`를 건너뛰는 로직이 있는지
3. `check_all()`을 우회하는 경로가 없는지

```bash
# check_all 호출 확인
Grep pattern="check_all" path="main.py" output_mode="content" -n=true

# execute 호출 확인
Grep pattern="executor.execute" path="main.py" output_mode="content" -n=true
```

**PASS:** check_all() → 통과 시에만 → execute() 순서가 보장됨
**FAIL:** check_all() 없이 execute()가 호출되거나, check_all 실패 시에도 execute가 실행될 수 있는 경로

**수정:** execute() 호출 전에 check_all() 가드를 추가

### Step 7: 레버리지 제한 검증

**파일:** 전체 Python 소스

**도구:** Grep

```bash
# 레버리지 설정 관련 코드 탐색
Grep pattern="leverage|set_leverage" glob="*.py" path="src/"
Grep pattern="leverage|set_leverage" glob="*.py" path="main.py"
```

발견된 레버리지 관련 코드에서 `risk_params.yaml`의 `max_leverage` 값(3)을 초과하는 하드코딩이 없는지 확인합니다.

**PASS:** 레버리지가 설정 파일의 max_leverage로 제한되거나, 레버리지 관련 코드가 없음
**FAIL:** max_leverage를 초과하는 하드코딩된 레버리지 값이 존재

### Step 8: PnLTracker 일관성 검증

**파일:** `src/risk/manager.py`, `main.py`

**도구:** Read, Grep

```bash
# PnLTracker 클래스 존재 확인
Grep pattern="class PnLTracker" path="src/risk/manager.py" output_mode="content"

# to_dict/from_dict 메서드 존재 확인
Grep pattern="def (to_dict|from_dict)" path="src/risk/manager.py" output_mode="content"

# main.py에서 PnLTracker 사용 확인
Grep pattern="PnLTracker|pnl_tracker" path="main.py" output_mode="content"
```

**검증:**
1. PnLTracker 클래스가 `to_dict()`/`from_dict()` 직렬화 메서드를 구현하는지
2. `to_dict()`가 반환하는 키 집합과 `from_dict()`가 복원하는 키 집합이 일치하는지
3. main.py에서 PnLTracker를 생성하고 `record_pnl()`로 거래 PnL을 기록하는지
4. main.py에서 `check_all()`에 `monthly_pnl=pnl_tracker.monthly_pnl`을 전달하는지

**PASS:** PnLTracker 직렬화가 완전하고, main.py에서 올바르게 사용
**FAIL:** 직렬화 키 불일치, 또는 main.py에서 check_all에 monthly_pnl 미전달

### Step 9: CircuitBreaker 상태 저장/복원 검증

**파일:** `src/risk/manager.py`, `main.py`

**도구:** Read, Grep

```bash
# CircuitBreaker to_dict/from_dict 확인
Grep pattern="class CircuitBreaker" path="src/risk/manager.py" output_mode="content" -A=5
Grep pattern="def (to_dict|from_dict)" path="src/risk/manager.py" output_mode="content"

# main.py에서 상태 복원 확인
Grep pattern="from_dict|circuit_breaker" path="main.py" output_mode="content"
```

**검증:**
1. CircuitBreaker가 `to_dict()`/`from_dict()` 메서드를 구현하는지
2. `to_dict()`가 `consecutive_losses`와 `is_tripped`를 포함하는지
3. main.py에서 프로세스 시작 시 `current_state.json`에서 CircuitBreaker 상태를 `from_dict()`로 복원하는지
4. main.py에서 매 순환마다 `_save_state(extra_state={..., "circuit_breaker": cb.to_dict(), ...})`로 저장하는지

**PASS:** CircuitBreaker 상태가 프로세스 재시작 시 유지됨
**FAIL:** to_dict/from_dict 미구현, 또는 main.py에서 상태 저장/복원 누락

## Output Format

```markdown
### verify-risk 검증 결과

| # | 검사 항목 | 결과 | 상세 |
|---|-----------|------|------|
| 1 | risk_params ↔ CLAUDE.md 동기화 | PASS/FAIL | 불일치 항목... |
| 2 | portfolio.yaml ↔ CLAUDE.md 동기화 | PASS/FAIL | 불일치 항목... |
| 3 | PortfolioRiskManager 검증 | PASS/FAIL | 기본값/메서드/사용 상태... |
| 4 | PortfolioManager 자본 배분 한도 | PASS/FAIL | 한도 일치/cap 적용... |
| 5 | CircuitBreaker 자동 리셋 금지 | PASS/FAIL | 발견된 자동 리셋... |
| 6 | check_all 우선 실행 | PASS/FAIL | 코드 흐름 분석... |
| 7 | 레버리지 제한 | PASS/FAIL | 초과 레버리지... |
| 8 | PnLTracker 일관성 | PASS/FAIL | 직렬화/사용 상태... |
| 9 | CircuitBreaker 상태 저장/복원 | PASS/FAIL | to_dict/from_dict... |
```

## Exceptions

1. **테스트 코드의 reset()** — `tests/` 디렉토리 내 테스트 코드에서 setUp/tearDown에 reset()을 호출하는 것은 허용
2. **백테스트 모드의 check_all 생략** — backtest.py에서는 RiskManager.check_all()을 호출하지 않아도 됨 (시뮬레이션이므로)
3. **risk_params.yaml 주석의 값** — YAML 주석(`#` 이후)의 값은 검증 대상이 아님, 실제 YAML 값만 대조
