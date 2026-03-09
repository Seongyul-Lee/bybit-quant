---
name: verify-risk
description: 리스크 관리 규칙 준수 여부를 검증합니다. 리스크 파라미터 변경 또는 실행 흐름 수정 후 사용.
---

# 리스크 관리 규칙 검증

## Purpose

1. **risk_params.yaml ↔ CLAUDE.md 동기화 검증** — 리스크 파라미터 값이 CLAUDE.md에 문서화된 값과 일치하는지 확인
2. **CircuitBreaker 자동 리셋 금지** — CircuitBreaker.reset()을 자동으로 호출하는 코드가 없는지 확인
3. **check_all 우선 실행** — 주문 실행 전 RiskManager.check_all()이 반드시 호출되는지 확인
4. **레버리지 제한** — 설정된 최대 레버리지를 초과하는 코드가 없는지 확인

## When to Run

- `config/risk_params.yaml`을 수정한 후
- `src/risk/manager.py`를 수정한 후
- `main.py`의 실거래 루프를 수정한 후
- CLAUDE.md의 리스크 파라미터 섹션을 수정한 후

## Related Files

| File | Purpose |
|------|---------|
| `config/risk_params.yaml` | 리스크 파라미터 설정 |
| `src/risk/manager.py` | RiskManager + CircuitBreaker 구현 |
| `main.py` | 실거래 루프 (check_all 호출 위치) |
| `CLAUDE.md` | 프로젝트 지침 (리스크 파라미터 문서) |

## Workflow

### Step 1: risk_params.yaml ↔ CLAUDE.md 값 동기화

**파일:** `config/risk_params.yaml`, `CLAUDE.md`

**도구:** Read

두 파일을 읽고 다음 값이 일치하는지 대조합니다:

| 파라미터 | CLAUDE.md 문서 값 | risk_params.yaml 키 |
|----------|-------------------|---------------------|
| 단일 포지션 최대 | 5% | `position.max_position_pct: 0.05` |
| 동시 최대 포지션 | 3개 | `position.max_concurrent_positions: 3` |
| 레버리지 | 3배 | `position.max_leverage: 3` |
| 일일 손실 한도 | 3% | `loss_limits.daily_loss_limit_pct: 0.03` |
| 월간 손실 한도 | 10% | `loss_limits.monthly_loss_limit_pct: 0.10` |
| 연속 손실 | 5회 | `circuit_breaker.max_consecutive_losses: 5` |
| 변동성 임계값 | 5% | `circuit_breaker.volatility_threshold: 0.05` |
| 기본 손절 | 2% | `trade.default_stop_loss_pct: 0.02` |
| 기본 익절 | 4% | `trade.default_take_profit_pct: 0.04` |
| 거래당 위험 | 1% | `trade.risk_per_trade_pct: 0.01` |

**PASS:** 모든 값이 CLAUDE.md에 문서화된 값과 일치
**FAIL:** 불일치하는 값이 존재

**수정:** risk_params.yaml 또는 CLAUDE.md 중 올바른 쪽을 업데이트 (사용자에게 확인)

### Step 2: CircuitBreaker 자동 리셋 금지 검증

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

### Step 3: check_all 우선 실행 검증

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

### Step 4: 레버리지 제한 검증

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

## Output Format

```markdown
### verify-risk 검증 결과

| # | 검사 항목 | 결과 | 상세 |
|---|-----------|------|------|
| 1 | risk_params ↔ CLAUDE.md 동기화 | PASS/FAIL | 불일치 항목... |
| 2 | CircuitBreaker 자동 리셋 금지 | PASS/FAIL | 발견된 자동 리셋... |
| 3 | check_all 우선 실행 | PASS/FAIL | 코드 흐름 분석... |
| 4 | 레버리지 제한 | PASS/FAIL | 초과 레버리지... |
```

## Exceptions

1. **테스트 코드의 reset()** — `tests/` 디렉토리 내 테스트 코드에서 setUp/tearDown에 reset()을 호출하는 것은 허용
2. **백테스트 모드의 check_all 생략** — backtest.py에서는 RiskManager.check_all()을 호출하지 않아도 됨 (시뮬레이션이므로)
3. **risk_params.yaml 주석의 값** — YAML 주석(`#` 이후)의 값은 검증 대상이 아님, 실제 YAML 값만 대조
