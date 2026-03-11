---
name: verify-portfolio
description: 포트폴리오 레이어(자본 배분, 리스크, 가상 포지션) 규칙 준수 여부를 검증합니다. 포트폴리오 설정 변경 또는 전략 추가/제거 후 사용.
---

# 포트폴리오 레이어 검증

## Purpose

1. **portfolio.yaml ↔ CLAUDE.md 동기화** — 자본 배분/리스크 파라미터가 CLAUDE.md 문서와 일치하는지 확인
2. **active_strategies 정합성** — 목록의 각 전략에 strategy.py와 config.yaml이 존재하는지 확인
3. **자본 배분 한도 일관성** — PortfolioManager 기본값이 portfolio.yaml 값을 정확히 로드하는지 확인
4. **VirtualPositionTracker 직렬화 완전성** — to_dict()/from_dict() 왕복 무결성 확인
5. **동시 포지션 한도 적용** — max_concurrent_positions가 allocate()에서 올바르게 적용되는지 확인
6. **동적 전략 로드** — load_strategies_from_config()가 BaseStrategy 서브클래스를 정확히 찾는지 확인

## When to Run

- `config/portfolio.yaml`을 수정한 후
- `src/portfolio/manager.py`, `risk.py`, `virtual_position.py`를 수정한 후
- 새로운 전략을 `active_strategies`에 추가한 후
- CLAUDE.md의 리스크 파라미터 섹션을 수정한 후
- `main.py`에서 PortfolioManager 사용 흐름을 변경한 후

## Related Files

| File | Purpose |
|------|---------|
| `config/portfolio.yaml` | 포트폴리오 설정 (전략 목록, 배분, 한도, 리스크) |
| `src/portfolio/manager.py` | PortfolioManager (전략 등록, 시그널 수집, 자본 배분) |
| `src/portfolio/risk.py` | PortfolioRiskManager (MDD, 일일 손실, 전략 비활성화) |
| `src/portfolio/virtual_position.py` | VirtualPositionTracker (전략별 가상 포지션 추적) |
| `src/portfolio/__init__.py` | 패키지 init (__all__ 내보내기) |
| `src/strategies/base.py` | BaseStrategy (동적 로드 시 서브클래스 탐색 대상) |
| `main.py` | 실거래 루프 (PortfolioManager 사용) |
| `CLAUDE.md` | 프로젝트 지침 (리스크 파라미터 문서) |

## Workflow

### Step 1: portfolio.yaml ↔ CLAUDE.md 값 동기화

**파일:** `config/portfolio.yaml`, `CLAUDE.md`

**도구:** Read

두 파일을 읽고 다음 값이 일치하는지 대조합니다:

| 파라미터 | CLAUDE.md 문서 값 | portfolio.yaml 키 |
|----------|-------------------|-------------------|
| 전략당 포지션 | 20% | `allocation.position_pct_per_strategy: 0.20` |
| 전체 노출 | 60% | `limits.max_total_exposure: 0.60` |
| 동일 심볼 | 30% | `limits.max_symbol_exposure: 0.30` |
| 포트폴리오 MDD | 10% | `risk.max_portfolio_mdd: -0.10` |
| 일손실 | 3% | `risk.max_daily_loss: -0.03` |

CLAUDE.md의 "리스크 파라미터" 섹션에서 값을 추출합니다:
```
전략당 포지션 20%, 전체 노출 60%, 동일 심볼 30%. 일손실 3%/포트폴리오 MDD 10%.
```

**PASS:** 모든 값이 CLAUDE.md에 문서화된 값과 일치
**FAIL:** 불일치하는 값이 존재

**수정:** portfolio.yaml 또는 CLAUDE.md 중 올바른 쪽을 업데이트 (사용자에게 확인)

### Step 2: active_strategies 정합성 검증

**파일:** `config/portfolio.yaml`, `strategies/*/`

**도구:** Read, Bash

portfolio.yaml의 `active_strategies` 목록을 읽고, 각 전략에 대해 필수 파일이 존재하는지 확인합니다:

```bash
# portfolio.yaml에서 active_strategies 목록 추출 후 각 전략 확인
for strategy in $(grep -A 20 "active_strategies:" config/portfolio.yaml | grep "^ *- " | sed 's/^ *- //'); do
  ls "strategies/$strategy/strategy.py" 2>/dev/null || echo "MISSING: strategies/$strategy/strategy.py"
  ls "strategies/$strategy/config.yaml" 2>/dev/null || echo "MISSING: strategies/$strategy/config.yaml"
done
```

**PASS:** 모든 전략에 strategy.py와 config.yaml이 존재
**FAIL:** 누락된 파일이 있음

**수정:** 누락된 전략 폴더/파일을 생성하거나, active_strategies에서 해당 전략을 제거

### Step 3: PortfolioManager 기본값과 portfolio.yaml 일관성

**파일:** `src/portfolio/manager.py`, `config/portfolio.yaml`

**도구:** Read, Grep

PortfolioManager.__init__()의 `.get()` 기본값이 portfolio.yaml에 설정된 값과 동일한지 확인합니다:

```bash
# PortfolioManager의 기본값 추출
Grep pattern="\.get\(\"(position_pct_per_strategy|max_total_exposure|max_symbol_exposure|max_concurrent_positions)\"" path="src/portfolio/manager.py" output_mode="content" -n=true
```

대조 항목:

| 파라미터 | manager.py 기본값 | portfolio.yaml 값 |
|----------|-------------------|-------------------|
| position_pct_per_strategy | 0.20 | 0.20 |
| max_total_exposure | 0.60 | 0.60 |
| max_symbol_exposure | 0.30 | 0.30 |
| max_concurrent_positions | 5 | 5 |

PortfolioRiskManager.__init__()도 동일하게 확인:

```bash
Grep pattern="\.get\(\"(max_portfolio_mdd|max_daily_loss|strategy_disable_threshold|strategy_disable_min_trades)\"" path="src/portfolio/risk.py" output_mode="content" -n=true
```

| 파라미터 | risk.py 기본값 | portfolio.yaml 값 |
|----------|---------------|-------------------|
| max_portfolio_mdd | -0.10 | -0.10 |
| max_daily_loss | -0.03 | -0.03 |
| strategy_disable_threshold | 0.8 | 0.8 |
| strategy_disable_min_trades | 50 | 50 |

**PASS:** 모든 기본값이 portfolio.yaml 값과 일치
**FAIL:** 기본값과 설정값이 불일치

**수정:** manager.py/risk.py의 기본값을 portfolio.yaml에 맞게 업데이트

### Step 4: VirtualPositionTracker 직렬화 완전성

**파일:** `src/portfolio/virtual_position.py`

**도구:** Read

VirtualPositionTracker의 to_dict()/from_dict() 메서드를 읽고 다음을 확인:

1. `to_dict()`가 `self.virtual_positions`를 반환하는지
2. `from_dict(data)`가 `self.virtual_positions = data`로 완전 복원하는지
3. 왕복(roundtrip) 무결성: `tracker.from_dict(tracker.to_dict())` 후 상태가 동일한지

PortfolioRiskManager의 to_dict()/from_dict()도 동일하게 확인:

1. `to_dict()`가 `{"strategy_stats": self.strategy_stats}`를 반환하는지
2. `from_dict(data)`가 `data.get("strategy_stats", {})`로 복원하는지
3. to_dict() 키 집합과 from_dict() 복원 키 집합이 일치하는지

```bash
# to_dict/from_dict 메서드 확인
Grep pattern="def (to_dict|from_dict)" path="src/portfolio/virtual_position.py" output_mode="content" -n=true
Grep pattern="def (to_dict|from_dict)" path="src/portfolio/risk.py" output_mode="content" -n=true
```

**PASS:** 모든 직렬화/역직렬화 메서드가 완전하고, 키 집합이 일치
**FAIL:** 누락된 필드가 있거나 키 집합 불일치

**수정:** to_dict()에 누락된 필드를 추가하거나, from_dict()에 복원 로직 추가

### Step 5: 동시 포지션 한도 적용 검증

**파일:** `src/portfolio/manager.py`

**도구:** Read

PortfolioManager.allocate() 메서드를 읽고 다음을 확인:

1. `virtual_tracker.get_all_symbols()`로 현재 포지션 수를 계산하는지
2. `self.max_concurrent_positions`와 비교하여 `remaining_slots`를 산출하는지
3. `remaining_slots <= 0`일 때 새 주문을 차단하는지
4. 각 주문 추가 후 `remaining_slots`를 감소시키는지

```bash
# allocate 메서드에서 동시 포지션 한도 관련 코드 확인
Grep pattern="(max_concurrent_positions|remaining_slots|get_all_symbols)" path="src/portfolio/manager.py" output_mode="content" -n=true
```

**PASS:** max_concurrent_positions가 allocate()에서 올바르게 적용됨
**FAIL:** 동시 포지션 한도가 적용되지 않거나 우회 가능한 경로 존재

**수정:** allocate()에 동시 포지션 한도 체크 로직 추가

### Step 6: 동적 전략 로드 검증

**파일:** `src/portfolio/manager.py`, `src/strategies/base.py`

**도구:** Read, Grep

load_strategies_from_config() 메서드를 읽고 다음을 확인:

1. `importlib.import_module(f"strategies.{strategy_name}.strategy")`로 동적 import하는지
2. `inspect.getmembers(module, inspect.isclass)`로 클래스를 탐색하는지
3. `issubclass(obj, BaseStrategy) and obj is not BaseStrategy`로 서브클래스를 식별하는지
4. 서브클래스를 찾지 못하면 `ValueError`를 발생시키는지
5. 전략 폴더/config.yaml이 없으면 `FileNotFoundError`를 발생시키는지

```bash
# 동적 import 관련 코드 확인
Grep pattern="(importlib|import_module|inspect|issubclass|BaseStrategy)" path="src/portfolio/manager.py" output_mode="content" -n=true
```

**PASS:** 동적 전략 로드가 BaseStrategy 서브클래스를 정확히 찾고, 에러 처리가 적절
**FAIL:** import/탐색 로직이 불완전하거나 에러 처리 누락

### Step 7: __init__.py 내보내기 검증

**파일:** `src/portfolio/__init__.py`

**도구:** Read

`__all__`에 PortfolioManager, PortfolioRiskManager, VirtualPositionTracker가 모두 포함되어 있는지 확인합니다.

```bash
Grep pattern="__all__" path="src/portfolio/__init__.py" output_mode="content" -n=true
```

**PASS:** 3개 클래스가 모두 __all__에 포함
**FAIL:** 누락된 클래스가 있음

**수정:** __all__에 누락된 클래스 추가

### Step 8: 심볼 캡/전체 노출 캡 로직 검증

**파일:** `src/portfolio/manager.py`

**도구:** Read

_apply_symbol_cap()과 _apply_total_cap() 메서드를 읽고 다음을 확인:

1. _apply_symbol_cap(): 기존 가상 포지션의 노출 + 신규 주문 합산이 max_symbol_exposure를 초과 시 비례 축소하는지
2. _apply_total_cap(): 기존 전체 노출 + 신규 주문 합산이 max_total_exposure를 초과 시 비례 축소하는지
3. 축소 비율(scale)이 음수가 되지 않도록 `max(0, ...)` 처리가 되어 있는지

```bash
Grep pattern="(_apply_symbol_cap|_apply_total_cap|max\(0)" path="src/portfolio/manager.py" output_mode="content" -n=true
```

**PASS:** 두 캡 메서드가 비례 축소를 올바르게 적용하고, scale이 음수 방지됨
**FAIL:** 비례 축소 로직 오류 또는 음수 scale 가능성

## Output Format

```markdown
### verify-portfolio 검증 결과

| # | 검사 항목 | 결과 | 상세 |
|---|-----------|------|------|
| 1 | portfolio.yaml ↔ CLAUDE.md 동기화 | PASS/FAIL | 불일치 항목... |
| 2 | active_strategies 정합성 | PASS/FAIL | 누락 파일... |
| 3 | 기본값 ↔ portfolio.yaml 일관성 | PASS/FAIL | 불일치 항목... |
| 4 | VirtualPositionTracker 직렬화 | PASS/FAIL | 누락 필드... |
| 5 | 동시 포지션 한도 적용 | PASS/FAIL | 코드 흐름... |
| 6 | 동적 전략 로드 | PASS/FAIL | import/탐색... |
| 7 | __init__.py 내보내기 | PASS/FAIL | 누락 클래스... |
| 8 | 심볼/전체 노출 캡 로직 | PASS/FAIL | 비례 축소... |
```

## Exceptions

1. **백테스트 모드** — backtest.py에서는 PortfolioManager를 사용하지 않으므로, 백테스트 관련 코드에서 포트폴리오 한도 미적용은 위반이 아님
2. **전략 개발 중 임시 비활성화** — active_strategies에서 전략을 일시적으로 제거한 경우, 해당 전략 폴더가 존재해도 정합성 위반이 아님
3. **테스트 코드의 기본값 사용** — tests/ 디렉토리 내에서 PortfolioManager를 하드코딩된 설정으로 초기화하는 것은 허용
