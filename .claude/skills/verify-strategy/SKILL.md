---
name: verify-strategy
description: 전략 컨벤션 준수 여부를 검증합니다. 새 전략 추가 또는 기존 전략 수정 후 사용.
---

# 전략 컨벤션 검증

## Purpose

1. **BaseStrategy 상속 검증** — 모든 전략이 BaseStrategy를 상속하고 generate_signal(df) → int를 구현하는지 확인
2. **config.yaml 구조 검증** — 각 전략 폴더에 올바른 구조의 config.yaml이 존재하는지 확인
3. **엔트리포인트 등록 검증** — main.py와 backtest.py에 전략 import 분기가 등록되어 있는지 확인
4. **피처 재활용 패턴 검증** — `_get_or_compute_*` 패턴으로 processor 피처를 재활용하는지 확인
5. **거래소 직접 의존 금지** — 전략 코드에서 ccxt를 직접 import하지 않는지 확인

## When to Run

- 새 전략을 추가한 후
- 기존 전략의 generate_signal 로직을 수정한 후
- main.py 또는 backtest.py의 전략 로드 로직을 변경한 후
- 전략의 config.yaml을 수정한 후

## Related Files

| File | Purpose |
|------|---------|
| `src/strategies/base.py` | BaseStrategy ABC 정의 |
| `strategies/ma_crossover/strategy.py` | 참조 전략 구현체 |
| `strategies/ma_crossover/config.yaml` | 참조 전략 설정 |
| `main.py` | 실거래 진입점 (load_strategy 함수) |
| `backtest.py` | 백테스트 진입점 (run 함수 내 전략 분기) |

## Workflow

### Step 1: 전략 폴더 탐색

**도구:** Glob

```
strategies/*/strategy.py
```

발견된 각 전략 폴더를 목록화합니다.

### Step 2: BaseStrategy 상속 검증

**파일:** 각 `strategies/*/strategy.py`

**도구:** Grep

```bash
# 각 전략 파일에서 BaseStrategy import 확인
Grep pattern="from src.strategies.base import BaseStrategy" path="strategies/" glob="*/strategy.py"
```

```bash
# 각 전략 파일에서 BaseStrategy 상속 확인
Grep pattern="class \w+\(BaseStrategy\)" path="strategies/" glob="*/strategy.py"
```

**PASS:** 모든 전략 파일이 BaseStrategy를 import하고 상속함
**FAIL:** BaseStrategy를 상속하지 않는 전략 클래스가 존재

**수정:** `from src.strategies.base import BaseStrategy`를 추가하고, 클래스 정의를 `class MyStrategy(BaseStrategy):`로 변경

### Step 3: generate_signal 구현 검증

**파일:** 각 `strategies/*/strategy.py`

**도구:** Grep

```bash
# generate_signal 메서드 정의 확인
Grep pattern="def generate_signal\(self, df.*DataFrame\).*int" path="strategies/" glob="*/strategy.py"
```

**PASS:** 모든 전략이 `generate_signal(self, df: pd.DataFrame) -> int` 시그니처를 갖춤
**FAIL:** generate_signal 메서드가 없거나 시그니처가 다름

**수정:** `def generate_signal(self, df: pd.DataFrame) -> int:` 메서드 추가/수정

### Step 4: generate_signal 반환값 검증

**파일:** 각 `strategies/*/strategy.py`

**도구:** Read + 코드 분석

각 전략의 generate_signal 메서드를 읽고, return 문이 1, -1, 0 중 하나만 반환하는지 확인합니다.

```bash
# return 문 확인
Grep pattern="return" path="strategies/" glob="*/strategy.py" output_mode="content"
```

**PASS:** 모든 return 문이 1, -1, 0, 또는 signal 변수(이전에 1/-1로 할당된)를 반환
**FAIL:** 1, -1, 0 이외의 값을 반환하는 return 문이 존재

### Step 5: config.yaml 구조 검증

**파일:** 각 `strategies/*/config.yaml`

**도구:** Read

각 config.yaml을 읽고 필수 섹션이 있는지 확인합니다:
- `strategy.name` — 전략 클래스 이름
- `strategy.symbol` — 심볼 (BTCUSDT 형식)
- `strategy.timeframe` — 타임프레임
- `params` — 전략 파라미터
- `execution` — 실행 설정
- `risk` — 리스크 설정

```bash
# 필수 키 존재 확인
Grep pattern="^strategy:" path="strategies/" glob="*/config.yaml"
Grep pattern="^params:" path="strategies/" glob="*/config.yaml"
Grep pattern="^execution:" path="strategies/" glob="*/config.yaml"
Grep pattern="^risk:" path="strategies/" glob="*/config.yaml"
```

**PASS:** 모든 config.yaml이 strategy, params, execution, risk 섹션을 포함
**FAIL:** 필수 섹션이 누락된 config.yaml 존재

### Step 6: 엔트리포인트 등록 검증

**파일:** `main.py`, `backtest.py`

**도구:** Read

Step 1에서 발견된 각 전략 이름이 main.py의 `load_strategy()` 함수와 backtest.py의 `run()` 함수에 import 분기로 등록되어 있는지 확인합니다.

```bash
# main.py에서 전략 분기 확인
Grep pattern="if strategy_name ==" path="main.py" output_mode="content"

# backtest.py에서 전략 분기 확인
Grep pattern="if strategy_name ==" path="backtest.py" output_mode="content"
```

**PASS:** 모든 전략 폴더에 대응하는 import 분기가 양쪽 파일에 존재
**FAIL:** 전략 폴더는 있지만 main.py 또는 backtest.py에 등록되지 않음

**수정:** 해당 전략의 elif 분기를 main.py:load_strategy()와 backtest.py:run()에 추가

### Step 7: ccxt 직접 의존 금지 검증

**파일:** 각 `strategies/*/strategy.py`

**도구:** Grep

```bash
# 전략 파일에서 ccxt import 확인
Grep pattern="import ccxt" path="strategies/" glob="*/strategy.py"
```

**PASS:** 전략 파일에서 ccxt import가 없음
**FAIL:** 전략 파일에서 ccxt를 직접 import

**수정:** ccxt 의존 코드를 제거하고, 필요한 데이터는 df 파라미터를 통해 전달받도록 수정

## Output Format

```markdown
### verify-strategy 검증 결과

| # | 검사 항목 | 대상 | 결과 | 상세 |
|---|-----------|------|------|------|
| 1 | BaseStrategy 상속 | strategies/*/strategy.py | PASS/FAIL | ... |
| 2 | generate_signal 구현 | strategies/*/strategy.py | PASS/FAIL | ... |
| 3 | 반환값 유효성 | strategies/*/strategy.py | PASS/FAIL | ... |
| 4 | config.yaml 구조 | strategies/*/config.yaml | PASS/FAIL | ... |
| 5 | 엔트리포인트 등록 | main.py, backtest.py | PASS/FAIL | ... |
| 6 | ccxt 미의존 | strategies/*/strategy.py | PASS/FAIL | ... |
```

## Exceptions

1. **`src/strategies/base.py`** — BaseStrategy 자체는 generate_signal을 추상 메서드로 정의하므로 return 문이 `pass`여도 위반이 아님
2. **벡터화 전용 메서드** — `generate_signals_vectorized()`는 pd.Series를 반환하므로 int 반환 검사 대상이 아님
3. **`_get_or_compute_*` 헬퍼** — 이 메서드들은 generate_signal의 일부로 간주하여 별도 시그니처 검증 불필요
