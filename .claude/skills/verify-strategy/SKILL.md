---
name: verify-strategy
description: 전략 컨벤션 준수 여부를 검증합니다. 새 전략 추가 또는 기존 전략 수정 후 사용.
---

# 전략 컨벤션 검증

## Purpose

1. **BaseStrategy 상속 검증** — 모든 전략이 BaseStrategy를 상속하고 generate_signal(df) → (int, float)를 구현하는지 확인
2. **config.yaml 구조 검증** — 각 전략 폴더에 올바른 구조의 config.yaml이 존재하는지 확인
3. **엔트리포인트 등록 검증** — config/portfolio.yaml에 전략이 등록되고, PortfolioManager가 동적 로드 가능한지 확인
4. **피처 재활용 패턴 검증** — `_get_or_compute_*` 패턴으로 processor 피처를 재활용하는지 확인
5. **거래소 직접 의존 금지** — 전략 코드에서 ccxt를 직접 import하지 않는지 확인

## When to Run

- 새 전략을 추가한 후
- 기존 전략의 generate_signal 로직을 수정한 후
- config/portfolio.yaml의 active_strategies를 변경한 후
- 전략의 config.yaml을 수정한 후

## Related Files

| File | Purpose |
|------|---------|
| `src/strategies/base.py` | BaseStrategy ABC 정의 |
| `strategies/btc_1h_momentum/strategy.py` | BTC 모멘텀 전략 구현체 |
| `strategies/eth_1h_momentum/strategy.py` | ETH 모멘텀 전략 구현체 |
| `strategies/btc_1h_mean_reversion/strategy.py` | BTC 평균회귀 전략 구현체 |
| `config/portfolio.yaml` | 전략 등록 관리 (active_strategies) |
| `src/portfolio/manager.py` | 동적 전략 로드 (load_strategies_from_config) |
| `main.py` | 실거래 진입점 (PortfolioManager 기반) |
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
# generate_signal 메서드 정의 확인 (tuple[int, float] 반환)
Grep pattern="def generate_signal\(self, df.*DataFrame\).*tuple" path="strategies/" glob="*/strategy.py"
```

**PASS:** 모든 전략이 `generate_signal(self, df: pd.DataFrame) -> tuple[int, float]` 시그니처를 갖춤
**FAIL:** generate_signal 메서드가 없거나 시그니처가 다름

**수정:** `def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:` 메서드 추가/수정. 반환값은 `(signal, probability)` 튜플이어야 함

### Step 4: generate_signal 반환값 검증

**파일:** 각 `strategies/*/strategy.py`

**도구:** Read + 코드 분석

각 전략의 generate_signal 메서드를 읽고, return 문이 `(signal, probability)` 튜플을 반환하는지 확인합니다.
- signal: 1(매수) 또는 0(비매수). -1은 사용하지 않음 (2클래스: 매수/비매수)
- probability: 0.0~1.0 범위의 float

```bash
# return 문 확인
Grep pattern="return" path="strategies/" glob="*/strategy.py" output_mode="content"
```

**PASS:** 모든 return 문이 `(signal, probability)` 튜플을 반환하며, signal이 1 또는 0
**FAIL:** -1을 반환하거나 튜플이 아닌 단일 int를 반환하는 return 문이 존재

### Step 5: config.yaml 구조 검증

**파일:** 각 `strategies/*/config.yaml`

**도구:** Read

각 config.yaml을 읽고 필수 섹션이 있는지 확인합니다:
- `strategy.name` — 전략 클래스 이름
- `strategy.symbol` — 심볼 (BTCUSDT 형식) **[필수]**
- `strategy.timeframe` — 타임프레임 **[필수]**
- `params` — 전략 파라미터
- `execution` — 실행 설정
- `risk` — 리스크 설정

선택적 키:
- `labeler_type` — 라벨러 종류 (평균회귀 등 비기본 라벨러 사용 시)

```bash
# 필수 키 존재 확인
Grep pattern="^strategy:" path="strategies/" glob="*/config.yaml"
Grep pattern="  symbol:" path="strategies/" glob="*/config.yaml"
Grep pattern="  timeframe:" path="strategies/" glob="*/config.yaml"
Grep pattern="^params:" path="strategies/" glob="*/config.yaml"
Grep pattern="^execution:" path="strategies/" glob="*/config.yaml"
Grep pattern="^risk:" path="strategies/" glob="*/config.yaml"
```

**PASS:** 모든 config.yaml이 strategy(symbol, timeframe 포함), params, execution, risk 섹션을 포함
**FAIL:** 필수 섹션이 누락된 config.yaml 존재

### Step 6: 엔트리포인트 등록 검증

**파일:** `config/portfolio.yaml`, `src/portfolio/manager.py`, `backtest.py`

**도구:** Read + Grep

#### 6-1. portfolio.yaml 등록 확인

Step 1에서 발견된 각 전략 이름이 `config/portfolio.yaml`의 `active_strategies` 목록에 포함되어 있는지 확인합니다.

```bash
# portfolio.yaml에서 active_strategies 확인
Grep pattern="active_strategies" path="config/portfolio.yaml" output_mode="content" -A=10
```

**PASS:** 모든 전략 폴더에 대응하는 항목이 active_strategies에 존재
**FAIL:** 전략 폴더는 있지만 portfolio.yaml에 등록되지 않음

**수정:** `config/portfolio.yaml`의 `active_strategies` 목록에 전략 이름 추가

#### 6-2. PortfolioManager 동적 로드 검증

`src/portfolio/manager.py`의 `load_strategies_from_config()` 메서드가 `importlib.import_module()`로 전략을 동적 import하고, BaseStrategy 서브클래스를 찾아 인스턴스를 생성하는지 확인합니다.

```bash
# 동적 로드 로직 확인
Grep pattern="import_module|BaseStrategy" path="src/portfolio/manager.py" output_mode="content"
```

**PASS:** `load_strategies_from_config()`가 importlib 기반 동적 로드를 수행
**FAIL:** 하드코딩된 import 분기만 존재

#### 6-3. backtest.py 전략 분기 확인

backtest.py에는 전략별 import 분기가 여전히 존재할 수 있습니다. 각 전략에 대응하는 분기가 있는지 확인합니다.

```bash
# backtest.py에서 전략 분기 확인
Grep pattern="strategy_name ==" path="backtest.py" output_mode="content"
```

**PASS:** 모든 전략 폴더에 대응하는 import 분기가 backtest.py에 존재
**FAIL:** 전략 폴더는 있지만 backtest.py에 등록되지 않음

**수정:** 해당 전략의 elif 분기를 backtest.py:run()에 추가

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
| 2 | generate_signal 구현 | strategies/*/strategy.py | PASS/FAIL | 시그니처: tuple[int, float] |
| 3 | 반환값 유효성 | strategies/*/strategy.py | PASS/FAIL | 2클래스: 1(매수)/0(비매수) + probability |
| 4 | config.yaml 구조 | strategies/*/config.yaml | PASS/FAIL | symbol, timeframe 필수 |
| 5a | portfolio.yaml 등록 | config/portfolio.yaml | PASS/FAIL | active_strategies 목록 |
| 5b | PortfolioManager 로드 | src/portfolio/manager.py | PASS/FAIL | 동적 import 확인 |
| 5c | backtest.py 분기 | backtest.py | PASS/FAIL | 전략별 import 분기 |
| 6 | ccxt 미의존 | strategies/*/strategy.py | PASS/FAIL | ... |
```

## Exceptions

1. **`src/strategies/base.py`** — BaseStrategy 자체는 generate_signal을 추상 메서드로 정의하므로 return 문이 `pass`여도 위반이 아님
2. **벡터화 전용 메서드** — `generate_signals_vectorized()`는 pd.Series를 반환하므로 tuple 반환 검사 대상이 아님
3. **`_get_or_compute_*` 헬퍼** — 이 메서드들은 generate_signal의 일부로 간주하여 별도 시그니처 검증 불필요
4. **`labeler_type` 키** — 평균회귀 등 비기본 라벨러를 사용하는 전략에만 존재. 상세 검증은 verify-ml 영역
