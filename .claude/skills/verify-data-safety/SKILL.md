---
name: verify-data-safety
description: 데이터 안전성 규칙을 검증합니다. 실행/저장 관련 코드 수정 후 사용.
---

# 데이터 안전성 검증

## Purpose

1. **Atomic Write 검증** — 상태 파일(current_state.json) 저장이 tempfile → shutil.move 패턴을 사용하는지 확인
2. **trade_log.csv Append-Only 검증** — 거래 이력 파일이 덮어쓰기/삭제 없이 append-only로만 기록되는지 확인
3. **.env 보안 검증** — config/.env 파일이 .gitignore에 포함되고, 코드에서 하드코딩된 API 키가 없는지 확인
4. **raw 데이터 미가공 검증** — data/raw/ 디렉토리의 파일을 직접 수정하는 코드가 없는지 확인
5. **Parquet snappy 압축 검증** — Parquet 저장 시 compression="snappy"를 사용하는지 확인

## When to Run

- `src/execution/executor.py`를 수정한 후
- 상태 파일 저장 로직을 변경한 후
- 데이터 저장 관련 코드를 수정한 후
- .gitignore를 변경한 후

## Related Files

| File | Purpose |
|------|---------|
| `src/execution/executor.py` | 주문 실행 + trade_log 기록 + 상태 저장 (close_position, record_closed_pnl 포함) |
| `src/analytics/reporter.py` | 리포트 저장 (trade_log 읽기) |
| `main.py` | 실거래 루프 (_save_state에 extra_state 전달) |
| `.gitignore` | 버전 관리 제외 파일 설정 |
| `config/.env` | API 키 (절대 커밋 금지) |
| `config/current_state.json` | 현재 포지션 + extra_state (circuit_breaker, pnl_tracker 등) |

## Workflow

### Step 1: Atomic Write 패턴 검증

**파일:** `src/execution/executor.py`

**도구:** Grep

```bash
# _save_state에서 tempfile 사용 확인
Grep pattern="tempfile" path="src/execution/executor.py" output_mode="content"

# shutil.move 사용 확인
Grep pattern="shutil.move" path="src/execution/executor.py" output_mode="content"
```

추가로, 프로젝트 전체에서 `current_state.json`에 직접 쓰는 코드가 있는지 확인합니다:

```bash
# current_state.json에 직접 write하는 코드 탐색 (atomic write 우회)
Grep pattern="current_state.json" glob="*.py" output_mode="content"
```

각 발견된 위치를 읽고, _save_state를 통하지 않고 직접 open(..., "w")로 쓰는 코드가 있는지 확인합니다.

**PASS:** current_state.json 저장이 모두 tempfile → shutil.move 패턴을 사용
**FAIL:** 직접 open → write로 current_state.json을 저장하는 코드가 존재

**수정:** 직접 쓰기를 OrderExecutor._save_state() 호출로 대체

### Step 2: trade_log.csv Append-Only 검증

**파일:** 전체 Python 소스

**도구:** Grep

```bash
# trade_log 관련 코드 탐색
Grep pattern="trade_log" glob="*.py" output_mode="content"
```

발견된 각 위치를 읽고 다음을 확인합니다:
- `to_csv(TRADE_LOG_PATH, ...)` 호출 시 기존 데이터를 concat으로 합쳐서 저장하는지 (append 패턴)
- `os.remove()`, `os.unlink()`, `truncate()` 등으로 trade_log를 삭제/초기화하는 코드가 없는지
- `mode="w"`로 trade_log를 열어서 기존 데이터를 덮어쓰는 코드가 없는지

```bash
# trade_log 파일 삭제 시도 탐색
Grep pattern="(remove|unlink|truncate).*trade_log" glob="*.py"
```

**PASS:** trade_log.csv가 append-only로만 사용됨
**FAIL:** trade_log.csv를 삭제/초기화/덮어쓰는 코드가 존재

**수정:** 삭제/초기화 코드를 제거하고, append 패턴만 사용

### Step 3: .env 보안 검증

**파일:** `.gitignore`, 전체 Python 소스

**도구:** Grep

```bash
# .gitignore에서 .env 패턴 확인
Grep pattern="\\.env" path=".gitignore" output_mode="content"
```

```bash
# 하드코딩된 API 키 패턴 탐색 (따옴표 안의 긴 영숫자 문자열)
Grep pattern="(api_key|api_secret|API_KEY|API_SECRET)\s*=\s*[\"'][A-Za-z0-9]" glob="*.py"
```

```bash
# dotenv 사용 확인 (환경 변수로 로드하는지)
Grep pattern="load_dotenv|os.environ|os.getenv" glob="*.py" output_mode="content"
```

**PASS:** .env가 .gitignore에 포함되고, 하드코딩된 키가 없으며, dotenv를 통해 환경 변수로 로드
**FAIL:** .env가 .gitignore에 없거나, 하드코딩된 API 키가 존재

**수정:** .gitignore에 `.env` 패턴 추가, 하드코딩된 키를 환경 변수로 대체

### Step 4: raw 데이터 미가공 검증

**파일:** 전체 Python 소스

**도구:** Grep

```bash
# data/raw 경로에 쓰기 작업을 하는 코드 탐색
Grep pattern="data/raw.*\.(to_parquet|to_csv|write|save)" glob="*.py" output_mode="content"

# data/raw 디렉토리에 쓰는 open 호출 탐색
Grep pattern="open.*data/raw.*[\"']w" glob="*.py"
```

`src/data/collector.py`의 `save_ohlcv`만 data/raw/에 쓰기 허용 (원본 저장).
그 외의 코드에서 data/raw/를 수정하는 것은 위반입니다.

**PASS:** data/raw/에 쓰는 코드가 collector.py의 save_ohlcv만 존재
**FAIL:** collector.py 외 코드에서 data/raw/를 수정하는 코드가 존재

**수정:** 해당 코드를 data/processed/로 대상 경로 변경

### Step 5: Parquet snappy 압축 검증

**파일:** 전체 Python 소스

**도구:** Grep

```bash
# to_parquet 호출 탐색
Grep pattern="to_parquet" glob="*.py" output_mode="content"
```

각 `to_parquet()` 호출에서 `compression="snappy"` 파라미터가 지정되어 있는지 확인합니다.

**PASS:** 모든 to_parquet() 호출이 compression="snappy"를 사용
**FAIL:** compression 파라미터가 없거나 다른 값인 to_parquet() 호출이 존재

**수정:** `compression="snappy"` 파라미터 추가

### Step 6: close_position / record_closed_pnl Append-Only 검증

**파일:** `src/execution/executor.py`

**도구:** Read

executor.py의 `close_position()` 및 `record_closed_pnl()` 메서드를 읽고 다음을 확인합니다:

1. `close_position()`은 `execute()`를 통해 주문을 실행 → `_record_trade()`에서 `mode="a"` 사용 (기존 검사에서 커버)
2. `record_closed_pnl()`이 trade_log에 직접 쓸 때 `mode="a"` (append)를 사용하는지
3. `record_closed_pnl()`이 기존 행을 수정/삭제하지 않는지

```bash
# record_closed_pnl의 CSV 쓰기 패턴 확인
Grep pattern="record_closed_pnl" path="src/execution/executor.py" output_mode="content" -A=20
```

**PASS:** 두 메서드 모두 trade_log에 append-only로 기록
**FAIL:** `mode="w"`, `truncate`, `remove` 등 기존 데이터 파괴 패턴 존재

### Step 7: extra_state Atomic Write 검증

**파일:** `src/execution/executor.py`, `main.py`

**도구:** Read, Grep

```bash
# _save_state의 extra_state 처리 확인
Grep pattern="extra_state" path="src/execution/executor.py" output_mode="content"

# main.py에서 extra_state 전달 확인
Grep pattern="extra_state|_save_state" path="main.py" output_mode="content"
```

**검증:**
1. `_save_state(positions, extra_state=...)` 시 extra_state가 기존 state dict에 `.update()`로 병합되는지
2. 병합 후 전체 state를 `tempfile → shutil.move` 패턴으로 저장하는지 (기존 atomic write 로직 유지)
3. main.py에서 매 순환마다 `_save_state(extra_state={...})` 형태로 호출하는지

**PASS:** extra_state가 기존 atomic write 파이프라인 내에서 저장됨
**FAIL:** extra_state를 별도 파일에 직접 쓰거나, atomic write를 우회

## Output Format

```markdown
### verify-data-safety 검증 결과

| # | 검사 항목 | 결과 | 상세 |
|---|-----------|------|------|
| 1 | Atomic Write 패턴 | PASS/FAIL | ... |
| 2 | trade_log Append-Only | PASS/FAIL | ... |
| 3 | .env 보안 | PASS/FAIL | ... |
| 4 | raw 데이터 미가공 | PASS/FAIL | ... |
| 5 | Parquet snappy 압축 | PASS/FAIL | ... |
| 6 | close_position/record_closed_pnl Append-Only | PASS/FAIL | ... |
| 7 | extra_state Atomic Write | PASS/FAIL | ... |
```

## Exceptions

1. **collector.py의 data/raw/ 쓰기** — `src/data/collector.py`의 `save_ohlcv`는 원본 데이터를 저장하는 유일한 허용된 쓰기 경로
2. **테스트 코드의 임시 파일** — `tests/` 디렉토리 내에서 tempfile을 사용한 테스트용 trade_log 생성/삭제는 허용
3. **config/.env.example** — `.env.example`은 실제 키가 아닌 템플릿이므로 커밋 가능
