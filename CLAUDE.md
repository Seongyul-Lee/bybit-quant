# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요
Bybit 거래소 기반 암호화폐 퀀트 트레이딩 시스템.
백테스팅과 실거래가 동일한 전략 코드(`BaseStrategy.generate_signal`)를 공유하는 구조.
규칙 기반(MA Crossover)과 ML 기반(LightGBM 3클래스 분류) 전략을 지원한다.

## 커맨드

```bash
# 환경 설정
.venv/Scripts/activate   # Windows venv 활성화
pip install -r requirements.txt

# 테스트
python -m pytest tests/ -v                    # 전체 테스트
python -m pytest tests/test_risk_manager.py -v  # 단일 파일 테스트
python -m pytest tests/test_risk_manager.py::test_function_name -v  # 단일 테스트

# 데이터 수집 → 처리 (순서 의존)
python -c "
from src.data.collector import BybitDataCollector
c = BybitDataCollector()
df = c.fetch_ohlcv_bulk('BTC/USDT:USDT', '1h', '2024-01-01T00:00:00Z')
c.save_ohlcv(df, 'BTC/USDT:USDT', '1h')
"

python -c "
import pandas as pd
from src.data.processor import DataProcessor
df = pd.read_parquet('data/raw/bybit/BTCUSDT/1h/2024-01.parquet')
p = DataProcessor()
p.process_and_save(df, 'BTCUSDT', '1h')
"

# 백테스트 (data/processed/{symbol}_{timeframe}_features.parquet 필요)
python backtest.py --strategy ma_crossover
python backtest.py --strategy lgbm_classifier
python backtest.py --strategy ma_crossover --symbol BTCUSDT --timeframe 1h

# LightGBM 모델 학습 (백테스트 전에 먼저 실행)
python train_lgbm.py --symbol BTCUSDT --timeframe 1h
python train_lgbm.py --symbol BTCUSDT --timeframe 1h --optuna-trials 100
python train_lgbm.py --symbol BTCUSDT --timeframe 1h --no-optuna

# 실거래
python main.py --strategy ma_crossover --mode live
python main.py --strategy lgbm_classifier --mode live

# main.py는 --mode backtest도 지원 (backtest.py와 동일 로직)
python main.py --strategy ma_crossover --mode backtest
```

## 아키텍처

### 실행 흐름 (실거래 루프 — main.py:run_live)
```
BybitDataCollector.fetch_ohlcv() → DataProcessor.add_features()
  → Strategy.generate_signal(df) → signal (1/0/-1)
  → RiskManager.check_all() → 통과 여부
  → RiskManager.calculate_atr_position_size() → 포지션 크기
  → RiskManager.get_stop_take_profit() → SL/TP 가격
  → OrderExecutor.execute() → 주문 제출 + trade_log.csv 기록
```

### 백테스트 흐름 (backtest.py)
```
data/processed/ Parquet 로드
  → Strategy.generate_signals_vectorized(df) → 전체 신호 시리즈
  → vectorbt.Portfolio.from_signals() → 성과 계산
  → Reporter.calculate_metrics() + save_backtest_result()
  → TelegramNotifier로 결과 알림
```

### ML 학습 흐름 (train_lgbm.py)
```
data/processed/ Parquet 로드
  → FeatureEngine.compute_all_features() → ~50개 피처
  → TripleBarrierLabeler.generate_labels() → 3클래스 라벨 (1/0/-1)
  → WalkForwardTrainer.run() → 확장 윈도우 학습 + Optuna 튜닝
  → 모델 저장 (strategies/lgbm_classifier/models/)
```

### 핵심 모듈 관계
- **전략 코드는 거래소에 직접 의존하지 않음** — ccxt는 collector/executor에서만 사용
- **전략은 processor 피처를 재활용** — `_get_or_compute_*` 패턴으로 `ma_N`, `rsi_14`, `atr_14`, `volume_ma_20` 컬럼이 있으면 재사용, 없으면 자체 계산
- **RiskManager는 전략 신호보다 항상 우선** — check_all() 실패 시 주문 미실행
- **OrderExecutor는 멱등성 보장** — 동일 symbol/side 미체결 주문 존재 시 중복 주문 방지
- **ML 전략은 별도 학습 → 모델 로드 구조** — train_lgbm.py로 학습, strategy.py에서 로드하여 추론만 수행

### 전략 유형
| 전략 | 방식 | 시그널 로직 |
|------|------|------------|
| `ma_crossover` | 규칙 기반 | Fast/Slow MA 크로스오버 + 4개 필터 (트렌드/RSI/거래량/ATR) |
| `lgbm_classifier` | ML 기반 | LightGBM predict_proba → confidence_threshold 이상 시 시그널 |

## 새 전략 추가 방법

1. `strategies/{name}/` 폴더 생성
2. `strategy.py` — `BaseStrategy` 상속, `generate_signal(df) → int` 구현 (1=매수, -1=매도, 0=중립)
3. `config.yaml` — strategy(name, symbol, timeframe), params, execution, risk 섹션
4. 백테스트 속도가 중요하면 `generate_signals_vectorized(df) → pd.Series` 추가 구현
5. `main.py:load_strategy()`와 `backtest.py:run()`에 전략 import 분기 추가

기존 전략 참고: `strategies/ma_crossover/` (규칙 기반), `strategies/lgbm_classifier/` (ML 기반)

## 폴더 구조
```
config/          → .env(git제외), exchanges.yaml, risk_params.yaml, current_state.json
data/raw/bybit/  → 원본 OHLCV (월별 Parquet, 가공 금지)
data/processed/  → 피처 엔지니어링 결과 Parquet ({symbol}_{timeframe}_features.parquet)
src/data/        → collector.py(REST/WS 수집), processor.py(피처 엔지니어링)
src/strategies/  → base.py(BaseStrategy ABC)
src/risk/        → manager.py(RiskManager + CircuitBreaker)
src/execution/   → executor.py(주문실행, 멱등성, trade_log 기록)
src/analytics/   → reporter.py(메트릭 계산, 리포트 저장)
src/utils/       → logger.py(RotatingFileHandler), notify.py(텔레그램)
strategies/      → 전략별 독립 폴더 (strategy.py + config.yaml + backtest_results/)
  ma_crossover/  → 규칙 기반 전략
  lgbm_classifier/ → ML 전략 (features.py, labeler.py, trainer.py, evaluator.py, models/)
reports/         → backtest/, live/(일별JSON), trades/(trade_log.csv)
tests/           → test_collector, test_processor, test_risk_manager, test_executor, test_lgbm_*
train_lgbm.py    → LightGBM 학습 CLI 스크립트
```

## 코딩 컨벤션
- 모든 전략은 `BaseStrategy`를 상속하고 `generate_signal(df) → int` 구현
- 상태 파일(current_state.json) 저장은 반드시 **atomic write** (tempfile → shutil.move)
- 거래 이력(trade_log.csv)은 **append-only** — 기존 행 수정/삭제 금지
- Parquet 저장 시 `compression="snappy"`
- 모든 Python 파일은 type hint와 docstring 포함
- 심볼 형식: REST API는 `BTC/USDT:USDT`, 파일명/config는 `BTCUSDT`

## 작업 시 주의사항
- `config/.env` 절대 수정/커밋 금지 (API 키 포함)
- `data/` 디렉토리 파일 직접 편집 금지 (코드를 통해서만 생성/갱신)
- `data/raw/`의 원본 데이터는 가공하지 않음 (처리 결과는 data/processed/에 저장)
- `reports/trades/trade_log.csv`는 append-only — 행 삭제나 수정 금지
- CircuitBreaker 리셋은 수동으로만 가능 (자동 리셋 코드 작성 금지)
- Fail-Safe 우선: 오류 발생 시 포지션 유지 대신 안전 종료
- ML 전략 실거래 재학습은 live 루프 내에서 하지 않음 (train_lgbm.py 별도 실행)

## 리스크 파라미터 (config/risk_params.yaml)
- 단일 포지션 최대 5%, 동시 최대 3개, 레버리지 3배
- 일일 손실 3%, 월간 손실 10% 초과 시 거래 중단
- 연속 손실 5회 또는 변동성 5% 초과 시 Circuit Breaker
- 기본 손절 2%, 익절 4% (R:R = 1:2), 거래당 위험 1%

## LightGBM 전략 상세

### 라벨 매핑
학습 시 원본(-1,0,1) → 학습용(0,1,2) 변환. 추론 시 역매핑(0,1,2) → (-1,0,1).

### 모델 저장 구조 (strategies/lgbm_classifier/models/)
- `latest.txt` — LightGBM Booster 텍스트 포맷
- `feature_names.json` — 피처 이름 목록
- `best_params.json` — Optuna 최적 하이퍼파라미터
- `training_meta.json` — fold별 성과, 피처 importance

### Walk-Forward 학습
- 확장 윈도우: 최소 6개월 학습 → 1개월 검증, 매 fold 학습 윈도우 확장
- Embargo: 학습 끝~검증 시작 사이 24봉 제거 (정보 유출 방지)
- Optuna: 첫 fold에서만 튜닝 → 이후 fold에 동일 파라미터 적용

## Skills

커스텀 검증 및 유지보수 스킬은 `.claude/skills/`에 정의되어 있습니다.

| Skill | Purpose |
|-------|---------|
| `verify-implementation` | 프로젝트의 모든 verify 스킬을 순차 실행하여 통합 검증 보고서를 생성합니다 |
| `verify-strategy` | 전략 컨벤션 준수 여부를 검증합니다 (BaseStrategy 상속, config.yaml 구조, 엔트리포인트 등록) |
| `verify-risk` | 리스크 관리 규칙 준수 여부를 검증합니다 (파라미터 동기화, CircuitBreaker 자동리셋 금지, check_all 우선) |
| `verify-data-safety` | 데이터 안전성 규칙을 검증합니다 (atomic write, trade_log append-only, .env 보안) |
| `verify-ml` | ML 전략(LightGBM) 전용 규칙을 검증합니다 (라벨 매핑, 모델 파일, 피처 일관성, Walk-Forward, Live 재학습 금지) |
| `manage-skills` | 세션 변경사항을 분석하고, 검증 스킬을 생성/업데이트하며, CLAUDE.md를 관리합니다 |
