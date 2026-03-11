# bybit-quant

Bybit 거래소 기반 암호화폐 퀀트 트레이딩 시스템.
백테스팅과 실거래가 동일한 전략 코드(`BaseStrategy.generate_signal`)를 공유하는 구조로,
규칙 기반(MA Crossover)과 ML 기반(LightGBM 2클래스 분류, 롱 전용, 앙상블 지원) 전략을 지원한다.

---

## 기술 스택

| 영역 | 선택 |
|------|------|
| 언어 | Python 3.11+ |
| 거래소 | Bybit (USDT 무기한 선물) |
| 거래소 연동 | ccxt |
| 데이터 처리 | pandas, numpy |
| 백테스팅 | vectorbt |
| ML | LightGBM, Optuna, scikit-learn |
| 알림 | python-telegram-bot |
| 파일 포맷 | Parquet (snappy), CSV, JSON, YAML |

---

## 빠른 시작

### 1. 환경 설정

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
```

`config/.env`에 API 키를 설정한다:

```
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 2. 데이터 수집

```bash
python -c "
from src.data.collector import BybitDataCollector
c = BybitDataCollector()
df = c.fetch_ohlcv_bulk('BTC/USDT:USDT', '1h', '2024-01-01T00:00:00Z')
c.save_ohlcv(df, 'BTC/USDT:USDT', '1h')
"
```

### 3. 데이터 처리 (피처 엔지니어링)

```bash
python -c "
import pandas as pd
from src.data.processor import DataProcessor
df = pd.read_parquet('data/raw/bybit/BTCUSDT/1h/2024-01.parquet')
p = DataProcessor()
p.process_and_save(df, 'BTCUSDT', '1h')
"
```

### 4. 백테스트

```bash
python backtest.py --strategy ma_crossover
python backtest.py --strategy lgbm_classifier
python backtest.py --strategy ma_crossover --symbol BTCUSDT --timeframe 1h
```

### 5. LightGBM 모델 학습 (ML 전략 사용 시)

```bash
python train_lgbm.py --no-optuna                            # 기본 학습
python train_lgbm.py --optuna-trials 100                    # Optuna 튜닝 학습
```

### 6. OOS 검증 및 실험 자동화

```bash
python oos_validation.py                                    # OOS (Out-of-Sample) 검증
python run_experiment.py --threshold 0.48 --no-optuna       # 학습→OOS 검증 자동화
```

### 7. 실거래

```bash
python main.py --strategy ma_crossover --mode live
python main.py --strategy lgbm_classifier --mode live
```

---

## 아키텍처

### 실거래 흐름

```
BybitDataCollector.fetch_ohlcv()
  -> DataProcessor.add_features()
  -> Strategy.generate_signal(df) -> signal (1=매수, 0=중립)
  -> RiskManager.check_all() -> 통과 여부
  -> RiskManager.calculate_atr_position_size() -> 포지션 크기
  -> RiskManager.get_stop_take_profit() -> SL/TP 가격
  -> OrderExecutor.execute() -> 주문 제출 + trade_log.csv 기록
```

### 백테스트 흐름

```
data/processed/ Parquet 로드
  -> Strategy.generate_signals_vectorized(df) -> 전체 신호 시리즈
  -> vectorbt.Portfolio.from_signals() -> 성과 계산
  -> Reporter.calculate_metrics() + save_backtest_result()
  -> TelegramNotifier로 결과 알림
```

### ML 학습 흐름

```
data/processed/ Parquet 로드
  -> FeatureEngine.compute_all_features() -> ~47개 피처 (선별 시 18개)
  -> TripleBarrierLabeler.generate_labels() -> 2클래스 라벨 (1=매수/0=비매수)
  -> WalkForwardTrainer.run() -> 확장 윈도우 학습 + Optuna 튜닝
  -> 모델 저장 (strategies/lgbm_classifier/models/)
  -> oos_validation.py로 OOS 검증
```

---

## 전략

### MA Crossover (규칙 기반)

Fast/Slow MA 크로스오버로 매매 신호를 생성하며, 4개 필터로 whipsaw를 억제한다.

| 필터 | 조건 |
|------|------|
| 트렌드 (MA200) | 장기 추세 방향과 일치하는 신호만 허용 |
| RSI (14) | 과매수/과매도 구간에서 역방향 신호 억제 |
| 거래량 (MA20) | 평균 거래량 대비 충분한 거래량 동반 시에만 유효 |
| ATR (14) | 최소 변동성 미달 시 가짜 크로스로 판단 |

### LightGBM Classifier (ML 기반)

LightGBM 2클래스 분류 모델로 `predict_proba` 기반 매매 신호를 생성한다. **롱 전용** (1=매수, 0=비매수).

- **피처**: ~47개 전체 또는 18개 선별 (기술적 지표, 가격/거래량 파생, 시간 피처, 멀티타임프레임)
- **라벨링**: Triple Barrier (상단 1.8×ATR, 하단 1.8×ATR 대칭, 최대 보유 16봉)
- **학습**: Walk-Forward 확장 윈도우 (최소 6개월 학습, 1개월 검증, 24봉 embargo)
- **튜닝**: Optuna (첫 fold에서만 튜닝, 이후 동일 파라미터 적용)
- **앙상블**: 여러 fold 모델 평균 예측 (`config.yaml`의 `ensemble_folds` 설정)
- **OOS 검증**: IS/PV 성과 비교로 일반화 성능 확인 (PF 하락률 50% 이내 목표)

---

## 리스크 관리

모든 주문은 `RiskManager.check_all()` 통과 후에만 실행된다.

| 파라미터 | 값 |
|----------|-----|
| 단일 포지션 최대 | 자산의 5% |
| 동시 최대 포지션 | 3개 |
| 최대 레버리지 | 3배 |
| 일일 손실 한도 | 3% |
| 월간 손실 한도 | 10% |
| 기본 손절 / 익절 | 1.5% / 1.5% (대칭) |
| 거래당 위험 | 1% |
| Circuit Breaker | 연속 손실 5회 또는 변동성 5% 초과 시 발동 |

Circuit Breaker는 수동으로만 리셋 가능하다.

---

## 프로젝트 구조

```
config/              환경변수(.env), 거래소/리스크 설정, 현재 상태
data/raw/bybit/      원본 OHLCV (월별 Parquet, 가공 금지)
data/processed/      피처 엔지니어링 결과 Parquet
src/
  data/              collector.py (REST/WS 수집), processor.py (피처 엔지니어링)
  strategies/        base.py (BaseStrategy ABC)
  risk/              manager.py (RiskManager + CircuitBreaker)
  execution/         executor.py (주문 실행, 멱등성, trade_log 기록)
  analytics/         reporter.py (메트릭 계산, 리포트 저장)
  utils/             logger.py (RotatingFileHandler), notify.py (Telegram)
strategies/
  ma_crossover/      규칙 기반 전략 (strategy.py + config.yaml)
  lgbm_classifier/   ML 전략 (strategy.py, features.py, labeler.py, trainer.py, evaluator.py, models/)
reports/             backtest/, live/ (일별 JSON), trades/ (trade_log.csv)
docs/                모델 분석 리포트
tests/               테스트 모듈
main.py              실거래 진입점
backtest.py          백테스트 단독 실행
train_lgbm.py        LightGBM 학습 CLI
oos_validation.py    OOS (Out-of-Sample) 검증
run_experiment.py    학습→OOS 검증 자동화
```

---

## 새 전략 추가

1. `strategies/{name}/` 폴더 생성
2. `strategy.py` - `BaseStrategy` 상속, `generate_signal(df) -> int` 구현 (1=매수, 0=중립; 롱 전용 전략은 -1 미사용)
3. `config.yaml` - strategy, params, execution, risk 섹션 포함
4. (선택) `generate_signals_vectorized(df) -> pd.Series` 구현으로 백테스트 고속화
5. `main.py:load_strategy()`와 `backtest.py:run()`에 전략 import 분기 추가

---

## 테스트

```bash
python -m pytest tests/ -v                        # 전체 테스트
python -m pytest tests/test_risk_manager.py -v     # 단일 파일
python -m pytest tests/test_risk_manager.py::test_function_name -v  # 단일 테스트
```

---

## 핵심 설계 원칙

- **전략-거래소 분리**: 전략 코드는 ccxt에 직접 의존하지 않음
- **피처 재활용**: `_get_or_compute_*` 패턴으로 processor 피처가 있으면 재사용
- **리스크 우선**: `check_all()` 실패 시 주문 미실행 (Fail-Safe)
- **멱등성 보장**: 동일 symbol/side 미체결 주문 존재 시 중복 주문 방지
- **데이터 무결성**: trade_log.csv는 append-only, 상태 파일은 atomic write
- **ML 분리 학습**: 실거래 루프 내에서 재학습하지 않음 (train_lgbm.py 별도 실행)

---

## 모델 버전 관리

유의미한 모델은 **git 커밋 + 태그**로 보존한다 (`model/{전략}/{run번호}`).

- 커밋 대상: `data/`, `strategies/**/models/`, `strategies/**/config.yaml`
- 주의: raw 재수집 시 값 미세 변동, Optuna 비결정성 → 좋은 모델은 즉시 커밋

---

## 문서

| 파일 | 설명 |
|------|------|
| `ARCHITECTURE.md` | 시스템 아키텍처 및 모듈 설계 |
| `FOLDER_STRUCTURE.md` | 프로젝트 폴더/파일 구조 상세 |
| `DATA_PIPELINE.md` | 데이터 수집 및 저장 전략 |
| `STRATEGY.md` | 전략 개발 가이드 |
| `RISK_MANAGEMENT.md` | 리스크 관리 원칙 및 구현 |
| `MONITORING.md` | 모니터링, 알림, 운영 가이드 |
| `docs/*.md` | 모델 분석 리포트 (run별) |

---

## 법적 고려사항

- 한국 기준 암호화폐 수익 250만 원 초과 시 과세 (2025년~)
- 해외 거래소 연간 $10,000 초과 송금 시 외국환거래법 신고 의무
- Bybit 이용약관 준수 필수
