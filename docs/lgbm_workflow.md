# LightGBM 전략 워크플로우

## 1. 데이터 수집

```bash
python -c "
from src.data.collector import BybitDataCollector
c = BybitDataCollector()
df = c.fetch_ohlcv_bulk('BTC/USDT:USDT', '1h', '2024-01-01T00:00:00Z')
c.save_ohlcv(df, 'BTC/USDT:USDT', '1h')
"
```

## 2. 피처 엔지니어링

```bash
python -c "
import pandas as pd
from src.data.processor import DataProcessor
df = pd.read_parquet('data/raw/bybit/BTCUSDT/1h/2024-01.parquet')
p = DataProcessor()
p.process_and_save(df, 'BTCUSDT', '1h')
"
```

→ `data/processed/BTCUSDT_1h_features.parquet` 생성

## 3. 모델 학습

```bash
# Optuna 하이퍼파라미터 튜닝 포함 (기본)
python train_lgbm.py --symbol BTCUSDT --timeframe 1h

# Optuna 시행 횟수 지정
python train_lgbm.py --symbol BTCUSDT --timeframe 1h --optuna-trials 100

# Optuna 없이 기본 파라미터로 학습
python train_lgbm.py --symbol BTCUSDT --timeframe 1h --no-optuna
```

### 학습 과정

- **피처 계산** → FeatureEngine이 ~50개 피처 생성
- **라벨 생성** → TripleBarrierLabeler가 3클래스(매수1/중립0/매도-1) 라벨 생성
- **Walk-Forward 학습** → 확장 윈도우(최소 6개월 학습 → 1개월 검증), fold 간 24봉 embargo
- **Optuna 튜닝** → 첫 fold에서만 수행, 이후 fold에 동일 파라미터 적용

### 저장 결과 (`strategies/lgbm_classifier/models/`)

- `latest.txt` — 모델 파일
- `feature_names.json` — 피처 목록
- `best_params.json` — 최적 하이퍼파라미터
- `training_meta.json` — fold별 성과/피처 importance

## 4. 백테스트

```bash
python backtest.py --strategy lgbm_classifier
```

## 5. 실거래

```bash
python main.py --strategy lgbm_classifier --mode live
```

## 핵심 포인트

- **학습과 실거래는 분리** — live 루프 내에서 재학습하지 않음
- **라벨 매핑** — 학습 시 (-1,0,1)→(0,1,2), 추론 시 역매핑
- **시그널 생성** — predict_proba로 확률 추출 → confidence_threshold 초과 시에만 시그널 발생
- **리스크 관리** — 시그널 발생 후 RiskManager.check_all() 통과해야 주문 실행
