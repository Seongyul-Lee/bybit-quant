# bybit-quant

Bybit 거래소 기반 암호화폐 멀티 전략 퀀트 트레이딩 시스템.
백테스팅과 실거래가 동일한 전략 코드(`BaseStrategy.generate_signal`)를 공유하는 구조로,
LightGBM 2클래스 분류(롱 전용, 앙상블 지원) 전략을 멀티 심볼 포트폴리오로 운용한다.

---

## 활성 전략

| 전략 | 심볼 | 타임프레임 | 앙상블 | OOS PV PF |
|------|------|-----------|--------|-----------|
| btc_1h_momentum | BTC/USDT | 1h | fold [11, 13] | 2.09 |
| eth_1h_momentum | ETH/USDT | 1h | fold [12, 14] | 2.11 |
| btc_1h_mean_reversion | BTC/USDT | 1h | fold 14 (단일) | 2.35 |
| funding_arb | BTC+ETH | - | - (Buy & Hold) | - |

포트폴리오: ML 전략은 전략당 20% 균등 배분, 전체 노출 60% 상한, 동일 심볼 30% 상한.
funding_arb는 별도 자본 40% 배분 (PortfolioManager 미경유, ArbExecutor 직접 실행).

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

# OHLCV
df = c.fetch_ohlcv_bulk('BTC/USDT:USDT', '1h', '2024-01-01T00:00:00Z')
c.save_ohlcv(df, 'BTC/USDT:USDT', '1h')

# 펀딩비
fr = c.fetch_funding_rate_bulk('BTC/USDT:USDT', '2024-01-01T00:00:00Z')
fr.to_parquet('data/raw/bybit/BTCUSDTUSDT/funding_rate.parquet', index=False)
"
```

### 3. 데이터 처리 (피처 엔지니어링)

```bash
python -c "
import pandas as pd, glob
from src.data.processor import DataProcessor
files = sorted(glob.glob('data/raw/bybit/BTCUSDTUSDT/1h/*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
DataProcessor().process_and_save(df, 'BTCUSDT', '1h')
"
```

### 4. 모델 학습

```bash
python train_lgbm.py --strategy btc_1h_momentum --no-optuna          # 기본 파라미터 학습
python train_lgbm.py --strategy eth_1h_momentum --optuna-trials 100   # Optuna 튜닝 학습
```

`--strategy`로 전략을 지정하면 `strategies/{name}/config.yaml`에서 심볼, 타임프레임, 배리어 등을 자동 로드한다.

### 5. 주기적 재학습

```bash
python retrain.py --strategy btc_1h_momentum                 # 단일 전략 재학습
python retrain.py --all                                       # 전체 전략 재학습
python retrain.py --all --dry-run --skip-data-collection      # 전체 전략 검증 (dry-run)
```

재학습 흐름: 데이터 증분 수집 → 기존 OOS 검증 → 백업 → 재학습 → 앙상블 자동 선택 → OOS 검증 → 교체/복원.
교체 기준: `strict OOS` 포함 배포 게이트 통과 + `new_pv_pf >= old_pv_pf × 0.9` (10% 하락 허용). 게이트 실패 시 자동 복원.

### 6. OOS 검증

```bash
python oos_validation.py --strategy btc_1h_momentum
python oos_validation.py --strategy eth_1h_momentum
python oos_validation.py --strategy btc_1h_mean_reversion
python oos_validation.py --all
```

검증 실행 시 JSON 아티팩트가 자동 저장된다.
- 전략별: `reports/validation/strategies/{strategy}/{timestamp}.json`
- 전체 실행: `reports/validation/runs/{timestamp}.json`

### 7. 백테스트

```bash
python backtest.py --strategy btc_1h_momentum
python backtest.py --strategy eth_1h_momentum
python backtest.py --strategy btc_1h_mean_reversion
```

### 8. 실거래

```bash
python main.py --mode live    # portfolio.yaml의 활성 전략 모두 로드
```

---

## 아키텍처

### 실거래 흐름 (멀티 전략)

Bybit Hedge Mode 활성화 — 동일 심볼에서 v1 전략과 funding_arb가 양방향 포지션으로 동시 운용 가능.

```
[ML 전략 — 봉 마감 주기]
봉 마감 감지
  → 각 전략의 generate_signal(df) → (signal, probability)
  → PortfolioManager.allocate() → 자본 배분 + 주문 목록
  → PortfolioRiskManager.check_portfolio() → 전체 리스크 체크
  → 전략별 RiskManager.check_all() → 개별 리스크 체크
  → VirtualPositionTracker → 가상→실제 포지션 변환
  → OrderExecutor.execute() → 거래소 주문 제출

[funding_arb — 별도 실행 경로]
펀딩비 수집
  → FundingArbStrategy.generate_signal() → 진입/청산 판단
  → ArbRiskMonitor.check_all() → basis/margin/funding trend 체크
  → ArbExecutor.execute() → spot+perp 동시 실행 (SpotExecutor + OrderExecutor)
```

### 백테스트 흐름

```
data/processed/ Parquet 로드
  → Strategy.generate_signals_vectorized(df) → 전체 신호 시리즈
  → vectorbt.Portfolio.from_signals() → 성과 계산
  → Reporter.calculate_metrics() + save_backtest_result()
```

### ML 학습 흐름

```
data/processed/ Parquet 로드
  → FeatureEngine.compute_all_features() → ~43개 피처 계산 → 선별 ~18개 사용
  → TripleBarrierLabeler / MeanReversionLabeler → 2클래스 라벨 (1=매수/0=비매수)
  → WalkForwardTrainer.run() → Sliding/Expanding 윈도우 학습 + Optuna 튜닝
  → 모델 저장 (strategies/{name}/models/)
  → oos_validation.py로 OOS 검증
```

---

## 리스크 관리

### 2계층 리스크 구조

| 계층 | 역할 |
|------|------|
| 전략 레벨 (RiskManager) | SL/TP, 포지션 크기, 일/월 손실 한도, CircuitBreaker |
| 포트폴리오 레벨 (PortfolioRiskManager) | 전체 MDD 한도, 전략 자동 비활성화 |
| 펀딩 아비트리지 (ArbRiskMonitor) | basis divergence, margin utilization, funding trend, 진입 슬리피지 |

### 주요 파라미터

| 파라미터 | 값 |
|----------|-----|
| 전략당 포지션 | 20% |
| 전체 노출 상한 | 60% |
| 동일 심볼 상한 | 30% |
| 일일 손실 한도 | 3% |
| 포트폴리오 MDD 한도 | 10% |
| SL / TP | 2.1% / 2.1% (대칭) |
| Circuit Breaker | 연속 손실 5회 시 발동 (수동 리셋) |
| 전략 자동 비활성화 | 50거래 후 PF < 0.8 |

---

## 프로젝트 구조

```
config/
  portfolio.yaml       포트폴리오 설정 (활성 전략, 배분, 한도, 리스크)
  .env                 API 키 (git 제외)
  risk_params.yaml     전략 레벨 리스크 파라미터
data/
  raw/bybit/{SYMBOL}USDT/
    {timeframe}/       월별 OHLCV Parquet
    funding_rate.parquet  펀딩비 데이터
    open_interest_1h.parquet  미결제약정 데이터
  processed/           피처 Parquet ({SYMBOL}_{timeframe}_features.parquet)
src/
  data/                collector.py, processor.py
  portfolio/           manager.py, risk.py, virtual_position.py
  strategies/          base.py (BaseStrategy ABC)
  risk/                manager.py (RiskManager + CircuitBreaker), arb_monitor.py (ArbRiskMonitor)
  execution/           executor.py (주문 실행, 멱등성), arb_executor.py, spot_executor.py
  analytics/           reporter.py
  utils/               logger.py, notify.py
strategies/
  _common/             features.py, labeler.py, trainer.py, evaluator.py (공통 ML)
  btc_1h_momentum/     BTC 1h 모멘텀 전략 (strategy.py, config.yaml, models/)
  eth_1h_momentum/     ETH 1h 모멘텀 전략 (strategy.py, config.yaml, models/)
  btc_1h_mean_reversion/ BTC 1h 평균회귀 전략 (strategy.py, labeler.py, config.yaml, models/)
  funding_arb/         펀딩비 차익거래 전략 (config.yaml, Buy & Hold, ArbExecutor 경유)
train_lgbm.py          학습 CLI (--strategy로 전략 지정)
oos_validation.py      OOS 검증 (--strategy 또는 --all, --json 구조화 출력, 아티팩트 자동 저장)
retrain.py             주기적 재학습 파이프라인 (증분수집→학습→strict OOS 게이트 검증→교체)
backtest.py            백테스트 (--strategy로 전략 지정)
main.py                실거래 진입점 (portfolio.yaml 기반 멀티 전략)
```

---

## 새 전략 추가

### ML 전략 (LightGBM 기반)

1. `strategies/{name}/` 폴더 생성 — `strategy.py`, `config.yaml`, `models/`
2. `strategy.py`에서 `BaseStrategy` 상속, `generate_signal(df) → (int, float)` 구현
3. `config.yaml`에 strategy (name, symbol, timeframe), params, execution, risk 섹션
4. `main.py:load_strategy()`와 `backtest.py:run()`에 전략 import 분기 추가
5. `config/portfolio.yaml`의 `active_strategies`에 전략 이름 추가

```bash
# 학습 → 검증 → 포트폴리오 추가 흐름
python train_lgbm.py --strategy {name} --optuna-trials 100
python oos_validation.py --strategy {name}
# strict OOS 게이트 통과 시 portfolio.yaml에 추가
```

### 규칙 기반 전략 (funding_arb 등)

ML 학습이 필요 없는 전략은 별도 실행 경로를 사용한다.
- `strategy.py`에서 규칙 기반 시그널 생성
- `config/portfolio.yaml`의 `funding_arb` 섹션에서 자본 배분 설정
- `ArbExecutor` 등 전용 실행기 사용 (PortfolioManager 미경유)

---

## 테스트

```bash
python -m pytest tests/ -v                        # 전체 테스트
python -m pytest tests/test_risk_manager.py -v     # 단일 파일
```

---

## 핵심 설계 원칙

- **전략-거래소 분리**: 전략 코드는 ccxt에 직접 의존하지 않음
- **전략 독립성**: 각 전략은 독립적으로 학습/백테스트/OOS 검증 가능
- **2계층 리스크**: 전략별 리스크 + 포트폴리오 전체 리스크
- **가상 포지션**: 동일 심볼 멀티 전략 시 내부 가상 포지션으로 관리
- **데이터 무결성**: trade_log.csv는 append-only, 상태 파일은 atomic write
- **ML 분리 학습**: 실거래 루프 내에서 재학습하지 않음
- **자동 재학습**: retrain.py로 데이터 수집→학습→OOS 검증→교체를 자동화, 실패 시 자동 복원

---

## 모델 버전 관리

유의미한 모델은 **git 커밋 + 태그**로 보존한다.

- 커밋 대상: `data/`, `strategies/**/models/`, `strategies/**/config.yaml`
- 주의: raw 재수집 시 값 미세 변동, Optuna 비결정성 → 좋은 모델은 즉시 커밋
- `.gitattributes`에서 모델 `.txt` 파일은 binary로 처리 (CRLF 변환 방지)

---

## 문서

| 파일 | 설명 |
|------|------|
| `docs/MULTI_STRATEGY_ARCHITECTURE.md` | 멀티 전략 포트폴리오 아키텍처 설계 |
| `docs/FUNDING_ARB_ARCHITECTURE.md` | 펀딩비 차익거래 아키텍처 |
| `docs/MAINNET_DEPLOYMENT_PLAN.md` | 메인넷 배포 계획 |
| `docs/REGRESSION_ARCHITECTURE.md` | 회귀 전략 아키텍처 |
| `docs/lgbm_backtest_metrics.md` | LightGBM 백테스트 지표 설명 |
