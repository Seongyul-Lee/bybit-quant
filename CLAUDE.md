# CLAUDE.md

## 프로젝트 개요
Bybit 거래소 기반 암호화폐 멀티 전략 퀀트 트레이딩 시스템.
백테스팅과 실거래가 동일한 전략 코드(`BaseStrategy.generate_signal`)를 공유한다.
현재 활성 전략: BTC/1h 모멘텀, ETH/1h 모멘텀 (LightGBM 2클래스, 롱 전용, 앙상블).

## 커맨드
```bash
.venv/Scripts/activate && pip install -r requirements.txt  # 환경 설정
python -m pytest tests/ -v                                  # 테스트
python train_lgbm.py --strategy btc_1h_momentum --no-optuna # 모델 학습
python train_lgbm.py --strategy eth_1h_momentum --optuna-trials 100
python backtest.py --strategy btc_1h_momentum               # 백테스트
python oos_validation.py --strategy btc_1h_momentum         # OOS 검증
python main.py --mode live                                   # 실거래 (포트폴리오)
```

## 아키텍처
```
실거래: Collector → Processor → 각 Strategy.generate_signal(df) → PortfolioManager.allocate()
        → PortfolioRiskManager → RiskManager → VirtualPositionTracker → OrderExecutor
백테스트: Parquet → Strategy.generate_signals_vectorized() → vectorbt → Reporter
ML학습:  Parquet → FeatureEngine(symbol별) → TripleBarrierLabeler → WalkForwardTrainer → 모델저장
```

### 핵심 규칙
- 전략은 거래소에 의존하지 않음 (ccxt는 collector/executor에서만)
- 2계층 리스크: 전략별 RiskManager + 포트폴리오 PortfolioRiskManager
- ML 전략은 별도 학습 → 모델 로드 (live 루프 내 재학습 금지)
- OrderExecutor 멱등성 보장 (동일 symbol/side 중복 주문 방지)

## 폴더 구조
```
config/              → .env(git제외), portfolio.yaml, risk_params.yaml
data/raw/bybit/      → {SYMBOL}USDT/{timeframe}/ 원본 OHLCV + funding_rate.parquet
data/processed/      → {SYMBOL}_{timeframe}_features.parquet
src/portfolio/       → manager.py, risk.py, virtual_position.py
src/                 → data/, strategies/base.py, risk/, execution/, analytics/, utils/
strategies/_common/  → features.py, labeler.py, trainer.py, evaluator.py (공통 ML)
strategies/btc_1h_momentum/  → strategy.py, config.yaml, models/
strategies/eth_1h_momentum/  → strategy.py, config.yaml, models/
```

## 코딩 컨벤션
- 모든 전략은 `BaseStrategy` 상속, `generate_signal(df) → (int, float)` 구현
- atomic write (tempfile → shutil.move), trade_log.csv는 append-only
- Parquet: `compression="snappy"` / 심볼: REST `BTC/USDT:USDT`, 파일명 `BTCUSDT`
- type hint + docstring 필수

## 작업 시 주의사항
- `config/.env` 수정/커밋 금지 (API 키)
- `data/` 직접 편집 금지, `data/raw/` 가공 금지
- CircuitBreaker 자동 리셋 코드 작성 금지
- Fail-Safe 우선: 오류 시 안전 종료

## LightGBM 전략 상세
- **라벨**: 2클래스 — 매수(1) / 비매수(0), 롱 전용
- **모델 파일**: `latest.txt`, `fold_XX.txt`, `feature_names.json`, `best_params.json`, `training_meta.json`
- **Walk-Forward**: 확장 윈도우(6개월+), Embargo 24봉, Optuna 첫 fold만 튜닝
- **앙상블**: config.yaml `ensemble_folds`로 여러 fold 평균 예측
- **FeatureEngine**: `symbol` config 키로 심볼별 펀딩비 경로 자동 결정

## 새 전략 추가
1. `strategies/{name}/` — `strategy.py`(BaseStrategy 상속) + `config.yaml` + `models/`
2. `main.py:load_strategy()`와 `backtest.py:run()`에 import 분기 추가
3. `config/portfolio.yaml`의 `active_strategies`에 전략 이름 추가

## 리스크 파라미터
전략당 포지션 20%, 전체 노출 60%, 동일 심볼 30%. 일손실 3%/포트폴리오 MDD 10%. CircuitBreaker: 연속손실 5회.

## Skills
| Skill | Purpose |
|-------|---------|
| `verify-implementation` | 모든 verify 스킬 순차 실행 → 통합 검증 보고서 |
| `verify-strategy` | 전략 컨벤션 검증 (BaseStrategy 상속, config 구조) |
| `verify-risk` | 리스크 규칙 검증 (파라미터 동기화, CircuitBreaker) |
| `verify-data-safety` | 데이터 안전성 검증 (atomic write, append-only, .env) |
| `verify-ml` | ML 규칙 검증 (라벨, 모델 파일, Walk-Forward, Live 재학습 금지) |
