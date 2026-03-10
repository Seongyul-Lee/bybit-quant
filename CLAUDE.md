# CLAUDE.md

## 프로젝트 개요
Bybit 거래소 기반 암호화폐 퀀트 트레이딩 시스템.
백테스팅과 실거래가 동일한 전략 코드(`BaseStrategy.generate_signal`)를 공유한다.
전략: MA Crossover(규칙 기반), LightGBM 2클래스 분류(ML 기반, 롱 전용, 앙상블 지원).

## 커맨드
```bash
.venv/Scripts/activate && pip install -r requirements.txt  # 환경 설정
python -m pytest tests/ -v                                  # 테스트
python train_lgbm.py --no-optuna                            # 모델 학습
python train_lgbm.py --optuna-trials 100                    # Optuna 튜닝 학습
python backtest.py --strategy lgbm_classifier               # 백테스트
python oos_validation.py                                    # OOS 검증
python run_experiment.py --threshold 0.48 --no-optuna       # 학습→OOS 자동화
python main.py --strategy lgbm_classifier --mode live       # 실거래
```

## 아키텍처
```
실거래: Collector → Processor → Strategy.generate_signal(df) → RiskManager.check_all()
        → OrderExecutor.execute() → trade_log.csv
백테스트: Parquet → Strategy.generate_signals_vectorized() → vectorbt → Reporter
ML학습:  Parquet → FeatureEngine → TripleBarrierLabeler(2클래스) → WalkForwardTrainer → 모델저장
```

### 핵심 규칙
- 전략은 거래소에 의존하지 않음 (ccxt는 collector/executor에서만)
- RiskManager는 전략 신호보다 항상 우선 (check_all 실패 시 주문 미실행)
- ML 전략은 별도 학습 → 모델 로드 (live 루프 내 재학습 금지)
- OrderExecutor 멱등성 보장 (동일 symbol/side 중복 주문 방지)

## 폴더 구조
```
config/              → .env(git제외), risk_params.yaml, current_state.json
data/raw/bybit/      → 원본 OHLCV (가공 금지)
data/processed/      → 피처 Parquet ({symbol}_{timeframe}_features.parquet)
src/                 → data/, strategies/base.py, risk/, execution/, analytics/, utils/
strategies/          → 전략별 폴더 (strategy.py + config.yaml)
  lgbm_classifier/   → features.py, labeler.py, trainer.py, evaluator.py, models/
reports/             → backtest/, live/, trades/trade_log.csv
train_lgbm.py        → 학습 CLI
oos_validation.py    → OOS 검증
run_experiment.py    → 학습→검증 자동화
```

## 코딩 컨벤션
- 모든 전략은 `BaseStrategy` 상속, `generate_signal(df) → int` 구현 (1=매수, 0=중립)
- atomic write (tempfile → shutil.move), trade_log.csv는 append-only
- Parquet: `compression="snappy"` / 심볼: REST `BTC/USDT:USDT`, 파일명 `BTCUSDT`
- type hint + docstring 필수

## 작업 시 주의사항
- `config/.env` 수정/커밋 금지 (API 키)
- `data/` 직접 편집 금지, `data/raw/` 가공 금지
- CircuitBreaker 자동 리셋 코드 작성 금지
- Fail-Safe 우선: 오류 시 안전 종료

## 모델 버전 관리
- 유의미한 모델은 **git 커밋 + 태그**로 보존 (`model/{전략}/{run번호}`)
- 커밋 대상: `data/`, `strategies/**/models/`, `strategies/**/config.yaml`
- 주의: raw 재수집 시 값 미세 변동, Optuna 비결정성 → 좋은 모델은 즉시 커밋

## LightGBM 전략 상세
- **라벨**: 2클래스 — 매수(1) / 비매수(0), 롱 전용
- **모델 파일**: `latest.txt`(best fold), `fold_XX.txt`(앙상블), `feature_names.json`, `best_params.json`, `training_meta.json`
- **Walk-Forward**: 확장 윈도우(6개월+), Embargo 24봉, Optuna 첫 fold만 튜닝
- **앙상블**: config.yaml `ensemble_folds`로 여러 fold 평균 예측

## 새 전략 추가
1. `strategies/{name}/` — `strategy.py`(BaseStrategy 상속) + `config.yaml`
2. `main.py:load_strategy()`와 `backtest.py:run()`에 import 분기 추가

## 리스크 파라미터
단일 포지션 5%, 동시 3개, 레버리지 3배. 일손실 3%/월손실 10% 초과 시 중단. 연속손실 5회 시 CircuitBreaker.

## Skills
| Skill | Purpose |
|-------|---------|
| `verify-implementation` | 모든 verify 스킬 순차 실행 → 통합 검증 보고서 |
| `verify-strategy` | 전략 컨벤션 검증 (BaseStrategy 상속, config 구조) |
| `verify-risk` | 리스크 규칙 검증 (파라미터 동기화, CircuitBreaker) |
| `verify-data-safety` | 데이터 안전성 검증 (atomic write, append-only, .env) |
| `verify-ml` | ML 규칙 검증 (라벨, 모델 파일, Walk-Forward, Live 재학습 금지) |
| `manage-skills` | 스킬 생성/업데이트 및 CLAUDE.md 관리 |
