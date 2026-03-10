# 백테스트 결함 보고서

> 작성일: 2026-03-10
> 대상: lgbm_classifier 전략 백테스트 파이프라인
> 백테스트 결과: 총 수익률 283,263%, Sharpe 2.03, 승률 74.1%, 1,067 거래

---

## Defect #1 — 포지션 사이징 미적용 (100% 자본 투입)

- **심각도**: 치명적
- **파일**: `backtest.py:83-92`
- **현상**: `vbt.Portfolio.from_signals()`에 `size`, `size_type` 파라미터가 없어 vectorbt 기본값(가용 자본 100%)으로 매 거래가 실행됨
- **영향**: config의 `max_position_pct: 0.05`(5%)가 완전히 무시됨. 1,067회 거래에서 100% 복리가 적용되어 283,263%라는 비현실적 수익률의 핵심 원인
- **현재 코드**:
  ```python
  portfolio = vbt.Portfolio.from_signals(
      close=df["close"],
      entries=(signal_series == 1),
      exits=(signal_series == -1),
      fees=0.0004,
      slippage=0.001,
      init_cash=1_000_000,
      sl_stop=sl_stop,
      tp_stop=tp_stop,
      # size, size_type 파라미터 없음
  )
  ```
- **수정 방향**: `size=config.risk.max_position_pct` (기본 0.05), `size_type="targetpercent"` 추가. 실거래 RiskManager의 `calculate_atr_position_size()`와 동일한 로직을 백테스트에도 반영하는 것이 이상적

---

## Defect #2 — 수수료율 과소 설정

- **심각도**: 치명적
- **파일**: `backtest.py:87`
- **현상**: `fees=0.0004` (0.04% 편도, 왕복 0.08%)로 설정됨
- **영향**: Bybit 실제 수수료(Taker 0.055%, Maker 0.02%)보다 낮게 설정되어 수익률 과대 추정

  | 항목 | 백테스트 | Bybit Taker | Bybit Maker |
  |------|---------|-------------|-------------|
  | 편도 수수료 | 0.04% | 0.055% | 0.02% |
  | 왕복 수수료 | 0.08% | 0.11% | 0.04% |

- **수정 방향**: config의 `execution.order_type`이 `limit`이므로 Maker 기준 0.02%도 가능하지만, 슬리피지로 Taker 체결될 가능성을 감안하여 보수적으로 `fees=0.0006` (0.06%) 이상 권장. 또는 config에서 수수료율을 명시적으로 관리

---

## Defect #3 — Sharpe Ratio 연환산 오류

- **심각도**: 치명적
- **파일**: `src/analytics/reporter.py:138-140`
- **현상**: 1시간봉 returns를 `sqrt(252)` (일간 기준)으로 연환산함
- **영향**: 1h 데이터에는 `sqrt(8760)` (=365×24) 또는 `sqrt(252×24)=sqrt(6048)`을 사용해야 함. 현재 `sqrt(252)≈15.87`을 사용하고 있으나 올바른 값은 `sqrt(8760)≈93.6`이므로 연환산 배율이 약 5.9배 과소 적용되어 Sharpe 수치 자체가 무의미
- **현재 코드**:
  ```python
  sharpe_ratio = float(
      np.sqrt(252) * excess_returns.mean() / excess_returns.std()
  ) if excess_returns.std() > 0 else 0.0
  ```
- **수정 방향**: 타임프레임 정보를 `calculate_metrics()`에 전달하여 연환산 계수를 동적으로 계산. 타임프레임별 연간 봉 수 매핑 필요:
  - `1h` → 8760 (365×24)
  - `4h` → 2190 (365×6)
  - `1d` → 365

---

## Defect #4 — 커스텀 메트릭의 total_trades / win_rate 오류

- **심각도**: 중요
- **파일**: `src/analytics/reporter.py:124-163`, `backtest.py:102-107`
- **현상**:
  1. `calculate_metrics(returns)`에서 `total_trades = len(returns)` → 19,166 (봉 수)을 반환. 실제 거래 수는 1,067
  2. `win_rate = (returns > 0).mean()` → 봉 기준 22.9%. vectorbt의 거래 기준 승률은 74.1%
  3. `profit_factor`도 봉 기준 수익/손실 비율로 거래 기준과 불일치 (1.89 vs 1.60)
- **영향**: 텔레그램 알림과 저장된 JSON 리포트의 지표가 vectorbt 지표와 모순됨. 의사결정에 잘못된 수치 사용 가능
- **수정 방향**: `portfolio` 객체에서 직접 거래 기반 메트릭 추출하거나, `calculate_metrics()`에 `portfolio.trades` 정보를 전달하여 거래 단위 지표 계산

---

## Defect #5 — 전 fold 과적합 + Best Fold Cherry-Picking

- **심각도**: 중요
- **파일**: `strategies/lgbm_classifier/trainer.py:282-307`, `train_lgbm.py:101-109`
- **현상**:
  1. 18개 fold **전부** 과적합 경고 (평균 Train F1: 0.82, 평균 Val F1: 0.40, 평균 Gap: 0.42)
  2. 저장되는 모델은 Val F1이 가장 높은 단일 fold (Fold 16, Val F1: 0.5797)
  3. 실거래에서는 평균 성능(F1 0.40)에 가까울 가능성이 높음

  | Fold | Train F1 | Val F1 | Gap |
  |------|----------|--------|-----|
  | 0 | 0.840 | 0.308 | 0.532 |
  | 6 | 0.728 | 0.491 | 0.237 |
  | 15 | 0.896 | 0.516 | 0.380 |
  | 16 (best) | 0.907 | 0.580 | 0.327 |
  | 17 (최신) | 0.670 | 0.289 | 0.381 |
  | **평균** | **0.82** | **0.40** | **0.42** |

- **수정 방향**:
  - **모델 선택**: "최고 Val F1 fold" 대신 (A) 최근 fold 모델 사용 또는 (B) 전 fold 앙상블 고려
  - **과적합 완화**: `num_leaves` 축소, `min_child_samples` 증가, `reg_alpha`/`reg_lambda` 강화, `feature_fraction`/`bagging_fraction` 축소
  - **조기 종료 조건 강화**: 현재 `early_stopping(50)` → 더 작은 patience (예: 20~30)
  - **과적합 심각도 기준**: gap > 0.2일 때 경고만 하지 말고, gap > 0.3인 fold는 모델 후보에서 제외하는 로직 고려

---

## Defect #6 — 라벨 불균형 (비대칭 배리어)

- **심각도**: 중요
- **파일**: `strategies/lgbm_classifier/labeler.py`, `strategies/lgbm_classifier/config.yaml:11-12`
- **현상**:
  - `upper_barrier_multiplier=2.0` (상방), `lower_barrier_multiplier=1.0` (하방)
  - 하방 배리어가 상방의 절반 거리에 위치하여 매도(-1) 터치 확률이 구조적으로 높음
  - 결과 라벨 분포: `{-1: 11503 (64%), 0: 952 (5%), 1: 5509 (31%)}`
- **영향**:
  - `class_weight="balanced"` 적용에도 불구하고 모델이 매도 편향될 수 있음
  - 중립(0) 라벨이 5%에 불과하여 중립 예측 정확도가 낮을 가능성
  - 실제 시장에서 매수 기회를 과소 포착할 위험
- **수정 방향**:
  - 대칭 배리어 (`upper=1.5`, `lower=1.5`) 적용 후 성능 비교
  - 또는 현재 비대칭 의도가 있다면 (숏 편향 전략) 그 근거를 config에 문서화
  - 중립(0) 라벨 비율이 너무 낮으면 `max_holding_period` 축소를 고려 (더 빨리 중립 판정)

---

## Defect #7 — 벡터화 vs 순차 신호 불일치 가능성

- **심각도**: 경미
- **파일**: `strategies/lgbm_classifier/strategy.py:76-108` vs `strategy.py:45-74`
- **현상**:
  - `generate_signals_vectorized(df)` (백테스트): 전체 데이터에서 한번에 `compute_all_features()` 호출
  - `generate_signal(df)` (실거래): 현재 봉까지의 데이터로 매번 `compute_all_features()` 호출
  - rolling 연산의 초기 warmup 구간(ma_200의 경우 처음 200봉)에서 미세 차이 발생 가능
  - 멀티타임프레임 resample (`_add_multitimeframe_features`)이 전체 데이터 vs 부분 데이터에서 경계값 처리가 다를 수 있음
- **영향**: 실제로는 warmup 이후 구간에서 동일한 결과를 반환하므로 전체 수익률에 미치는 영향은 미미. 단, 엣지 케이스에서 신호 1~2개 차이 가능
- **수정 방향**: 검증 테스트 추가 — 동일 데이터로 `generate_signal()` 순차 호출과 `generate_signals_vectorized()` 결과를 비교하는 단위 테스트 작성

---

## 수정 우선순위

| 순위 | Defect | 기대 효과 |
|------|--------|----------|
| 1 | #1 포지션 사이징 | 수익률이 현실적 수준으로 보정됨 |
| 2 | #3 Sharpe 연환산 | 성과 지표 신뢰성 확보 |
| 3 | #4 메트릭 오류 | 리포트/알림 정확성 확보 |
| 4 | #2 수수료율 | 수익률 추가 보정 |
| 5 | #5 과적합 | 실거래 성능 예측 정확도 향상 |
| 6 | #6 라벨 불균형 | 매수/매도 균형 개선 |
| 7 | #7 벡터화 불일치 | 백테스트-실거래 일관성 보장 |
