# LightGBM 모델 결함 보고서

> 작성일: 2026-03-10
> 대상: `bybit-quant` — `lgbm_classifier` 전략
> 기준 백테스트: run20 (총 수익률 +6.93%, Val F1 0.369)
> 분석 범위: 라벨링, 학습 파이프라인, 하이퍼파라미터

---

## 수정 우선순위 요약

| 순위 | ID | 심각도 | 영역 | 핵심 | 상태 |
|------|----|--------|------|------|------|
| 1 | L-1 | CRITICAL | 라벨링 | SL/TP ↔ 배리어 불일치 | [ ] |
| 2 | H-1 | CRITICAL | 하이퍼 | num_leaves 62 과도 | [ ] |
| 3 | H-2 | CRITICAL | 하이퍼 | reg_lambda 0.012 무효 | [ ] |
| 4 | T-1 | CRITICAL | 파이프 | 전 fold 과적합 (H-1, H-2 원인) | [ ] |
| 5 | T-2 | HIGH | 파이프 | Optuna Fold 0 편향 | [ ] |
| 6 | T-3 | HIGH | 파이프 | 선택 모델 Val F1 ≈ 랜덤 | [ ] |
| 7 | H-3 | HIGH | 하이퍼 | min_child_samples 부족 | [ ] |
| 8 | T-5 | MEDIUM | 파이프 | 피처 중요도 편중 | [ ] |
| 9 | H-4 | MEDIUM | 하이퍼 | feature_fraction 과소 | [ ] |
| 10 | T-4 | MEDIUM | 파이프 | Early Stopping 무력화 | [ ] |
| 11 | L-2 | MEDIUM | 라벨링 | 배리어 배수 고정 | [ ] |
| 12 | H-5 | LOW | 하이퍼 | confidence_threshold | [ ] |

---

## 라벨링 결함

### L-1. 라벨 배리어와 SL/TP 불일치 (CRITICAL)

- **파일:** `strategies/lgbm_classifier/config.yaml`, `backtest.py:80-81`

**문제:**

모델은 대칭 ±1.81% (2.5×ATR) 움직임을 예측하도록 학습되었으나, 실제 거래는 SL -2% / TP +4% 비대칭으로 실행된다.

```
학습 목표:   ±1.81% 방향 예측 (대칭)
실행 조건:   -2% 손절 / +4% 익절 (비대칭)
→ 학습한 것과 실행하는 것이 다름
```

**데이터 검증 결과:**

| 항목 | 수치 |
|------|------|
| 24봉 내 TP(+4%) 도달 확률 | 10.4% |
| 24봉 내 SL(-2%) 도달 확률 | 35.2% |
| 24봉 내 둘 다 미도달 | 54.4% |
| SL이 TP보다 도달 빈도 | 3.4배 |

TP +4%는 24봉 내 10.4%만 도달하는 거의 불가능한 목표이며, 거래의 64.5%가 타임아웃으로 종료된다. 모델이 방향을 올바르게 예측해도 수익으로 전환되지 않는 구조적 문제.

**시뮬레이션 비교 (현재 모델 시그널 기반):**

| 설정 | 거래수 | 승률 | PF | 수익률 | MDD | Calmar | 보유기간 |
|------|--------|------|-----|--------|-----|--------|----------|
| 현재 SL2%/TP4% | 395 | 58.5% | 2.06 | +10.47% | -1.66% | 6.31 | 18.5봉 |
| 대칭 1.81% (2.5×ATR) | 469 | 64.4% | 2.02 | +9.70% | -1.40% | 6.91 | 13.0봉 |
| **대칭 1.45% (2.0×ATR)** | **513** | **66.1%** | **2.11** | **+9.40%** | **-0.69%** | **13.64** | **10.9봉** |
| 대칭 1.09% (1.5×ATR) | 636 | 64.2% | 1.85 | +6.52% | -0.67% | 9.74 | 7.8봉 |

**수정 방향:**

SL/TP를 라벨 배리어에 맞춰 대칭으로 변경한다. 시뮬레이션 결과 **2.0×ATR (±1.45%)** 대칭이 최적:

- MDD: -1.66% → -0.69% (58% 감소)
- Calmar: 6.31 → 13.64 (2.16배 향상)
- 승률: 58.5% → 66.1% (+7.6%p)

```yaml
# config.yaml 수정안
params:
  upper_barrier_multiplier: 2.0
  lower_barrier_multiplier: 2.0
  max_holding_period: 24

risk:
  # SL/TP를 ATR 기반 동적 값으로 변경하거나,
  # 고정 비율을 배리어와 일치시킴
  stop_loss_pct: 0.015     # 1.5% (2.0×ATR ≈ 1.45%에 슬리피지 마진)
  take_profit_pct: 0.015   # 대칭
```

**검증:**

배리어 변경 후 라벨 분포 확인 → 재학습 → 백테스트 수행.

---

### L-2. 배리어 배수가 고정값이라 시장 국면 변화에 무대응 (MEDIUM)

- **파일:** `strategies/lgbm_classifier/config.yaml:6-8`

**문제:**

ATR 자체는 변동성에 따라 변하지만, 배수(2.5×)는 고정이다. 고변동성 구간과 저변동성 구간에서 동일한 배수를 사용하면, 각 국면에서의 라벨 품질이 달라질 수 있다.

**현재 영향:**

ATR 기반이라 어느 정도 자동 조절이 되므로 즉시 수정이 필요한 수준은 아님. L-1 해결 후 성과를 관찰한 뒤 필요시 검토.

**수정 방향:**

변동성 레짐별 배수를 다르게 적용하거나, 퍼센타일 기반 동적 배수를 도입한다.

---

## 학습 파이프라인 결함

### T-1. 18개 fold 전부 과적합 — 구조적 문제 (CRITICAL)

- **파일:** `strategies/lgbm_classifier/trainer.py`, `models/training_meta.json`

**문제:**

전체 18개 Walk-Forward fold에서 예외 없이 과적합이 발생한다.

| 구분 | Train F1 | Val F1 | Gap |
|------|----------|--------|-----|
| 전체 평균 | 0.68 | 0.41 | 0.27 |
| 10K 미만 (Fold 0~7) | 0.69 | 0.42 | 0.269 |
| 10K 이상 (Fold 8~17) | 0.67 | 0.41 | 0.260 |

데이터 크기(10K 미만 vs 이상)와 과적합 간 상관계수는 -0.26으로 약한 음의 상관만 존재한다. 데이터가 커져도 Gap이 0.009밖에 줄지 않으므로, **데이터 부족이 아니라 모델 복잡도 과잉이 원인**이다.

**근본 원인:** H-1(num_leaves 62)과 H-2(reg_lambda 0.012)의 결합. H-1, H-2 수정 시 연쇄적으로 해결될 것으로 예상.

---

### T-2. Optuna가 Fold 0에서만 튜닝 후 전 fold에 동일 적용 (HIGH)

- **파일:** `strategies/lgbm_classifier/trainer.py:276-286`

**문제:**

Fold 0은 학습 데이터 **4,344행**으로 가장 작은 fold이다. 여기서 최적화된 파라미터가 16,774행짜리 Fold 17까지 동일하게 적용된다.

```python
# trainer.py:284 — Fold 0에서만 Optuna 실행
if self.n_optuna_trials > 0:
    logger.info(f"Fold 0: Optuna 튜닝 ({self.n_optuna_trials} trials)...")
    best_params = self.optimize_hyperparams(X_train_0, y_train_0, X_val_0, y_val_0)
```

**영향:**

- 4,344행에 최적화된 `num_leaves: 62`가 16,774행에도 적용됨
- 소규모 데이터에서 과적합 방향으로 튜닝된 파라미터가 전파됨

**수정 방향:**

- **방법 A (권장):** Optuna를 비활성화하고 보수적 기본값을 고정 사용
- **방법 B:** 각 fold에서 Optuna를 실행 (학습 비용 증가)
- **방법 C:** Optuna 탐색 범위를 보수적으로 제한

```python
# 방법 A: 보수적 기본값 (Optuna 비활성화 시)
best_params = {
    **self.FIXED_PARAMS,
    "num_leaves": 15,
    "min_child_samples": 100,
    "learning_rate": 0.05,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
}
```

---

### T-3. 선택된 모델(Fold 17)의 Val F1이 랜덤 수준 (HIGH)

- **파일:** `strategies/lgbm_classifier/trainer.py:331-355`

**문제:**

모델 선택 로직이 "최신 fold부터 역순으로, gap ≤ 0.3인 첫 번째"를 선택한다.

```
선택된 모델:    Fold 17 (Val F1: 0.369, Gap: 0.232)
3클래스 랜덤:   Val F1: 0.333
차이:          +0.036 (3.6%p)
```

Val F1 0.369는 랜덤 분류기(0.333) 대비 **3.6%p** 나은 수준에 불과하다. 가장 최근 데이터로 학습했다는 이점은 있지만, 예측력 자체가 부족한 모델이 선택되고 있다.

**수정 방향:**

H-1/H-2 수정으로 과적합이 완화되면 Val F1이 자연스럽게 상승할 것으로 예상. 추가로 모델 선택 기준에 Val F1 최소 임계값(예: 0.40)을 추가하는 것을 검토.

---

### T-4. Early Stopping이 과적합을 방지하지 못함 (MEDIUM)

- **파일:** `strategies/lgbm_classifier/trainer.py:221-225`

**문제:**

```
best_iteration 범위:   11 ~ 159
best_iteration 평균:   46.4
best_iteration 중앙값: 35
n_estimators 설정:     1000
→ 대부분 1000의 3~5%에서 조기 종료
```

모델이 빠르게 학습 데이터를 암기한 후 검증 손실이 즉시 상승하여 조기 종료된다. 이는 early_stopping patience(30)가 관대해서가 아니라, 트리 하나하나가 과도하게 복잡해서(num_leaves 62) 몇 개만 쌓여도 과적합이 시작되기 때문이다.

**수정 방향:**

H-1(num_leaves 축소) 수정 시 트리당 복잡도가 줄어, 더 많은 iteration이 필요해지면서 조기 종료 시점이 자연스럽게 늦춰질 것으로 예상. patience를 20으로 줄이는 것도 보조적으로 도움.

---

### T-5. 피처 중요도가 단일 피처에 과도하게 집중 (MEDIUM)

- **파일:** `strategies/lgbm_classifier/features.py`
- **데이터:** `models/training_meta.json:feature_importance`

**문제:**

| 피처 | 중요도 | 점유율 |
|------|--------|--------|
| ma_50_1d_ratio | 531 | 17.1% |
| ma_200_ratio | 273 | 8.8% |
| rsi_14_1d | 255 | 8.2% |
| atr_14_pct | 222 | 7.2% |
| adx_14 | 207 | 6.7% |
| 나머지 11개 | 1620 | 52.2% |

`ma_50_1d_ratio` 하나가 전체 중요도의 ~17%를 차지한다. 상위 3개(ma_50_1d_ratio, ma_200_ratio, rsi_14_1d)가 ~34%를 점유하여, 모델이 사실상 "일봉 MA50 대비 현재가 위치"에 의존하는 단순 추세 추종 수준이다.

**수정 방향:**

- H-4(feature_fraction 증가)로 피처 활용 다양성 확보
- 중요도 하위 피처(hour_sin: 95, hour_cos: 80, ma_10_ratio: 76)의 실제 기여도를 ablation 테스트로 확인
- 모델 복잡도 축소(H-1) 후 중요도 재분석

---

## 하이퍼파라미터 결함

### H-1. num_leaves 62가 과도하게 높음 (CRITICAL)

- **파일:** `strategies/lgbm_classifier/models/best_params.json`

**현재:** `num_leaves: 62`
**max_depth:** 6 (최대 가능 leaf: 64)

사실상 depth 제한이 없는 것과 동일하다. 각 트리가 62개 결정 경계를 만들고, 이것이 평균 46회 반복되면 약 2,852개의 결정 규칙이 생성되어 학습 데이터를 암기하게 된다.

**Fold별 leaf당 샘플 수:**

| Fold | 학습 행수 | leaf당 샘플 |
|------|-----------|------------|
| 0 (최소) | 4,344 | 70 |
| 7 | 9,430 | 152 |
| 17 (최대) | 16,774 | 271 |

leaf당 70샘플은 통계적으로 분할 자체는 가능하나, 62개 leaf가 동시에 학습되면 전체적으로 과적합을 유발한다.

**수정:**

```json
"num_leaves": 15
```

권장 범위: 15~23. max_depth 6과 결합 시 충분한 표현력을 유지하면서 과적합을 억제한다.

---

### H-2. reg_lambda 0.012로 L2 정규화가 사실상 비활성 (CRITICAL)

- **파일:** `strategies/lgbm_classifier/models/best_params.json`

**현재:** `reg_lambda: 0.012`

L2 정규화가 사실상 꺼진 상태이다. num_leaves 62와 결합되면 트리가 과도하게 복잡해져도 페널티가 없어, 전 fold 과적합의 직접적 원인이 된다.

참고로 `reg_alpha: 1.85`(L1 정규화)는 적절한 수준이다.

**수정:**

```json
"reg_lambda": 2.0
```

권장 범위: 1.0~5.0.

---

### H-3. min_child_samples 44가 num_leaves 62 대비 불충분 (HIGH)

- **파일:** `strategies/lgbm_classifier/models/best_params.json`

**현재:** `min_child_samples: 44`

num_leaves 62와 결합 시 leaf 분할 제약이 거의 없다. Fold 0(4,344행) 기준 `4,344 / 44 = 98`개 leaf까지 이론적으로 가능하므로, 62개 leaf 설정이 min_child_samples에 의해 제한되지 않는다.

**수정:**

```json
"min_child_samples": 100
```

권장 범위: 80~120. num_leaves를 15로 줄이면 제약 효과가 더 커진다.

---

### H-4. feature_fraction 0.33이 너무 낮음 (MEDIUM)

- **파일:** `strategies/lgbm_classifier/models/best_params.json`

**현재:** `feature_fraction: 0.33` → 16개 피처 중 ~5개만 각 트리에서 사용

피처 다양성 확보 목적이나, 5개 피처로는 트리가 유의미한 패턴을 잡기 어렵다. 특히 `ma_50_1d_ratio` 같은 강한 피처 1~2개에 과도하게 의존하게 만드는 원인이 된다.

**수정:**

```json
"feature_fraction": 0.6
```

권장 범위: 0.5~0.7. 16개 피처 기준 9~11개를 사용하게 되어 다양한 피처 조합을 활용 가능.

---

### H-5. confidence_threshold 0.40이 너무 낮을 수 있음 (LOW)

- **파일:** `strategies/lgbm_classifier/config.yaml:9`

**현재:** `confidence_threshold: 0.40`

3클래스 균등 확률이 0.333이므로, 0.40은 랜덤 대비 0.067 차이만으로 시그널을 발생시킨다. 모델 확신도가 낮은 시그널까지 거래로 이어질 수 있다.

**현재 시그널 분포:**

| 시그널 | 건수 | 비율 |
|--------|------|------|
| 매수(1) | 3,375 | 17.6% |
| 중립(0) | 13,331 | 69.5% |
| 매도(-1) | 2,476 | 12.9% |

**수정 방향:**

모델 예측력(Val F1) 개선 후 0.45~0.50으로 상향 조정을 검토한다. 현재 Val F1이 0.369인 상태에서 임계값을 올리면 거래 수가 과도하게 줄어들 수 있으므로, 모델 개선이 선행되어야 한다.

---

## 수정 전략

### Phase 1: 라벨링 + 하이퍼파라미터 일괄 수정 후 재학습

L-1, H-1, H-2, H-3, H-4를 동시에 적용한다.

**config.yaml 수정안:**

```yaml
params:
  model_path: strategies/lgbm_classifier/models/latest.txt
  feature_names_path: strategies/lgbm_classifier/models/feature_names.json
  confidence_threshold: 0.40
  upper_barrier_multiplier: 2.0   # 2.5 → 2.0
  lower_barrier_multiplier: 2.0   # 2.5 → 2.0
  max_holding_period: 24

risk:
  max_position_pct: 0.05
  stop_loss_pct: 0.015            # 0.02 → 0.015 (배리어와 정합)
  take_profit_pct: 0.015          # 0.04 → 0.015 (대칭)
```

**하이퍼파라미터 수정안 (Optuna 비활성화, 고정값 사용):**

```python
# trainer.py의 FIXED_PARAMS 또는 기본값
{
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 1000,
    "bagging_freq": 1,
    "max_depth": 6,
    "class_weight": "balanced",
    "verbose": -1,
    "num_leaves": 15,              # 62 → 15
    "min_child_samples": 100,      # 44 → 100
    "learning_rate": 0.05,         # 고정
    "reg_alpha": 1.5,              # 유지 (기존 1.85와 유사)
    "reg_lambda": 2.0,             # 0.012 → 2.0
    "feature_fraction": 0.6,       # 0.33 → 0.6
    "bagging_fraction": 0.7,       # 0.52 → 0.7
}
```

**실행:**

```bash
python train_lgbm.py --symbol BTCUSDT --timeframe 1h --no-optuna
python backtest.py --strategy lgbm_classifier
```

### Phase 2: 결과 평가 기준

| 지표 | 현재 (run20) | 목표 |
|------|-------------|------|
| Val F1 (평균) | 0.41 | > 0.43 |
| 과적합 Gap (평균) | 0.27 | < 0.15 |
| 백테스트 승률 | 43.7% | > 55% |
| MDD | -2.15% | < -1.5% |
| SL 초과 손실 | -4.91% | 없음 |

### Phase 3: 추가 개선 (Phase 2 통과 후)

- T-5: 피처 ablation 테스트 (중요도 하위 피처 제거 실험)
- H-5: confidence_threshold 조정 (0.45~0.50)
- L-2: 동적 배리어 배수 도입 검토

---

## 참고 데이터

### 전체 데이터 통계

| 항목 | 값 |
|------|-----|
| 데이터 기간 | 2024-01-01 ~ 2026-03-10 |
| 총 행 수 | 19,182 |
| 평균 ATR(14) | $584 |
| 평균 Close | $83,298 |
| ATR/Close 비율 | 0.73% |

### 현재 라벨 분포 (2.5×ATR 대칭)

| 라벨 | 건수 | 비율 | 24봉 후 평균 수익률 | 실제 방향 일치율 |
|------|------|------|---------------------|-----------------|
| 매수(1) | 7,228 | 37.8% | +1.789% | 83.0% |
| 중립(0) | 4,748 | 24.8% | +0.140% | 55.3% |
| 매도(-1) | 7,169 | 37.4% | -1.666% | 81.9% |

### 수정 후 예상 라벨 분포 (2.0×ATR 대칭)

| 라벨 | 건수 | 비율 |
|------|------|------|
| 매수(1) | 8,314 | 43.4% |
| 중립(0) | 2,666 | 13.9% |
| 매도(-1) | 8,165 | 42.6% |
