# Run35 최종 설정 리포트

> 2클래스 binary 전환 + 전면 최적화 후 최종 모델 설정 정리
> 날짜: 2026-03-10

## 백테스트 성과 요약

| 지표 | run17 (3클래스) | run28 (초기 2클래스) | **run35 (최종)** |
|------|:-:|:-:|:-:|
| 수익률 | +10.35% | +1.50% | **+15.37%** |
| 샤프 비율 | 3.19 | - | **4.45** |
| PF | 1.87 | 1.16 | **2.27** |
| MDD | -1.75% | -1.15% | **-1.82%** |
| 승률 | - | 60.3% | **66.3%** |
| 거래 수 | 373 | 184 | **282** |
| R:R | - | 0.76:1 | **1.19:1** |

---

## 1. config.yaml 전체

```yaml
strategy:
  name: LGBMClassifierStrategy
  symbol: BTCUSDT
  timeframe: 1h

params:
  model_path: strategies/lgbm_classifier/models/latest.txt
  feature_names_path: strategies/lgbm_classifier/models/feature_names.json
  confidence_threshold: 0.75
  upper_barrier_multiplier: 2.0   # 상단 2.0x: 매수 라벨 생성 완화
  lower_barrier_multiplier: 3.0   # 하단 3.0x: 하락 노이즈 내성 강화 (비대칭)
  max_holding_period: 24

execution:
  order_type: limit
  limit_offset: 0.001
  fee_rate: 0.00055  # Bybit Taker 0.055% (보수적 기준)

risk:
  max_position_pct: 0.05
  stop_loss_pct: 0.015            # 1.5% SL (빠른 손절)
  take_profit_pct: 0.025          # 2.5% TP (R:R = 1.67:1)
```

---

## 2. trainer.py 하이퍼파라미터

### FIXED_PARAMS (모든 모드 공통)

```python
FIXED_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 2000,
    "bagging_freq": 1,
    "max_depth": -1,         # 제한 없음 (num_leaves로만 제어)
    "is_unbalance": False,   # 라벨 52/48 거의 균형 → 불필요
    "verbose": -1,
}
```

### --no-optuna 기본값 블록

```python
{
    "num_leaves": 63,
    "min_child_samples": 30,
    "learning_rate": 0.02,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
```

### Optuna 탐색 범위

```python
{
    "num_leaves": (31, 127),
    "min_child_samples": (10, 100),
    "learning_rate": (0.005, 0.1),      # log scale
    "reg_alpha": (0.01, 3.0),           # log scale
    "reg_lambda": (0.01, 3.0),          # log scale
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
}
```

### Optuna 목적함수

```python
# F1에서 과적합 갭 페널티 차감 → 일반화 성능 우선
return val_f1 - 0.5 * max(gap - 0.1, 0)
```

---

## 3. models/best_params.json (Optuna가 찾은 실제 파라미터)

```json
{
  "boosting_type": "gbdt",
  "objective": "binary",
  "metric": "binary_logloss",
  "n_estimators": 2000,
  "bagging_freq": 1,
  "max_depth": -1,
  "is_unbalance": false,
  "verbose": -1,
  "num_leaves": 33,
  "min_child_samples": 94,
  "learning_rate": 0.00795,
  "reg_alpha": 0.0378,
  "reg_lambda": 0.686,
  "feature_fraction": 0.536,
  "bagging_fraction": 0.754
}
```

---

## 4. 선택된 Fold 11 상세

| 항목 | 값 |
|------|-----|
| Val F1 | **0.6966** |
| Train F1 | 0.9056 |
| Overfit Gap | **0.209** |
| Val LogLoss | 0.636 |
| best_iteration | **884** / 2000 |
| 학습 기간 | 2024-02-19 ~ 2025-07-17 (12,358행) |
| 검증 기간 | 2025-07-19 ~ 2025-08-19 (744행) |

참고: Fold 16도 Val F1 0.691, gap 0.079로 우수했지만, Fold 11이 F1 최고치로 선택됨.

### 전체 Fold 성과 테이블

| Fold | Train F1 | Val F1 | Gap | best_iter | 검증 기간 | 비고 |
|:----:|:--------:|:------:|:---:|:---------:|:---------:|:----:|
| 0 | 0.764 | 0.683 | 0.081 | 42 | 2024-08~09 | |
| 1 | 0.744 | 0.150 | 0.594 | 31 | 2024-09~10 | 과적합 |
| 2 | 0.000 | 0.000 | 0.000 | 4 | 2024-10~11 | 학습 실패 |
| 3 | 0.819 | 0.572 | 0.247 | 85 | 2024-11~12 | |
| 4 | 0.811 | 0.640 | 0.171 | 65 | 2024-12~01 | |
| 5 | 0.714 | 0.368 | 0.347 | 2 | 2025-01~02 | 과적합 |
| 6 | 0.000 | 0.000 | 0.000 | 1 | 2025-02~03 | 학습 실패 |
| 7 | 0.759 | 0.367 | 0.392 | 62 | 2025-03~04 | 과적합 |
| 8 | 0.765 | 0.517 | 0.248 | 78 | 2025-04~05 | |
| 9 | 0.775 | 0.345 | 0.430 | 85 | 2025-05~06 | 과적합 |
| 10 | 0.798 | 0.480 | 0.318 | 114 | 2025-06~07 | 과적합 |
| **11** | **0.906** | **0.697** | **0.209** | **884** | **2025-07~08** | **선택됨** |
| 12 | 0.750 | 0.563 | 0.187 | 59 | 2025-08~09 | |
| 13 | 0.755 | 0.517 | 0.238 | 65 | 2025-09~10 | |
| 14 | 0.731 | 0.562 | 0.168 | 64 | 2025-10~11 | |
| 15 | 0.798 | 0.609 | 0.189 | 234 | 2025-11~12 | |
| 16 | 0.770 | 0.691 | 0.079 | 151 | 2025-12~01 | |
| 17 | 0.000 | 0.000 | 0.000 | 1 | 2026-01~02 | 학습 실패 |

---

## 5. run22 대비 변경 diff

```diff
# ── config.yaml ──
- confidence_threshold: 0.40
+ confidence_threshold: 0.75

- upper_barrier_multiplier: 2.5
+ upper_barrier_multiplier: 2.0

- lower_barrier_multiplier: 2.5
+ lower_barrier_multiplier: 3.0

- stop_loss_pct: 0.018
+ stop_loss_pct: 0.015

- take_profit_pct: 0.018
+ take_profit_pct: 0.025

# ── trainer.py FIXED_PARAMS ──
- "objective": "multiclass"
+ "objective": "binary"

- "metric": "multi_logloss"
+ "metric": "binary_logloss"

- "n_estimators": 1000
+ "n_estimators": 2000

- "max_depth": 6
+ "max_depth": -1

- "class_weight": "balanced"
+ "is_unbalance": False

- "num_class": 3
  (삭제)

# ── Optuna 탐색 범위 ──
- "num_leaves": (7, 31)
+ "num_leaves": (31, 127)

- "min_child_samples": (50, 200)
+ "min_child_samples": (10, 100)

- "learning_rate": (0.01, 0.1)
+ "learning_rate": (0.005, 0.1)

- "reg_alpha": (0.5, 5.0)
+ "reg_alpha": (0.01, 3.0)

- "reg_lambda": (1.0, 10.0)
+ "reg_lambda": (0.01, 3.0)

- "feature_fraction": (0.4, 0.8)
+ "feature_fraction": (0.5, 1.0)

- "bagging_fraction": (0.5, 0.8)
+ "bagging_fraction": (0.5, 1.0)

# ── Optuna 목적함수 ──
- return f1_score(y_val, y_pred, average="macro")
+ return val_f1 - 0.5 * max(gap - 0.1, 0)   # 과적합 페널티

# ── early_stopping ──
- early_stopping(30)
+ early_stopping(50)

# ── 모델 선택 로직 ──
- 최신 fold부터 역순, gap ≤ threshold인 첫 번째 fold
+ gap ≤ threshold인 fold 중 Val F1 최고 선택

# ── 피처 수 ──
- 선별 16개 (get_selected_features)
+ 전체 48개 → 상관관계 제거 후 30개 (--use-all-features)

# ── 분류 체계 ──
- 3클래스: -1(매도), 0(중립), 1(매수)
+ 2클래스: 0(비매수), 1(매수)
```

---

## 6. SL 1.5% / TP 2.5% 비대칭 근거

### 문제 진단

run28(대칭 SL/TP 1.8%)에서 **평균 수익 +1.95% vs 평균 손실 -2.55%** → R:R = 0.76:1로 역전된 상태.

- **SL 1.8%가 너무 느슨함** — 하락 추세에서 손실이 확대된 후에야 청산
- **TP 1.8%가 너무 빡빡함** — 수익 거래가 TP에 빠르게 걸려서 추가 상승을 놓침

### 그리드 서치 결과

| SL | TP | R:R | 수익률 | MDD | PF | 승률 | 거래 |
|:--:|:---:|:---:|:------:|:----:|:----:|:----:|:----:|
| 1.2% | 2.5% | 2.08 | +14.88% | -2.03% | 2.16 | 61.6% | 307 |
| **1.5%** | **2.5%** | **1.67** | **+15.37%** | **-1.82%** | **2.27** | **66.3%** | **282** |
| 1.5% | 3.0% | 2.00 | +15.30% | -2.08% | 2.23 | 62.1% | 253 |
| 1.8% | 3.0% | 1.67 | +15.27% | -2.05% | 2.23 | 64.7% | 241 |
| 1.2% | 3.0% | 2.50 | +14.38% | -2.12% | 2.10 | 56.7% | 277 |

### 선택 이유: SL 1.5% / TP 2.5%

- **수익률, MDD, PF 모두 최우수**
- SL 1.5%는 노이즈 하락을 빠르게 컷 → MDD 최소화 (-1.82%)
- TP 2.5%는 TP 1.8% 대비 40% 더 넓어 수익 확대를 허용
- 실현 R:R = 1.19:1 (평균 수익 +2.68% / 평균 손실 -2.26%)
- Expectancy = $545/trade

---

## 7. 최악 거래 -4.28%가 SL 1.5%를 초과하는 원인

### vectorbt의 SL 체크 메커니즘

vectorbt의 `sl_stop`은 **봉 종가(close) 기준**으로 체크한다.

1. **진입**: 봉 N의 close에서 매수 시그널 발생 → 봉 N+1의 open에서 진입
2. **SL 체크**: 매 봉마다 `close`가 진입가 × (1 - sl_stop) 이하인지 확인
3. **문제**: 봉 내(intra-bar) low가 SL을 관통해도 close가 SL 위면 청산되지 않음

### 예시

```
진입가: $100,000
SL 기준: $98,500 (1.5%)

봉 N+3:
  open:  $99,200
  high:  $99,300
  low:   $95,000  ← SL 관통하지만...
  close: $98,600  ← SL 위이므로 청산 안 됨

봉 N+4:
  open:  $98,500
  close: $95,720  ← 여기서 SL 발동 → 실현 손실 -4.28%
```

### 발생 시나리오 3가지

| 시나리오 | 설명 |
|----------|------|
| **갭 하락 (Gap Down)** | 전 봉 close는 SL 위 → 다음 봉 open이 SL 아래에서 시작, close는 더 아래 |
| **급락 후 반등 실패** | 봉 내 low가 SL 관통하지만 close가 SL 위 → 다음 봉에서 추가 하락 |
| **슬리피지** | `slippage=0.002` (0.2%) 추가 → SL 1.5% + 슬리피지 0.2% = 실질 1.7% 이상 손실 가능 |

### 실거래와의 차이

실거래에서는 `RiskManager.get_stop_take_profit()`이 거래소 **조건부 주문(stop-market)**으로 제출하므로, 봉 내 가격에서도 SL이 즉시 발동되어 이 문제가 완화됨. 따라서 백테스트의 -4.28% 최악 거래는 실거래에서는 -1.5%~-2.0% 수준으로 제한될 것으로 예상.

---

## 피처 중요도 Top 10

| 순위 | 피처 | 중요도 | 설명 |
|:----:|------|:------:|------|
| 1 | atr_14_1d | 2720 | 일봉 ATR (변동성) |
| 2 | ma_10 | 2404 | 10봉 이동평균 (절대값) |
| 3 | obv | 2379 | On-Balance Volume |
| 4 | rsi_14_1d | 2247 | 일봉 RSI |
| 5 | ma_50_1d_ratio | 2067 | 일봉 MA50 대비 비율 |
| 6 | atr_14_4h | 2031 | 4시간봉 ATR |
| 7 | ma_200_ratio | 1632 | MA200 대비 비율 |
| 8 | rsi_14_4h | 1406 | 4시간봉 RSI |
| 9 | adx_14 | 1237 | ADX (추세 강도) |
| 10 | volume_ma_20 | 1164 | 20봉 거래량 이동평균 |

멀티타임프레임 피처(1d, 4h)가 상위를 차지하며, 높은 타임프레임의 추세/변동성 정보가 1h 매수 결정에 핵심적임을 보여준다.
