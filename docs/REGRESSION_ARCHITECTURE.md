# Stage 1 아키텍처: 분류(Classification) → 회귀(Regression) + 양방향 트레이딩

## 1. 개요

### 1.1 목표

현재 시스템의 구조적 한계(롱 전용 + 이진 분류 + 고정 포지셔닝)를 해결하여,
**양방향(롱+숏) + 연속 예측(회귀) + 동적 포지셔닝** 구조로 전환한다.

### 1.2 핵심 변경 요약

| 항목 | 현재 (분류) | 변경 후 (회귀) |
|------|-----------|-------------|
| 모델 | LGBMClassifier (이진 분류) | LGBMRegressor (회귀) |
| 예측값 | 매수 확률 0.0~1.0 | 예상 수익률 -5%~+5% |
| 라벨 | 1(매수) / 0(비매수) | 연속값: 미래 N봉 수익률 |
| 시그널 | 1(매수) or 0(대기) | 1(롱) / -1(숏) / 0(대기) |
| 방향 | 롱 전용 | 롱 + 숏 양방향 |
| 포지션 크기 | 고정 20% | 예측 크기에 비례 (5%~25%) |
| SL/TP | 고정 2.1% / 2.1% | 예측 크기에 비례 (동적) |
| 시장 활용 | 상승장 ~40% | 상승 + 하락 ~80% |

### 1.3 변경 범위

```
재활용 (변경 없음):
  ✅ strategies/_common/features.py     — 피처 엔진 전체
  ✅ src/portfolio/risk.py              — 5단계 방어 로직
  ✅ src/portfolio/virtual_position.py  — 가상 포지션 추적
  ✅ src/risk/manager.py                — 전략 레벨 리스크
  ✅ src/execution/executor.py          — 주문 실행 (숏 이미 지원)
  ✅ src/data/collector.py              — 데이터 수집
  ✅ src/utils/notify.py                — 텔레그램 알림
  ✅ config/portfolio.yaml              — 포트폴리오 설정 (확장만)
  ✅ retrain.py                         — 재학습 파이프라인 (최소 수정)

변경 필요:
  🔄 strategies/_common/labeler.py      — 회귀 라벨러 추가
  🔄 strategies/_common/trainer.py      — Regressor 모드 추가
  🔄 strategies/_common/evaluator.py    — 회귀 메트릭 추가
  🔄 src/strategies/base.py             — 시그널 인터페이스 확장
  🔄 strategies/*/strategy.py           — 회귀 전략 클래스 신규
  🔄 strategies/*/config.yaml           — 회귀 파라미터 추가
  🔄 backtest.py                        — 양방향 + 동적 SL/TP
  🔄 oos_validation.py                  — 양방향 시뮬레이션
  🔄 train_lgbm.py                      — regressor 모드 CLI
  🔄 src/portfolio/manager.py           — 롱/숏 동시 관리
  🔄 main.py                            — 숏 주문 실행 로직
```

---

## 2. 라벨링 (Labeling)

### 2.1 현재: Triple Barrier 이진 라벨

```
입력:  OHLCV + ATR
출력:  1 (상단 배리어 터치 = 매수) or 0 (그 외 = 비매수)
문제:  +0.1%든 +5%든 같은 라벨 1 → 크기 정보 손실
```

### 2.2 변경: Forward Return 연속 라벨

```python
# strategies/_common/labeler.py에 추가

class ForwardReturnLabeler:
    """미래 N봉 수익률을 연속값으로 라벨링.
    
    각 봉에서 미래 forward_period 봉 동안의 수익률을 계산하되,
    ATR 기반 배리어로 클리핑하여 극단값을 제한한다.
    
    Parameters:
        forward_period: 미래 수익률 계산 기간 (봉 수). 기본 24 (1h × 24 = 1일).
        barrier_atr_mult: 수익률 클리핑 배리어 (ATR 배수). 기본 3.0.
            예: ATR 2%이고 mult 3.0이면 수익률을 ±6% 범위로 클리핑.
        use_log_return: True면 로그 수익률 사용 (분포 대칭성).
    """
    
    def __init__(
        self,
        forward_period: int = 24,
        barrier_atr_mult: float = 3.0,
        use_log_return: bool = False,
    ):
        self.forward_period = forward_period
        self.barrier_atr_mult = barrier_atr_mult
        self.use_log_return = use_log_return
    
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """미래 수익률 라벨 생성.
        
        Args:
            df: OHLCV + atr_14 데이터프레임.
        
        Returns:
            연속값 라벨 시리즈 (미래 수익률, %).
            마지막 forward_period 봉은 NaN.
        """
        close = df["close"]
        
        # 미래 수익률 계산
        if self.use_log_return:
            future_return = np.log(close.shift(-self.forward_period) / close)
        else:
            future_return = (close.shift(-self.forward_period) - close) / close
        
        # ATR 기반 클리핑 (극단값 제한)
        if "atr_14" in df.columns:
            atr_pct = df["atr_14"] / close
            clip_bound = atr_pct * self.barrier_atr_mult
            future_return = future_return.clip(
                lower=-clip_bound, upper=clip_bound
            )
        
        return future_return.rename("label")
```

### 2.3 라벨 설계 근거

**왜 forward_period = 24인가:**
- 현재 max_holding_period가 24봉(1h × 24 = 1일)
- 기존 배리어 기반 라벨과 동일한 시간 지평
- 너무 짧으면(6봉) 노이즈, 너무 길면(48봉) 예측 불가

**왜 ATR 클리핑인가:**
- 극단적 수익률(flash crash 등)은 모델이 학습할 수 있는 패턴이 아님
- 클리핑은 기존 배리어의 역할과 동일 (SL/TP 한도 = 실현 가능 수익 한도)
- 모델이 현실적 범위의 수익률에 집중하도록 유도

**왜 로그 수익률 옵션인가:**
- 로그 수익률은 대칭적 분포를 가져 회귀 모델에 유리
- 단순 수익률은 -100%~+∞ 비대칭, 로그 수익률은 -∞~+∞ 대칭
- 기본값은 단순 수익률(직관적), 성능 비교 후 전환 가능

---

## 3. 모델 (Model)

### 3.1 현재: LGBMClassifier

```python
# trainer.py FIXED_PARAMS
{
    "objective": "binary",
    "metric": "binary_logloss",
    # ...
}
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)  # y: 0 or 1
proba = model.predict_proba(X_val)[:, 1]  # 매수 확률
```

### 3.2 변경: LGBMRegressor

```python
# trainer.py에 REGRESSOR_FIXED_PARAMS 추가
REGRESSOR_FIXED_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "huber",        # MAE보다 이상치에 강건
    "metric": "mae",             # 해석 용이
    "n_estimators": 2000,
    "bagging_freq": 1,
    "max_depth": -1,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)  # y: 연속값 (미래 수익률)
pred = model.predict(X_val)  # 예상 수익률
```

### 3.3 Huber Loss 선택 이유

```
MSE (L2):   이상치에 민감 → flash crash가 학습을 지배
MAE (L1):   이상치에 강건하지만 0 근처에서 미분 불연속
Huber:      작은 오차는 MSE처럼, 큰 오차는 MAE처럼 → 최적 균형
```

Huber의 delta 파라미터는 Optuna로 탐색: `delta ∈ [0.5, 2.0]`

### 3.4 WalkForwardTrainer 확장

```python
class WalkForwardTrainer:
    """mode='classifier' 또는 mode='regressor' 지원.
    
    mode에 따라:
    - FIXED_PARAMS / REGRESSOR_FIXED_PARAMS 자동 선택
    - LGBMClassifier / LGBMRegressor 자동 선택
    - 평가 메트릭 자동 전환 (F1 → MAE/IC)
    """
    
    def __init__(
        self,
        mode: str = "classifier",  # "classifier" | "regressor"
        # ... 기존 파라미터 동일
    ):
        self.mode = mode
```

**평가 메트릭 전환:**

| 분류 (기존) | 회귀 (신규) |
|------------|-----------|
| F1 Macro | MAE (Mean Absolute Error) |
| Log Loss | IC (Information Coefficient = Spearman corr) |
| Accuracy | Directional Accuracy (방향 맞춤 비율) |
| 과적합 갭: train_f1 - val_f1 | 과적합 갭: val_mae - train_mae |

**IC (Information Coefficient)가 핵심 메트릭:**
- 예측값과 실제 수익률의 Spearman 상관계수
- IC > 0.05면 통계적으로 유의미한 예측력
- 대형 퀀트 펀드들의 표준 평가 지표

---

## 4. 전략 (Strategy)

### 4.1 BaseStrategy 인터페이스 확장

```python
# src/strategies/base.py 수정

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """매매 신호 + 확신도.
        
        Returns:
            (signal, confidence) 튜플.
            signal:     1(롱) / -1(숏) / 0(대기)
            confidence: 예측 강도의 절대값 (0.0 ~ 1.0 정규화)
                        분류: 매수 확률
                        회귀: |예상 수익률| / 스케일링 팩터
        """
        pass
```

**하위 호환성:** 기존 분류 전략은 signal이 1 또는 0만 반환하므로 기존 코드와 호환됨.

### 4.2 LGBMRegressorStrategy (신규)

```python
# strategies/_common/regressor_strategy.py (신규 파일)

class LGBMRegressorStrategy(BaseStrategy):
    """LightGBM 회귀 기반 양방향 전략.
    
    모델이 예상 수익률을 예측하고:
    - 양수 + 임계값 초과 → 롱 (signal = 1)
    - 음수 + 임계값 초과 → 숏 (signal = -1)
    - 임계값 미만 → 대기 (signal = 0)
    
    Config 키 (신규):
        min_pred_threshold: 진입 최소 예측값 (기본 0.005 = 0.5%)
        max_position_scale: 최대 포지션 스케일 (기본 2.0)
        sl_atr_mult: 동적 SL ATR 배수 (기본 2.0)
        tp_atr_mult: 동적 TP ATR 배수 (기본 3.0)
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.feature_engine = FeatureEngine(config)
        self.min_pred_threshold = config.get("min_pred_threshold", 0.005)
        self.max_position_scale = config.get("max_position_scale", 2.0)
        self.sl_atr_mult = config.get("sl_atr_mult", 2.0)
        self.tp_atr_mult = config.get("tp_atr_mult", 3.0)
        # 모델 로드 (앙상블 지원)
        self.models = self._load_ensemble_models()
        self.feature_names = self._load_feature_names()
    
    def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """예상 수익률 기반 양방향 시그널.
        
        Returns:
            (signal, confidence)
            signal: 1(롱), -1(숏), 0(대기)
            confidence: |예측값| 정규화 (0.0 ~ 1.0)
        """
        df_feat = self.feature_engine.compute_all_features(df)
        last_row = df_feat[self.feature_names].iloc[[-1]]
        
        if last_row.isna().any(axis=1).iloc[0]:
            return 0, 0.0
        
        pred = self._predict(last_row)[0]  # 예상 수익률 (예: +0.012 = +1.2%)
        
        # 펀딩비 필터 적용
        effective_threshold = self._get_adaptive_threshold(df_feat.iloc[-1], pred)
        
        abs_pred = abs(pred)
        if abs_pred < effective_threshold:
            return 0, 0.0
        
        signal = 1 if pred > 0 else -1
        # confidence: 예측 크기를 0~1로 정규화
        confidence = min(abs_pred / (self.min_pred_threshold * self.max_position_scale), 1.0)
        
        return signal, float(confidence)
    
    def get_dynamic_sl_tp(self, df: pd.DataFrame, signal: int) -> tuple[float, float]:
        """예측 크기 기반 동적 SL/TP 계산.
        
        Args:
            df: OHLCV + 피처 데이터프레임.
            signal: 1(롱) or -1(숏).
        
        Returns:
            (sl_pct, tp_pct) — 0.0~1.0 범위의 비율.
        """
        atr_pct = df["atr_14"].iloc[-1] / df["close"].iloc[-1]
        sl_pct = atr_pct * self.sl_atr_mult
        tp_pct = atr_pct * self.tp_atr_mult
        
        # 최소/최대 제한
        sl_pct = max(0.01, min(sl_pct, 0.05))  # 1% ~ 5%
        tp_pct = max(0.01, min(tp_pct, 0.08))  # 1% ~ 8%
        
        return sl_pct, tp_pct
```

### 4.3 포지션 사이징 로직

```python
def calculate_position_size(
    self,
    portfolio_value: float,
    confidence: float,
    base_pct: float = 0.20,
) -> float:
    """confidence에 비례하는 동적 포지션 사이징.
    
    confidence가 높을수록 큰 포지션:
        confidence 0.3 → base_pct × 0.5 = 10%
        confidence 0.5 → base_pct × 0.75 = 15%
        confidence 1.0 → base_pct × 1.0 = 20%
    
    하한: base_pct × 0.25 (최소 진입 크기)
    상한: base_pct × 1.0 (기존 상한 유지)
    """
    scale = 0.25 + 0.75 * confidence  # 0.25 ~ 1.0
    return portfolio_value * base_pct * scale
```

---

## 5. 백테스트 / OOS 검증

### 5.1 양방향 시뮬레이션 (oos_validation.py)

현재 `simulate_period()`는 롱 전용이야. 양방향 확장:

```python
def simulate_period_v2(
    df_period, signals_period, confidences_period,
    sl_fn, tp_fn,  # 동적 SL/TP 함수
    max_hold=24,
    base_position_pct=0.20,
    fee_per_side=0.00055,
    slippage_per_side=0.0,
):
    """양방향 + 동적 포지셔닝 시뮬레이션.
    
    Args:
        signals_period: 1(롱), -1(숏), 0(대기) 시리즈.
        confidences_period: 0.0~1.0 시리즈.
        sl_fn: (atr_pct, signal) → sl_pct 함수.
        tp_fn: (atr_pct, signal) → tp_pct 함수.
    """
    close = df_period["close"].values
    high = df_period["high"].values
    low = df_period["low"].values
    sigs = signals_period.values
    confs = confidences_period.values
    
    trades = []
    i = 0
    
    while i < n:
        if sigs[i] == 0:
            i += 1
            continue
        
        direction = sigs[i]  # 1(롱) or -1(숏)
        entry_price = close[i]
        
        # 동적 SL/TP
        atr_pct = atr_values[i] / close[i]
        sl_pct = sl_fn(atr_pct, direction)
        tp_pct = tp_fn(atr_pct, direction)
        
        # 동적 포지션 크기
        scale = 0.25 + 0.75 * confs[i]
        position_pct = base_position_pct * scale
        
        if direction == 1:  # 롱
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:  # 숏
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)
        
        # 배리어 체크
        for j in range(i + 1, min(i + 1 + max_hold, n)):
            if direction == 1:  # 롱
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price
            else:  # 숏
                hit_tp = low[j] <= tp_price
                hit_sl = high[j] >= sl_price
            
            if hit_tp and hit_sl:
                exit_return = -sl_pct  # 보수적: 동시 터치는 SL
                exit_type = "sl"
                break
            elif hit_tp:
                exit_return = tp_pct
                exit_type = "tp"
                break
            elif hit_sl:
                exit_return = -sl_pct
                exit_type = "sl"
                break
        else:
            # 타임아웃
            if direction == 1:
                exit_return = (close[exit_bar] - entry_price) / entry_price
            else:
                exit_return = (entry_price - close[exit_bar]) / entry_price
            exit_type = "timeout"
        
        trades.append({
            "direction": direction,
            "entry_bar": i,
            "exit_bar": exit_bar,
            "return": exit_return,
            "position_pct": position_pct,
            "type": exit_type,
        })
        
        i = exit_bar + 1
```

### 5.2 성과 메트릭

기존 PF에 추가로 회귀 전략 전용 메트릭:

```python
{
    # 기존
    "trades": int,
    "win_rate": float,
    "pf": float,
    "total_return": float,
    "mdd": float,
    
    # 신규
    "long_trades": int,        # 롱 거래 수
    "short_trades": int,       # 숏 거래 수
    "long_win_rate": float,    # 롱 승률
    "short_win_rate": float,   # 숏 승률
    "long_pf": float,          # 롱 Profit Factor
    "short_pf": float,         # 숏 Profit Factor
    "avg_position_pct": float, # 평균 포지션 크기
    "direction_accuracy": float, # 방향 예측 정확도
}
```

---

## 6. 포트폴리오 매니저

### 6.1 롱/숏 동시 관리

현재 `PortfolioManager.allocate()`는 `signal == 1`만 처리.
`signal == -1`(숏)도 처리하도록 확장:

```python
def allocate(self, signals, portfolio_value, virtual_tracker,
             portfolio_scale=1.0, strategy_scales=None):
    """양방향 시그널 처리.
    
    signal == 1  → 롱 주문 생성
    signal == -1 → 숏 주문 생성
    signal == 0  → 무시
    """
    for name, (signal, confidence) in signals.items():
        if signal == 0:
            continue
        
        direction = "long" if signal == 1 else "short"
        
        # 동적 포지션 크기
        scale = 0.25 + 0.75 * confidence
        effective_pct = self.position_pct * scale * portfolio_scale
        strat_scale = strategy_scales.get(name, 1.0) if strategy_scales else 1.0
        effective_pct *= strat_scale
        
        orders.append({
            "strategy": name,
            "symbol": sym,
            "direction": direction,  # "long" or "short"
            "size_usd": portfolio_value * effective_pct,
            "confidence": confidence,
        })
```

### 6.2 동일 심볼 롱/숏 충돌 방지

BTC에서 롱 모멘텀과 숏 모멘텀이 동시에 시그널을 낼 수는 없지만
(같은 모델이 동시에 +와 -를 예측하지 않으므로),
**BTC 모멘텀(롱) + BTC 평균회귀(숏)**은 동시에 발생 가능.

정책: **넷 포지션으로 합산.**

```python
# 같은 심볼에 대해:
# btc_momentum: 롱 $200
# btc_mean_reversion: 숏 $150
# → 넷 롱 $50만 실행
```

이 로직은 `VirtualPositionTracker.get_real_position()`에서 이미 처리 가능.

---

## 7. 실행 (main.py)

### 7.1 숏 주문 실행

```python
# 현재: 롱만 지원
if order["direction"] == "long":
    executor.open_long(symbol, size, sl, tp)

# 변경: 양방향
if order["direction"] == "long":
    executor.open_long(symbol, size, sl, tp)
elif order["direction"] == "short":
    executor.open_short(symbol, size, sl, tp)
```

Bybit의 USDT 무기한 선물은 **숏 포지션을 네이티브로 지원**하므로
ccxt의 `create_order(side="sell")` 호출만 추가하면 됨.

### 7.2 청산 로직 확장

```python
# 현재: 롱 포지션만 감지
if existing_pos and existing_pos["side"] == "long":
    # SL/TP 체크

# 변경: 양방향 포지션 감지
if existing_pos:
    if existing_pos["side"] == "long":
        # 롱 SL/TP 체크
    elif existing_pos["side"] == "short":
        # 숏 SL/TP 체크 (가격 방향 반대)
```

---

## 8. Config 구조

### 8.1 회귀 전략 config.yaml 예시

```yaml
strategy:
  name: btc_1h_momentum_v2
  type: lgbm_regressor            # 신규: 분류/회귀 구분
  symbol: BTCUSDT
  timeframe: 1h

params:
  model_path: strategies/btc_1h_momentum_v2/models/latest.txt
  feature_names_path: strategies/btc_1h_momentum_v2/models/feature_names.json
  models_dir: strategies/btc_1h_momentum_v2/models
  ensemble_folds: [10, 12]

  # 회귀 전용 파라미터
  mode: regressor
  min_pred_threshold: 0.005       # 최소 예측값 0.5% 미만이면 대기
  max_position_scale: 2.0         # confidence 스케일링 상한
  forward_period: 24              # 라벨링 미래 수익률 기간

  # 동적 SL/TP
  sl_atr_mult: 2.0               # SL = ATR × 2.0
  tp_atr_mult: 3.0               # TP = ATR × 3.0
  min_sl_pct: 0.01               # SL 최소 1%
  max_sl_pct: 0.05               # SL 최대 5%
  min_tp_pct: 0.01               # TP 최소 1%
  max_tp_pct: 0.08               # TP 최대 8%

  # 펀딩비 필터 (기존 유지)
  funding_filter:
    enabled: true
    zscore_thresholds:
      - {zscore_below: 0, confidence: 0.003}
      - {zscore_below: 2, confidence: 0.005}

  # OI 필터 (기존 유지)
  oi_filter:
    enabled: true
    block_zscore: 1.0

retrain:
  enabled: true
  interval_days: 30
  window_type: sliding
  window_months: 15
  min_pf_ratio: 0.9
  auto_ensemble: true
  ensemble_n_folds: 2

execution:
  order_type: limit
  limit_offset: 0.001
  fee_rate: 0.00055

risk:
  max_position_pct: 0.20         # 기본 포지션 (confidence로 스케일링)
  min_position_pct: 0.05         # 최소 포지션
```

---

## 9. 학습 파이프라인 (train_lgbm.py)

### 9.1 CLI 확장

```bash
# 기존 (분류)
python train_lgbm.py --strategy btc_1h_momentum

# 신규 (회귀)
python train_lgbm.py --strategy btc_1h_momentum_v2 --mode regressor \
    --forward-period 24 --barrier-atr-mult 3.0

# --mode 옵션:
#   classifier (기본값, 하위 호환)
#   regressor  (회귀 모드)
```

### 9.2 Optuna 탐색 공간 (회귀 전용)

```python
REGRESSOR_OPTUNA_SPACE = {
    "num_leaves": (8, 31),
    "min_child_samples": (50, 300),
    "learning_rate": (0.005, 0.05),
    "reg_alpha": (0.1, 10.0),
    "reg_lambda": (0.1, 10.0),
    "feature_fraction": (0.3, 0.8),
    "bagging_fraction": (0.5, 0.9),
    "huber_delta": (0.5, 2.0),     # 신규: Huber loss delta
}
```

### 9.3 Fold 선택 기준 변경

```
분류 (기존):
  1. val_f1_macro > 0.40
  2. overfit_gap < 0.30
  3. val_f1 높은 순

회귀 (신규):
  1. val_ic > 0.03 (최소 예측력)
  2. overfit_gap_mae < 0.005 (MAE 기준 과적합)
  3. val_ic 높은 순 (IC = Spearman 상관)
  4. val_directional_accuracy > 0.52 (방향 정확도)
```

---

## 10. 디렉토리 구조

### 10.1 기존 전략 보존 + 신규 v2 전략 추가

```
strategies/
  _common/
    features.py           # 변경 없음
    labeler.py            # ForwardReturnLabeler 추가
    trainer.py            # mode='regressor' 지원 추가
    evaluator.py          # 회귀 메트릭 추가
    regressor_strategy.py # 신규: LGBMRegressorStrategy 기본 클래스
  
  # 기존 분류 전략 (보존, 건드리지 않음)
  btc_1h_momentum/
  eth_1h_momentum/
  btc_1h_mean_reversion/
  
  # 신규 회귀 전략 (v2)
  btc_1h_momentum_v2/
    config.yaml
    strategy.py           # LGBMRegressorStrategy 상속
    models/
  eth_1h_momentum_v2/
    config.yaml
    strategy.py
    models/
  btc_1h_mean_reversion_v2/
    config.yaml
    strategy.py
    models/
```

**기존 전략을 보존하는 이유:**
- 검증된 OOS 결과를 무효화하지 않음
- 회귀 모델 성능이 분류보다 나쁘면 즉시 롤백 가능
- 분류 vs 회귀 A/B 비교 가능
- 회귀가 검증되면 기존 분류 전략은 비활성화

---

## 11. 구현 Phase 분할

### Phase 0: 공통 인프라 확장 (기반 작업)
- `ForwardReturnLabeler` 구현 (`strategies/_common/labeler.py`)
- `WalkForwardTrainer` regressor 모드 추가 (`strategies/_common/trainer.py`)
- `ModelEvaluator` 회귀 메트릭 추가 (`strategies/_common/evaluator.py`)
- `BaseStrategy` 시그널 인터페이스 확장 (`src/strategies/base.py`)
- `LGBMRegressorStrategy` 기본 클래스 생성 (`strategies/_common/regressor_strategy.py`)
- `train_lgbm.py` regressor 모드 CLI 추가

**검증:** `python train_lgbm.py --strategy btc_1h_momentum_v2 --mode regressor` 실행 → fold 학습 성공 + IC 출력

### Phase 1: BTC 모멘텀 회귀 전략 구현
- `strategies/btc_1h_momentum_v2/` 디렉토리 생성
- `config.yaml` 작성 (회귀 파라미터)
- `strategy.py` 구현 (LGBMRegressorStrategy 상속)
- 학습: `python train_lgbm.py --strategy btc_1h_momentum_v2 --mode regressor`
- `oos_validation.py` 양방향 시뮬레이션 확장
- OOS 검증: PV PF, 롱/숏 분리 메트릭

**검증:** BTC 회귀 모델 IC > 0.03, PV PF > 1.5 (보수적 비용)

### Phase 2: ETH + 평균회귀 회귀 전략
- `strategies/eth_1h_momentum_v2/` 구현
- `strategies/btc_1h_mean_reversion_v2/` 구현
- 각 전략 학습 + OOS 검증
- 3전략 상관관계 분석

**검증:** 3전략 모두 IC > 0.03, PV PF > 1.5

### Phase 3: 포트폴리오 통합 + 양방향 실행
- `portfolio.yaml`에 v2 전략 추가
- `PortfolioManager.allocate()` 양방향 확장
- `backtest.py` 양방향 + 동적 SL/TP 지원
- `portfolio_backtest.py` 회귀 전략 포트폴리오 백테스트
- 분류 포트폴리오 vs 회귀 포트폴리오 A/B 비교

**검증:** 회귀 포트폴리오 PV PF ≥ 분류 포트폴리오 PV PF

### Phase 4: 실거래 통합
- `main.py` 숏 주문 실행 로직 추가
- `executor.py` `open_short()` / `close_short()` 확장
- `VirtualPositionTracker` 숏 포지션 추적
- testnet 검증: 롱 + 숏 주문 실행 확인
- `retrain.py` regressor 모드 지원

**검증:** testnet에서 롱/숏 주문 체결 + SL/TP 동작 확인

---

## 12. 성공 기준

| 메트릭 | 분류 (현재) | 회귀 (목표) |
|--------|-----------|-----------|
| PV PF (보수적) | 2.09 (BTC) | ≥ 1.8 |
| Strict OOS PF | 0.56 (BTC) | ≥ 0.9 (양방향) |
| 시장 활용 | ~40% (롱만) | ~80% (양방향) |
| 월 거래 수 | ~20건 | ~40건+ |
| IC (예측력) | N/A | > 0.03 |
| 방향 정확도 | ~55% (분류) | > 52% (회귀) |

**PV PF가 분류보다 약간 낮아도 OK — Strict OOS PF 개선과 거래 빈도 증가가 더 중요.**
분류에서 PV PF 2.09이지만 Strict OOS 0.56인 것보다,
회귀에서 PV PF 1.8이지만 Strict OOS 0.9인 것이 실전에서 더 나아.

---

## 13. 핵심 주의사항

1. **기존 분류 전략은 절대 수정하지 않는다.** v2 전략을 별도 디렉토리에 만들고, 검증 후 교체.

2. **회귀 모델은 분류보다 과적합에 취약하다.** 연속값을 예측하면 모델이 노이즈까지 학습하기 쉬워. 강한 정규화(Huber loss, 높은 reg_alpha/lambda, 낮은 num_leaves)가 필수.

3. **IC 0.03은 낮아 보이지만 충분하다.** Medallion의 승률이 50.75%인 것처럼, 미세한 예측력도 거래 횟수와 비용 관리로 수익으로 전환 가능.

4. **숏 포지션은 롱보다 리스크가 크다.** 상승은 무한하므로 숏의 최대 손실은 이론적으로 무한. 반드시 SL 설정 + 포지션 크기 제한. 숏의 max_position_pct를 롱보다 작게(15%) 설정하는 것도 고려.

5. **동적 SL/TP의 최소값을 반드시 설정.** ATR이 극도로 낮은 구간에서 SL/TP가 0.3% 같이 작아지면 수수료만으로 손실. min_sl_pct = 1%, min_tp_pct = 1% 하한 필수.

6. **Phase 0에서 기존 분류 모드가 깨지지 않는지 반드시 검증.** `--mode classifier`(기본값)로 기존 학습이 동일하게 동작하는지 테스트 포함.
