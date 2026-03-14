# 펀딩비 차익거래 전략 아키텍처

## 1. 개요

### 1.1 전략 원리

크립토 무기한 선물(Perpetual Futures)에는 **펀딩비(Funding Rate)**라는 메커니즘이 존재한다.
8시간마다 롱 포지션 보유자와 숏 포지션 보유자 사이에서 수수료가 이전된다.

```
펀딩비 > 0 (양수): 롱이 숏에게 지불 → 숏 보유자가 수익
펀딩비 < 0 (음수): 숏이 롱에게 지불 → 롱 보유자가 수익
```

**펀딩비 차익거래(Funding Rate Arbitrage):**
현물 매수(롱) + 선물 매도(숏)를 동시에 실행하여 **델타 중립(시장 방향 무관)**을 유지하면서,
양수 펀딩비를 수취하는 전략.

```
포지션:  현물 BTC 1개 매수 + 선물 BTC 1개 숏
→ BTC 가격 상승: 현물 +$1000, 선물 -$1000 → 순손익 $0
→ BTC 가격 하락: 현물 -$1000, 선물 +$1000 → 순손익 $0
→ 펀딩비 수취: 8시간마다 숏 포지션 크기 × 펀딩비율 = 순수익
```

### 1.2 수익 구조

```
Bybit 펀딩비 결제 시간 (UTC): 00:00, 08:00, 16:00 (하루 3회)
BTC 평균 펀딩비: ~0.01% / 8시간 (시장 상황에 따라 -0.05% ~ +0.3%)

기본 계산 (레버리지 없이):
  포지션 $1,000
  일 수익: $1,000 × 0.01% × 3 = $0.30/일
  연 수익: $0.30 × 365 = $109.50/년 = 10.95%

레버리지 2배:
  자본 $1,000 → 포지션 $2,000
  연 수익: $219/년 = 21.9%

ML 강화 (펀딩비 예측 + 동적 포지셔닝):
  높은 펀딩비 예측 시 포지션 확대, 낮을 때 축소
  기대: 연 25~31%, 샤프 2.0+
```

### 1.3 리스크

```
리스크 1: 베이시스 리스크 (Basis Risk)
  현물-선물 가격 차이가 벌어져 일시적 평가 손실 발생
  → 시간이 지나면 수렴하므로 일시적. 청산만 안 당하면 OK

리스크 2: 청산 리스크 (Liquidation Risk)
  선물 숏 포지션이 급등 시 청산될 수 있음
  → 레버리지 제한 (최대 2~3배)으로 방어
  → 현물이 담보 역할

리스크 3: 음수 펀딩비 전환
  펀딩비가 음수로 전환되면 숏이 지불해야 함
  → 음수 전환 시 포지션 축소/청산
  → ML로 음수 전환 사전 예측

리스크 4: 거래소 리스크
  거래소 해킹, 파산 (FTX 사례)
  → Bybit 단일 거래소 의존 → 자본의 일부만 투입

리스크 5: ADL (Auto-Deleveraging)
  극단적 시장 상황에서 수익 포지션이 강제 청산
  → 발생 확률 매우 낮지만, 대규모 자본에서 고려 필요
```

### 1.4 기존 v1 모멘텀과의 관계

```
                    v1 모멘텀 롱          펀딩비 차익거래
엣지 원천           가격 예측 (모델)       구조적 비효율 (펀딩비 메커니즘)
시장 방향 의존      상승장에서만 수익       방향 무관
상관관계            -                      v1과 거의 0 (완전 독립)
리스크 프로필       높은 변동성             낮은 변동성, 안정적 수익
기대 수익           불확실 (연 -5%~+10%)   안정적 (연 15~30%)
샤프 비율           ~0.5 (실전 추정)       1.4~2.3
```

**포트폴리오에서의 역할: v1은 상승장 알파, 펀딩비는 전천후 안정 수익.**

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────┐
│                    main.py (통합 루프)                    │
├──────────────────────┬──────────────────────────────────┤
│   v1 모멘텀 전략     │     펀딩비 차익거래 전략          │
│   (기존, 변경 없음)  │     (신규)                        │
│                      │                                   │
│   BTC/ETH 롱 전용    │   현물 매수 + 선물 숏 동시 실행   │
│   1h 봉 기반         │   8h 펀딩비 주기 기반             │
│   LightGBM 분류      │   LightGBM 회귀 (펀딩비 예측)    │
├──────────────────────┴──────────────────────────────────┤
│                 PortfolioManager                         │
│            (자본 배분: v1 40% + 차익 40% + 현금 20%)     │
├─────────────────────────────────────────────────────────┤
│                 PortfolioRiskManager                     │
│              (5단계 MDD 방어, 일일 손실 등)              │
├─────────────┬───────────────────────────────────────────┤
│ Spot Executor│        Perp Executor (기존)               │
│ (신규)       │                                           │
├─────────────┴───────────────────────────────────────────┤
│              Bybit Exchange (Unified Trading Account)    │
│              현물 + 무기한 선물 동일 계정                 │
└─────────────────────────────────────────────────────────┘
```

### 2.2 디렉토리 구조

```
strategies/
  _common/                          # 기존 공통 모듈 (변경 없음)
  btc_1h_momentum/                  # 기존 v1 (변경 없음)
  eth_1h_momentum/                  # 기존 v1 (변경 없음)
  btc_1h_mean_reversion/            # 기존 v1 (변경 없음)
  
  funding_arb/                      # 신규: 펀딩비 차익거래
    __init__.py
    config.yaml                     # 전략 설정
    strategy.py                     # FundingArbStrategy (BaseStrategy 상속)
    predictor.py                    # FundingRatePredictor (LightGBM 회귀)
    hedger.py                       # DeltaHedger (현물-선물 헤지 관리)
    models/                         # 펀딩비 예측 모델

src/
  data/
    collector.py                    # 확장: 현물 가격 수집 추가
    funding_collector.py            # 신규: 실시간 펀딩비 + 베이시스 수집
  execution/
    executor.py                     # 기존 (선물 전용)
    spot_executor.py                # 신규: 현물 매수/매도 실행
    arb_executor.py                 # 신규: 현물+선물 동시 실행 (원자적)

config/
  portfolio.yaml                    # 확장: funding_arb 전략 추가
  funding_arb.yaml                  # 신규: 차익거래 전용 상세 설정
```

---

## 3. 데이터 파이프라인

### 3.1 수집 데이터

```
기존 (변경 없음):
  ✅ OHLCV 1h (BTC, ETH) — v1 전략용
  ✅ 펀딩비 이력 — 이미 fetch_funding_rate_bulk() 존재
  ✅ OI (미결제약정)

신규:
  🔄 실시간 펀딩비 — 다음 펀딩비 결제까지 남은 시간 + 예상 펀딩비
  🔄 현물 가격 — 현물-선물 베이시스 계산용
  🔄 롱/숏 비율 — 시장 센티먼트 (Bybit API 지원)
  🔄 선물 프리미엄/디스카운트 — 베이시스 트렌드
```

### 3.2 Bybit API 엔드포인트

```python
# 1. 실시간 펀딩비 (다음 결제 예정)
exchange.fetch_funding_rate("BTC/USDT:USDT")
# → {"fundingRate": 0.0001, "fundingTimestamp": ..., "nextFundingTimestamp": ...}

# 2. 현물 가격
spot_exchange = ccxt.bybit({"options": {"defaultType": "spot"}})
spot_ticker = spot_exchange.fetch_ticker("BTC/USDT")
# → {"last": 71000.0, "bid": 70999.5, "ask": 71000.5}

# 3. 선물 가격
perp_ticker = exchange.fetch_ticker("BTC/USDT:USDT")
# → {"last": 71050.0, "bid": 71049.5, "ask": 71050.5}

# 4. 베이시스
basis = (perp_price - spot_price) / spot_price
# → +0.0007 = +0.07% (선물 프리미엄)

# 5. 롱/숏 비율 (Bybit 전용 API)
# GET /v5/market/account-ratio
# → {"list": [{"buyRatio": "0.55", "sellRatio": "0.45", "timestamp": "..."}]}
```

### 3.3 피처 엔지니어링 (펀딩비 예측용)

```python
# strategies/funding_arb/predictor.py

class FundingRateFeatureEngine:
    """펀딩비 예측용 피처 엔지니어링.
    
    피처 카테고리:
    1. 펀딩비 자체 패턴 (자기상관)
    2. 베이시스 (현물-선물 스프레드)
    3. 시장 구조 (OI, 롱숏 비율)
    4. 가격 패턴 (변동성, 추세)
    """
    
    FEATURES = {
        # 펀딩비 패턴 (핵심 — 자기상관이 높아 예측력 강함)
        "fr_current":        "현재 펀딩비",
        "fr_ma_3":           "최근 3회(24h) 평균 펀딩비",
        "fr_ma_7":           "최근 7회(56h) 평균 펀딩비",
        "fr_ma_21":          "최근 21회(7일) 평균 펀딩비",
        "fr_std_7":          "최근 7회 펀딩비 표준편차",
        "fr_zscore":         "펀딩비 z-score (현재 - MA21) / STD21",
        "fr_trend":          "펀딩비 3회 회귀 기울기 (추세)",
        "fr_positive_ratio": "최근 7회 중 양수 비율",
        
        # 베이시스 (현물-선물 스프레드)
        "basis_current":     "현재 베이시스 %",
        "basis_ma_24h":      "24시간 평균 베이시스",
        "basis_trend":       "베이시스 추세 (기울기)",
        
        # 시장 구조
        "oi_change_8h":      "최근 8시간 OI 변화율",
        "oi_change_24h":     "최근 24시간 OI 변화율",
        "long_short_ratio":  "롱/숏 비율",
        
        # 가격 패턴
        "return_8h":         "최근 8시간 수익률",
        "return_24h":        "최근 24시간 수익률",
        "volatility_24h":    "24시간 변동성",
        "rsi_14":            "RSI(14) — 1h 봉 기준",
        
        # 시간 피처
        "hour_utc":          "UTC 시간 (0~23)",
        "day_of_week":       "요일 (0~6)",
    }
```

### 3.4 라벨링 (예측 대상)

```python
# 예측 대상: 다음 펀딩비 결제 시 실제 펀딩비율
# 
# 타겟: next_funding_rate (8시간 후 펀딩비)
# 단위: 비율 (예: 0.0001 = 0.01%)
#
# 학습 데이터: 8시간 간격 (펀딩비 결제 주기)
#   X: 결제 시점의 피처
#   y: 다음 결제 시점의 실제 펀딩비
```

---

## 4. ML 모델: 펀딩비 예측

### 4.1 모델 구조

```python
# strategies/funding_arb/predictor.py

class FundingRatePredictor:
    """LightGBM 기반 펀딩비 예측기.
    
    입력: 현재 시점의 피처 벡터
    출력: 다음 펀딩비 결제 시 예상 펀딩비율
    
    v1 모멘텀의 "가격이 오를까"와 근본적으로 다름:
    - 펀딩비는 자기상관이 높아 예측이 상대적으로 쉬움
    - 예측 대상이 연속값이지만, 범위가 매우 좁음 (-0.05% ~ +0.3%)
    - 예측 정확도보다 방향(양수/음수)이 더 중요
    """
    
    MODEL_PARAMS = {
        "boosting_type": "gbdt",
        "objective": "huber",
        "metric": "mae",
        "n_estimators": 500,       # 데이터가 적으므로 (8h 간격)
        "num_leaves": 15,
        "min_child_samples": 30,
        "learning_rate": 0.02,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "seed": 42,
    }
    
    def predict_next_funding(self, features: dict) -> float:
        """다음 펀딩비 예측.
        
        Returns:
            예상 펀딩비율 (예: 0.0003 = 0.03%)
        """
        ...
    
    def predict_confidence(self, features: dict) -> float:
        """예측 확신도 (앙상블 분산 기반).
        
        여러 fold 모델의 예측값 분산이 작으면 확신도 높음.
        
        Returns:
            0.0 ~ 1.0
        """
        ...
```

### 4.2 학습 파이프라인

```
데이터: 8시간 간격 펀딩비 이력 (2024-01 ~ 현재)
       BTC: ~2,700건, ETH: ~2,700건
       
Walk-Forward (기존 인프라 재활용):
  학습: 최근 6개월 (Sliding Window)
  검증: 1개월
  Embargo: 1 결제 주기 (8시간)

평가 메트릭:
  MAE: 예측 오차 절대값 (낮을수록 좋음)
  방향 정확도: 양수/음수 방향 맞춤 비율 (> 65% 목표)
  수익 시뮬레이션: 예측 기반 포지셔닝 vs 정적 포지셔닝 수익 비교
```

### 4.3 왜 펀딩비 예측은 가격 예측보다 쉬운가

```
가격 예측:
  - 노이즈가 매우 크고, 신호 대 잡음비(SNR)가 낮음
  - 외부 요인(뉴스, 규제, 매크로)에 크게 영향받음
  - 자기상관이 거의 없음 (랜덤워크에 가까움)
  - IC 0.03~0.08 → 매우 얇은 엣지

펀딩비 예측:
  - 구조적으로 자기상관이 높음 (양수 펀딩비는 지속되는 경향)
  - 시장 구조(OI, 롱숏 비율)와 직접적 인과관계
  - 범위가 좁고 평균 회귀 특성 (극단적 펀딩비는 빠르게 정상화)
  - IC 0.3~0.5 예상 → 두꺼운 엣지
```

---

## 5. 실행 (Execution)

### 5.1 Bybit Unified Trading Account

Bybit의 통합 거래 계정(UTA)은 **현물과 선물을 같은 계정**에서 거래 가능.
현물 BTC가 선물 마진의 담보로 자동 인식되어 자본 효율이 높음.

```
계정 구조:
  Unified Trading Account
    ├── 현물 BTC 보유 → 담보 자동 인정 (haircut 적용)
    ├── 무기한 선물 BTC 숏 → 마진은 UTA 전체 잔고에서
    └── USDT 잔고 → 나머지 마진 + v1 전략 자본
```

### 5.2 거래소 연결 (ccxt)

```python
# 현물 거래용 exchange
spot_exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_SECRET"),
    "options": {"defaultType": "spot"},
    "enableRateLimit": True,
})

# 선물 거래용 exchange (기존)
perp_exchange = ccxt.bybit({
    "apiKey": os.getenv("BYBIT_API_KEY"),
    "secret": os.getenv("BYBIT_SECRET"),
    "options": {"defaultType": "linear"},
    "enableRateLimit": True,
})

# 동일 API 키로 두 인스턴스 생성 → 같은 UTA 계정
```

### 5.3 SpotExecutor (신규)

```python
# src/execution/spot_executor.py

class SpotExecutor:
    """현물 매수/매도 실행기.
    
    펀딩비 차익거래의 현물 레그(leg)를 담당.
    """
    
    def __init__(self, spot_exchange: ccxt.Exchange):
        self.exchange = spot_exchange
    
    def buy_spot(self, symbol: str, amount: float, 
                 order_type: str = "limit") -> dict:
        """현물 매수.
        
        Args:
            symbol: "BTC/USDT" (현물 심볼, :USDT 없음)
            amount: 매수 수량 (BTC 단위)
            order_type: "limit" or "market"
        """
        ...
    
    def sell_spot(self, symbol: str, amount: float,
                  order_type: str = "limit") -> dict:
        """현물 매도 (포지션 청산 시)."""
        ...
    
    def get_spot_balance(self, coin: str = "BTC") -> float:
        """현물 잔고 조회."""
        ...
```

### 5.4 ArbExecutor (신규 — 핵심)

```python
# src/execution/arb_executor.py

class ArbExecutor:
    """현물+선물 동시 실행기 (펀딩비 차익거래 전용).
    
    원자적(atomic) 실행: 현물 매수와 선물 숏을 최대한 동시에 실행하여
    슬리피지 및 delta 노출을 최소화.
    """
    
    def __init__(self, spot_executor: SpotExecutor, 
                 perp_executor: OrderExecutor):
        self.spot = spot_executor
        self.perp = perp_executor
    
    def open_arb_position(
        self,
        symbol_spot: str,        # "BTC/USDT"
        symbol_perp: str,        # "BTC/USDT:USDT"
        amount: float,           # BTC 수량
        order_type: str = "limit",
    ) -> dict:
        """차익거래 포지션 진입.
        
        1. 현물 BTC 매수 (limit)
        2. 선물 BTC 숏 (limit) — 거의 동시에
        3. 체결 확인 + 델타 검증
        
        Returns:
            {"spot_order": ..., "perp_order": ..., 
             "delta": ..., "basis_at_entry": ...}
        """
        # Step 1: 현물 매수
        spot_order = self.spot.buy_spot(symbol_spot, amount, order_type)
        
        # Step 2: 선물 숏 (즉시)
        perp_order = self.perp.execute(
            symbol=symbol_perp, side="sell", amount=amount,
            order_type=order_type, ...
        )
        
        # Step 3: 델타 검증
        delta = self._calculate_delta(spot_order, perp_order)
        if abs(delta) > amount * 0.05:  # 5% 이상 차이
            logger.warning(f"델타 불일치: {delta} — 수동 확인 필요")
        
        return {
            "spot_order": spot_order,
            "perp_order": perp_order,
            "delta": delta,
            "entry_time": datetime.utcnow(),
        }
    
    def close_arb_position(
        self,
        symbol_spot: str,
        symbol_perp: str,
        amount: float,
    ) -> dict:
        """차익거래 포지션 청산.
        
        1. 선물 숏 청산 (buy)
        2. 현물 BTC 매도
        """
        ...
    
    def get_current_delta(self, symbol: str) -> float:
        """현재 델타 (현물 보유량 - 선물 숏 수량).
        
        이상적: 0 (완전 헤지)
        """
        spot_balance = self.spot.get_spot_balance(symbol.split("/")[0])
        perp_position = self.perp.sync_positions().get(symbol + ":USDT", {})
        perp_size = abs(perp_position.get("contracts", 0))
        return spot_balance - perp_size
```

### 5.5 실행 타이밍

```
펀딩비 결제: UTC 00:00, 08:00, 16:00

실행 주기:
  결제 4시간 전: 펀딩비 예측 실행
  결제 1시간 전: 예측 기반 포지션 조정 (확대/축소/유지)
  결제 시점:     펀딩비 수취
  결제 직후:     다음 사이클 준비, 델타 재확인

main.py 폴링:
  기존 v1: 600초(10분) 간격 → 1h 봉 기반
  펀딩비:  300초(5분) 간격 → 펀딩비 결제 시간 정밀 추적
  → 두 전략의 폴링 주기가 다름 → 펀딩비 전략은 별도 루프 또는 통합 루프의 최소 주기 적용
```

---

## 6. 전략 로직 (FundingArbStrategy)

### 6.1 진입 조건

```python
class FundingArbStrategy:
    """펀딩비 차익거래 전략.
    
    진입 조건 (기본):
      1. 현재 펀딩비 > min_funding_rate (기본 0.005% = 0.00005)
      2. 예측 펀딩비 > 0 (양수 지속 예상)
      3. 베이시스 < max_basis (선물 프리미엄이 과도하지 않음)
    
    ML 강화 진입:
      1. 예측 펀딩비 > entry_threshold
      2. 예측 확신도 > min_confidence
      3. 예측 펀딩비 × 남은 결제 횟수 > 진입 비용
    """
    
    def should_enter(self, features: dict) -> tuple[bool, float]:
        """진입 여부 + 포지션 크기 결정.
        
        Returns:
            (enter, position_scale)
            enter: 진입 여부
            position_scale: 0.0 ~ 1.0 (기본 포지션 대비 비율)
        """
        predicted_fr = self.predictor.predict_next_funding(features)
        confidence = self.predictor.predict_confidence(features)
        
        if predicted_fr <= self.entry_threshold:
            return False, 0.0
        
        if confidence < self.min_confidence:
            return False, 0.0
        
        # 동적 포지셔닝: 예측 펀딩비에 비례
        scale = min(predicted_fr / self.max_funding_scale, 1.0)
        scale *= confidence  # 확신도 가중
        
        return True, scale
```

### 6.2 포지션 관리

```python
    def manage_position(self, features: dict, 
                        current_position: dict) -> str:
        """기존 포지션 관리.
        
        Returns:
            "hold": 유지
            "increase": 포지션 확대
            "decrease": 포지션 축소
            "close": 전체 청산
        """
        predicted_fr = self.predictor.predict_next_funding(features)
        
        # 음수 전환 예측 → 청산
        if predicted_fr < -self.close_threshold:
            return "close"
        
        # 펀딩비 크게 하락 예측 → 축소
        if predicted_fr < self.decrease_threshold:
            return "decrease"
        
        # 펀딩비 상승 예측 → 확대
        if predicted_fr > self.increase_threshold:
            return "increase"
        
        return "hold"
```

### 6.3 청산 조건

```
즉시 청산:
  1. 예측 펀딩비 < -0.01% (음수 전환 확신)
  2. 베이시스 역전 > max_basis_reversal
  3. 포트폴리오 MDD 한도 도달 (기존 방어 로직)
  4. 델타 불일치 > 5% (헤지 깨짐)

축소 청산:
  1. 예측 펀딩비 < 0.005% (수익성 낮음)
  2. 연속 2회 음수 펀딩비 실현
```

---

## 7. 리스크 관리

### 7.1 포지션 한도

```yaml
# config/funding_arb.yaml

funding_arb:
  # 자본 배분
  max_capital_pct: 0.40              # 전체 자본의 최대 40%
  leverage: 2                        # 선물 레버리지 (보수적)
  
  # 진입 조건
  entry_threshold: 0.00005           # 최소 예측 펀딩비 0.005%
  min_confidence: 0.5                # 최소 예측 확신도
  max_basis_pct: 0.005               # 진입 시 최대 베이시스 0.5%
  
  # 청산 조건
  close_threshold: 0.0001            # 음수 펀딩비 -0.01% 예측 시 청산
  decrease_threshold: 0.00003        # 펀딩비 0.003% 미만 시 축소
  max_basis_reversal: 0.01           # 베이시스 역전 1% 초과 시 청산
  
  # 자산
  symbols:
    - {spot: "BTC/USDT", perp: "BTC/USDT:USDT"}
    - {spot: "ETH/USDT", perp: "ETH/USDT:USDT"}
  
  # 델타 관리
  max_delta_pct: 0.03                # 최대 델타 허용 3%
  rebalance_delta_pct: 0.02          # 델타 2% 초과 시 리밸런스
  
  # 모니터링
  check_interval_sec: 300            # 5분마다 체크
  funding_schedule_utc: [0, 8, 16]   # 펀딩비 결제 시간
  pre_funding_hours: 4               # 결제 4시간 전 예측 실행
```

### 7.2 자본 배분 (v1 + 펀딩비)

```
전체 자본 $1,400 기준:

v1 모멘텀 전략: 40% ($560)
  - BTC 모멘텀: 포지션 $112 (20% of $560)
  - ETH 모멘텀: 포지션 $112
  - BTC 평균회귀: 포지션 $112
  - 나머지: 마진 여유

펀딩비 차익: 40% ($560)
  - BTC 차익: $280 (현물 $280 + 선물 숏 $280, 레버리지 1x)
  - ETH 차익: $280

현금 버퍼: 20% ($280)
  - 긴급 마진, v1 방어 로직용
```

### 7.3 기존 PortfolioRiskManager 통합

```
펀딩비 차익은 "전략"으로 등록되므로 기존 방어 로직이 자동 적용:
  - MDD 5단계 축소: 차익 포지션도 축소
  - Rolling PF: 차익 전략의 PF 모니터링
  - 일일 손실 한도: 차익 포함 전체 손실 관리
  
추가 방어:
  - 델타 모니터링: 현물-선물 불일치 알림
  - 베이시스 급변 알림: 베이시스 ±1% 초과 시 텔레그램 경고
  - 청산 위험 알림: 마진율 50% 미만 시 경고
```

---

## 8. 백테스트

### 8.1 시뮬레이션 구조

```python
# backtest_funding_arb.py (신규)

def simulate_funding_arb(
    funding_rates: pd.DataFrame,    # 8h 간격 실제 펀딩비
    predictions: pd.DataFrame,      # 모델 예측값
    basis_series: pd.Series,        # 베이시스 시계열
    config: dict,                   # 전략 설정
) -> dict:
    """펀딩비 차익거래 백테스트.
    
    시뮬레이션 로직:
    1. 각 펀딩비 결제 시점에서:
       - 예측 기반 진입/청산/유지 결정
       - 포지션 크기 조정
    2. 수익 계산:
       - 펀딩비 수취: position_size × funding_rate
       - 비용: 진입/청산 수수료 + 슬리피지
       - 베이시스 변동 손익 (일시적)
    3. 메트릭 산출
    
    Returns:
        {
            "total_return": float,
            "annualized_return": float,
            "sharpe_ratio": float,
            "max_drawdown": float,
            "total_funding_collected": float,
            "total_costs": float,
            "avg_position_size": float,
            "time_in_market": float,         # 포지션 보유 시간 비율
            "funding_positive_ratio": float, # 양수 펀딩비 비율
        }
    """
```

### 8.2 비교 벤치마크

```
벤치마크 1: 정적 전략 (ML 없이)
  펀딩비 > 0.01%면 진입, < 0.005%면 청산
  → 기본 수익률 확인

벤치마크 2: ML 강화 전략 (예측 기반)
  LightGBM 예측으로 동적 포지셔닝
  → 정적 대비 개선 확인

벤치마크 3: v1 모멘텀 합산
  v1만 vs v1 + 펀딩비 → 포트폴리오 효과 확인
```

---

## 9. 구현 Phase

### Phase 0: 데이터 수집 + 탐색적 분석
- 펀딩비 이력 데이터 수집 (BTC, ETH, 2024-01 ~ 현재)
- 펀딩비 자기상관 분석 (예측 가능성 확인)
- 베이시스 분포 분석
- 롱/숏 비율 데이터 수집 (Bybit API)
- 피처 후보 탐색

**검증:** 펀딩비 자기상관 > 0.5 (lag 1), 양수 비율 > 60%

### Phase 1: 정적 전략 백테스트
- ML 없이 단순 규칙 기반 전략 구현
- 진입: 펀딩비 > threshold, 청산: 펀딩비 < threshold
- 비용 모델 (수수료, 슬리피지, 베이시스 변동)
- 기본 수익률/샤프/MDD 확인

**검증:** 연환산 수익률 > 10%, 샤프 > 1.0

### Phase 2: ML 예측 모델
- FundingRatePredictor 학습 (LightGBM)
- Walk-Forward 검증
- 정적 전략 vs ML 전략 A/B 비교

**검증:** ML 전략 샤프 > 정적 전략 샤프 × 1.3

### Phase 3: 실행 인프라
- SpotExecutor, ArbExecutor 구현
- 현물+선물 동시 실행 테스트 (testnet)
- 델타 모니터링 + 리밸런스 로직
- FundingArbStrategy (BaseStrategy 상속)

**검증:** testnet에서 진입/청산/펀딩비 수취 정상 동작

### Phase 4: 포트폴리오 통합
- portfolio.yaml에 funding_arb 추가
- v1 + 펀딩비 자본 배분
- 통합 백테스트 (v1 only vs v1 + 펀딩비)
- mainnet 배포

**검증:** v1+펀딩비 포트폴리오 샤프 > v1 only 샤프, MDD 감소

---

## 10. 성공 기준

| 메트릭 | 정적 전략 | ML 강화 | 포트폴리오(v1+차익) |
|--------|---------|--------|-------------------|
| 연환산 수익률 | > 10% | > 20% | > 15% |
| 샤프 비율 | > 1.0 | > 1.8 | > 1.5 |
| MDD | < -3% | < -3% | < -5% |
| v1과 상관관계 | - | - | < 0.1 |

---

## 11. 핵심 주의사항

1. **현물 매수 수수료가 핵심 비용.** Bybit 현물 maker 0.10%, taker 0.10%. 진입+청산 왕복 0.20% + 선물 0.11% = 총 0.31%. 펀딩비 0.01% × 3회/일 = 0.03%/일. 진입 비용 회수에 10일 소요. 자주 진입/청산하면 비용이 수익을 초과.

2. **레버리지는 2배 이하로 제한.** 급등 시 선물 숏 청산 리스크. 현물 BTC가 담보로 인정되지만 haircut(할인율) 적용. Bybit UTA에서 BTC 담보 haircut은 약 5~10%.

3. **v1 전략과 같은 USDT 잔고를 공유.** 펀딩비 전략이 USDT를 현물 BTC로 전환하면 v1 전략의 가용 마진이 줄어듦. 자본 배분을 명확히 분리해야 함.

4. **testnet에서 펀딩비 결제가 mainnet과 다를 수 있음.** testnet의 유동성이 낮아 펀딩비율이 비정상적. testnet에서는 인프라(주문 실행, 델타 관리)만 검증하고, 수익성은 백테스트 기준으로 판단.

5. **Bybit API에서 현물과 선물의 심볼이 다름.** 현물: `BTC/USDT`, 선물: `BTC/USDT:USDT`. 혼동하면 주문 실패. ArbExecutor에서 심볼 변환을 명확히 처리.

6. **펀딩비 결제 시간 직전에 포지션을 열면 슬리피지 증가.** 많은 차익거래자가 같은 시간에 포지션을 잡으므로, 결제 1~4시간 전에 미리 진입하는 것이 유리.
