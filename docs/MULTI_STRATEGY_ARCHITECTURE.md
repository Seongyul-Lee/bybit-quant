# 멀티 전략 아키텍처 설계 문서

> 버전: 1.0
> 작성일: 2026-03-11
> 대상: bybit-quant 프로젝트
> 목적: 단일 전략(BTC/1h 롱 전용)에서 멀티 전략 포트폴리오로 확장하기 위한 아키텍처 설계

---

## 1. 현재 상태 (AS-IS)

### 1.1 구조

```
단일 전략, 단일 자산, 단일 타임프레임:
  BTC/USDT 1h → LGBMClassifierStrategy → generate_signal() → 1 or 0
  → RiskManager.check_all() → OrderExecutor.execute()
```

### 1.2 검증된 인프라

- 2클래스 binary LightGBM (롱 전용)
- Walk-Forward 학습 + 시드 고정 + 전 fold 저장
- 3-fold 앙상블 (fold 10, 11, 13)
- 펀딩비 적응형 threshold 필터
- OOS 검증 프레임워크 (보수적 비용 포함)
- 배리어 3.0×ATR, SL/TP 2.1% 대칭
- 보수적 비용 Post-Val PF 2.09, 하락률 4.1%

### 1.3 한계

- 연환산 수익률 ~3% (포지션 20%) — 단일 전략의 엣지 한계
- 단일 자산(BTC) 의존 — 분산 효과 없음
- 단일 접근법(모멘텀/추세) — 횡보장에서 비효율

---

## 2. 목표 상태 (TO-BE)

### 2.1 구조

```
멀티 전략 포트폴리오:
  전략 A (BTC/1h 모멘텀)  → (signal, probability)
  전략 B (ETH/1h 모멘텀)  → (signal, probability)
  전략 C (BTC/4h 모멘텀)  → (signal, probability)
  전략 D (BTC/1h 평균회귀) → (signal, probability)
      ↓
  PortfolioManager → 자본 배분 → 합산 주문
      ↓
  PortfolioRiskManager → 전체 리스크 제어
      ↓
  OrderExecutor → 가상→실제 포지션 변환 → 거래소 주문
```

### 2.2 설계 원칙

1. **전략 독립성**: 각 전략은 독립적으로 학습/백테스트/OOS 검증 가능
2. **공유 인프라**: FeatureEngine, Trainer, Labeler 등은 공통 모듈로 재사용
3. **2계층 리스크**: 전략별 리스크 + 포트폴리오 리스크
4. **플러그인 방식**: 새 전략 추가 시 폴더 + config + strategy.py만 생성
5. **단일 프로세스**: 동기 순차 실행, 멀티스레드 없음
6. **최대 5전략**: 초기 2~3전략, 자본 확장 시 최대 5전략

---

## 3. 프로젝트 구조

### 3.1 디렉토리 구조

```
bybit-quant/
├── config/
│   ├── .env                          # API 키 (기존)
│   ├── exchanges.yaml                # 거래소 설정 (기존)
│   ├── risk_params.yaml              # 전략 레벨 리스크 (기존)
│   └── portfolio.yaml                # [신규] 포트폴리오 설정
│
├── data/
│   ├── raw/bybit/
│   │   ├── BTCUSDTUSDT/
│   │   │   ├── 1h/                   # BTC 1h OHLCV (기존)
│   │   │   ├── 4h/                   # [신규] BTC 4h OHLCV
│   │   │   └── funding_rate.parquet  # BTC 펀딩비 (기존)
│   │   ├── ETHUSDTUSDT/              # [신규]
│   │   │   ├── 1h/
│   │   │   └── funding_rate.parquet
│   │   └── SOLUSDTUSDT/              # [신규, Phase 1 이후]
│   │       └── ...
│   └── processed/
│       ├── BTCUSDT_1h_features.parquet   # (기존)
│       ├── BTCUSDT_4h_features.parquet   # [신규]
│       └── ETHUSDT_1h_features.parquet   # [신규]
│
├── src/
│   ├── data/
│   │   ├── collector.py              # 멀티 심볼/타임프레임 수집 (확장)
│   │   └── processor.py              # 심볼별 처리 (기존 호환)
│   ├── portfolio/                    # [신규] 포트폴리오 레이어
│   │   ├── __init__.py
│   │   ├── manager.py                # PortfolioManager
│   │   ├── risk.py                   # PortfolioRiskManager
│   │   └── virtual_position.py       # VirtualPositionTracker
│   ├── execution/
│   │   └── executor.py               # 가상→실제 포지션 변환 추가 (확장)
│   ├── risk/
│   │   └── manager.py                # 전략 레벨 리스크 (기존 유지)
│   ├── strategies/
│   │   └── base.py                   # BaseStrategy 인터페이스 확장
│   ├── analytics/
│   │   └── reporter.py               # (기존 유지)
│   └── utils/
│       ├── logger.py                 # (기존 유지)
│       └── notify.py                 # 전략별 + 포트폴리오 알림 (확장)
│
├── strategies/
│   ├── _common/                      # [신규] 공통 ML 모듈
│   │   ├── __init__.py
│   │   ├── features.py               # FeatureEngine (기존에서 이동)
│   │   ├── labeler.py                # TripleBarrierLabeler (기존에서 이동)
│   │   ├── trainer.py                # WalkForwardTrainer (기존에서 이동)
│   │   └── evaluator.py              # ModelEvaluator (기존에서 이동)
│   ├── btc_1h_momentum/              # 기존 lgbm_classifier 리네임
│   │   ├── __init__.py
│   │   ├── config.yaml               # 전략별 설정
│   │   ├── strategy.py               # LGBMClassifierStrategy
│   │   └── models/                   # fold 모델 파일
│   ├── eth_1h_momentum/              # [신규] Phase 1
│   │   ├── __init__.py
│   │   ├── config.yaml
│   │   ├── strategy.py
│   │   └── models/
│   ├── btc_4h_momentum/              # [신규] Phase 2
│   │   └── ...
│   └── btc_1h_mean_reversion/        # [신규] Phase 3
│       └── ...
│
├── main.py                           # 실거래 진입점 (리팩토링)
├── backtest.py                       # 전략별 백테스트 (확장)
├── portfolio_backtest.py             # [신규] 포트폴리오 합산 백테스트
├── train_lgbm.py                     # 전략별 학습 (확장)
├── oos_validation.py                 # 전략별 OOS 검증 (확장)
└── tests/
    ├── test_portfolio_manager.py     # [신규]
    ├── test_virtual_position.py      # [신규]
    └── ...                           # 기존 테스트 유지
```

### 3.2 기존 파일 마이그레이션

| 현재 위치 | 이동 후 위치 | 비고 |
|-----------|-------------|------|
| `strategies/lgbm_classifier/features.py` | `strategies/_common/features.py` | 공통 모듈화 |
| `strategies/lgbm_classifier/labeler.py` | `strategies/_common/labeler.py` | 공통 모듈화 |
| `strategies/lgbm_classifier/trainer.py` | `strategies/_common/trainer.py` | 공통 모듈화 |
| `strategies/lgbm_classifier/evaluator.py` | `strategies/_common/evaluator.py` | 공통 모듈화 |
| `strategies/lgbm_classifier/` | `strategies/btc_1h_momentum/` | 리네임 |
| `strategies/lgbm_classifier/strategy.py` | `strategies/btc_1h_momentum/strategy.py` | import 경로 변경 |
| `strategies/ma_crossover/` | 삭제 | 테스트 전용이었음 |

---

## 4. 핵심 컴포넌트 설계

### 4.1 BaseStrategy 인터페이스 확장

```python
# src/strategies/base.py

class BaseStrategy(ABC):
    """전략 기본 추상 클래스.
    
    모든 전략은 이 클래스를 상속하고 generate_signal을 구현해야 한다.
    """
    
    def __init__(self, config: dict) -> None:
        self.config = config
        self.strategy_name: str = config.get("strategy_name", "unknown")
        self.symbol: str = config.get("symbol", "BTCUSDT")
        self.timeframe: str = config.get("timeframe", "1h")
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """매매 신호 + 확신도를 반환.
        
        Returns:
            (signal, probability) 튜플.
            signal: 1(매수) 또는 0(비매수).
            probability: 매수 확률 (0.0 ~ 1.0). 자본 배분에 활용.
        """
        pass
    
    @abstractmethod
    def generate_signals_vectorized(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """벡터화 신호 + 확신도 시리즈 반환 (백테스트 전용).
        
        Returns:
            (signal_series, probability_series) 튜플.
        """
        pass
    
    def get_params(self) -> dict:
        return self.config
```

### 4.2 PortfolioManager

```python
# src/portfolio/manager.py

class PortfolioManager:
    """멀티 전략 포트폴리오 관리.
    
    전략별 시그널을 수집하고, 자본 배분 규칙에 따라
    최종 주문 목록을 생성한다.
    
    자본 배분 방식: 균등 고정 비중 (전략당 1/N).
    동일 심볼 합산 포지션 상한 적용.
    """
    
    def __init__(self, config: dict) -> None:
        """
        Args:
            config: portfolio.yaml 설정.
                - strategies: 활성 전략 이름 리스트
                - max_total_exposure: 전체 포지션 합산 상한 (예: 0.6 = 60%)
                - max_symbol_exposure: 동일 심볼 합산 상한 (예: 0.3 = 30%)
                - position_pct_per_strategy: 전략당 포지션 비중 (예: 0.2 = 20%)
        """
        self.strategies: dict[str, BaseStrategy] = {}
        self.max_total_exposure = config.get("max_total_exposure", 0.6)
        self.max_symbol_exposure = config.get("max_symbol_exposure", 0.3)
        self.position_pct = config.get("position_pct_per_strategy", 0.2)
    
    def register_strategy(self, name: str, strategy: BaseStrategy) -> None:
        """전략 등록."""
        self.strategies[name] = strategy
    
    def collect_signals(self, data: dict[str, pd.DataFrame]) -> dict[str, tuple[int, float]]:
        """모든 전략의 시그널을 수집.
        
        Args:
            data: {전략이름: 해당 전략의 데이터프레임} 딕셔너리.
        
        Returns:
            {전략이름: (signal, probability)} 딕셔너리.
        """
        signals = {}
        for name, strategy in self.strategies.items():
            if name in data:
                signal, prob = strategy.generate_signal(data[name])
                signals[name] = (signal, prob)
        return signals
    
    def allocate(
        self,
        signals: dict[str, tuple[int, float]],
        portfolio_value: float,
        current_positions: dict,
    ) -> list[dict]:
        """시그널을 기반으로 주문 목록 생성.
        
        자본 배분 규칙:
        1. 각 전략에 균등 비중 (position_pct_per_strategy)
        2. 동일 심볼 합산이 max_symbol_exposure 초과 시 비례 축소
        3. 전체 합산이 max_total_exposure 초과 시 비례 축소
        
        Returns:
            [{"strategy": name, "symbol": sym, "side": "long",
              "size_pct": 0.20, "signal": 1, "probability": 0.65}, ...]
        """
        pass  # 구현 시 상세 로직
```

### 4.3 PortfolioRiskManager

```python
# src/portfolio/risk.py

class PortfolioRiskManager:
    """포트폴리오 레벨 리스크 관리.
    
    전략 레벨 RiskManager 위에 위치하여
    포트폴리오 전체의 리스크를 제어한다.
    """
    
    def __init__(self, config: dict) -> None:
        """
        Args:
            config: portfolio.yaml 리스크 설정.
                - max_portfolio_mdd: 전체 MDD 한도 (예: -0.10 = -10%)
                - max_daily_loss: 일일 손실 한도 (예: -0.03 = -3%)
                - strategy_disable_threshold: 전략 비활성화 PF 기준 (예: 0.8)
                - strategy_disable_min_trades: 비활성화 판단 최소 거래 수 (예: 50)
        """
        self.max_portfolio_mdd = config.get("max_portfolio_mdd", -0.10)
        self.max_daily_loss = config.get("max_daily_loss", -0.03)
        self.strategy_disable_threshold = config.get("strategy_disable_threshold", 0.8)
        self.strategy_disable_min_trades = config.get("strategy_disable_min_trades", 50)
        self.strategy_stats: dict[str, dict] = {}  # 전략별 실전 성과 추적
    
    def check_portfolio(self, portfolio_value: float, peak_value: float) -> tuple[bool, str]:
        """포트폴리오 전체 리스크 체크.
        
        Returns:
            (통과 여부, 사유).
        """
        current_dd = (portfolio_value - peak_value) / peak_value
        if current_dd < self.max_portfolio_mdd:
            return False, f"포트폴리오 MDD 한도 초과: {current_dd:.2%}"
        return True, "OK"
    
    def check_strategy_health(self, strategy_name: str) -> bool:
        """개별 전략의 실전 성과를 기반으로 활성 상태 판단.
        
        50거래 이상 축적 후 PF < 0.8이면 비활성화.
        """
        stats = self.strategy_stats.get(strategy_name)
        if not stats or stats["total_trades"] < self.strategy_disable_min_trades:
            return True  # 데이터 부족 → 활성 유지
        return stats["profit_factor"] >= self.strategy_disable_threshold
    
    def record_trade(self, strategy_name: str, pnl: float) -> None:
        """전략별 거래 결과 기록."""
        pass  # 구현 시 PF, 승률 등 실시간 업데이트
```

### 4.4 VirtualPositionTracker

```python
# src/portfolio/virtual_position.py

class VirtualPositionTracker:
    """전략별 가상 포지션 추적.
    
    Bybit는 심볼당 하나의 포지션만 허용 (원웨이 모드).
    여러 전략이 동일 심볼에 대해 포지션을 가질 때,
    내부적으로 전략별 가상 포지션을 관리하고
    실제 포지션은 합산으로 유지한다.
    
    예시:
        전략A: BTC 0.002 롱 (가상)
        전략B: BTC 0.001 롱 (가상)
        → 실제 Bybit 포지션: BTC 0.003 롱
        
        전략A가 청산 → 실제 포지션을 0.001로 줄이는 주문 실행
    """
    
    def __init__(self) -> None:
        # {strategy_name: {symbol: {"side": "long", "size": 0.002, "entry_price": 80000}}}
        self.virtual_positions: dict[str, dict[str, dict]] = {}
    
    def update(self, strategy_name: str, symbol: str, side: str, size: float, entry_price: float) -> None:
        """가상 포지션 업데이트."""
        pass
    
    def close(self, strategy_name: str, symbol: str) -> None:
        """전략의 가상 포지션 청산."""
        pass
    
    def get_real_position(self, symbol: str) -> dict:
        """심볼의 실제 포지션 (전략별 합산) 반환.
        
        Returns:
            {"side": "long", "size": 0.003} 또는 빈 dict.
        """
        total_size = 0.0
        for strategy_positions in self.virtual_positions.values():
            if symbol in strategy_positions:
                pos = strategy_positions[symbol]
                if pos["side"] == "long":
                    total_size += pos["size"]
                else:
                    total_size -= pos["size"]
        
        if total_size > 0:
            return {"side": "long", "size": total_size}
        elif total_size < 0:
            return {"side": "short", "size": abs(total_size)}
        return {}
    
    def get_delta_orders(self, symbol: str, current_real: dict) -> list[dict]:
        """가상 합산과 실제 포지션의 차이를 주문으로 변환.
        
        Returns:
            [{"symbol": "BTCUSDT", "side": "buy", "amount": 0.001}, ...]
        """
        pass
    
    def to_dict(self) -> dict:
        """직렬화 (상태 저장용)."""
        return self.virtual_positions
    
    def from_dict(self, data: dict) -> None:
        """역직렬화 (상태 복원용)."""
        self.virtual_positions = data
```

### 4.5 portfolio.yaml 설정 구조

```yaml
# config/portfolio.yaml

portfolio:
  # 활성 전략 목록
  active_strategies:
    - btc_1h_momentum
    # - eth_1h_momentum      # Phase 1에서 활성화
    # - btc_4h_momentum      # Phase 2에서 활성화
  
  # 자본 배분
  allocation:
    mode: equal                     # equal(균등) | fixed(수동 지정)
    position_pct_per_strategy: 0.20 # 전략당 포지션 비중
    # fixed_weights:                # mode=fixed일 때만 사용
    #   btc_1h_momentum: 0.25
    #   eth_1h_momentum: 0.15
  
  # 포지션 한도
  limits:
    max_total_exposure: 0.60        # 전체 포지션 합산 상한 (60%)
    max_symbol_exposure: 0.30       # 동일 심볼 합산 상한 (30%)
    max_concurrent_positions: 5     # 동시 최대 포지션 수

  # 포트폴리오 리스크
  risk:
    max_portfolio_mdd: -0.10        # 전체 MDD 한도 → 전략 신규 진입 차단
    max_daily_loss: -0.03           # 일일 손실 한도
    strategy_disable_threshold: 0.8 # 전략 PF 이 기준 미달 시 비활성화
    strategy_disable_min_trades: 50 # 비활성화 판단 최소 거래 수
```

---

## 5. 전략 설정 구조

### 5.1 전략별 config.yaml

각 전략 폴더에 독립적인 설정 파일:

```yaml
# strategies/btc_1h_momentum/config.yaml

strategy:
  name: btc_1h_momentum
  type: lgbm_binary               # lgbm_binary | rule_based | statistical
  symbol: BTCUSDT
  timeframe: 1h

params:
  model_path: strategies/btc_1h_momentum/models/latest.txt
  feature_names_path: strategies/btc_1h_momentum/models/feature_names.json
  confidence_threshold: 0.48
  upper_barrier_multiplier: 3.0
  lower_barrier_multiplier: 3.0
  max_holding_period: 24
  ensemble_folds: [11, 13]
  models_dir: strategies/btc_1h_momentum/models

  # 펀딩비 필터
  funding_filter:
    enabled: true
    zscore_aggressive: 0           # fr_zscore < 0 → 공격적 threshold
    zscore_block: 2                # fr_zscore ≥ 2 → 진입 차단
    threshold_aggressive: 0.42
    threshold_normal: 0.44

execution:
  order_type: limit
  limit_offset: 0.001
  fee_rate: 0.00055

risk:
  stop_loss_pct: 0.021
  take_profit_pct: 0.021
```

### 5.2 새 전략 추가 체크리스트

새 전략을 추가할 때 필요한 최소 파일:

```
strategies/{strategy_name}/
├── __init__.py
├── config.yaml          # 전략 설정
├── strategy.py          # BaseStrategy 상속, generate_signal 구현
└── models/              # 학습된 모델 파일 (ML 전략의 경우)
```

추가 후 활성화:
1. `config/portfolio.yaml`의 `active_strategies`에 전략 이름 추가
2. `main.py`가 자동으로 전략 로드 (전략 레지스트리 기반)

---

## 6. 데이터 파이프라인

### 6.1 멀티 심볼/타임프레임 수집

```python
# src/data/collector.py 확장

class BybitDataCollector:
    def fetch_multi_symbols(
        self,
        symbols: list[str],
        timeframes: list[str],
        since: str,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """여러 심볼 × 타임프레임의 OHLCV를 일괄 수집.
        
        Returns:
            {symbol: {timeframe: DataFrame}} 중첩 딕셔너리.
        """
        pass
```

API rate limit 관리:
- Bybit: 분당 120회 요청 제한
- 심볼당 수집 사이에 `sleep(0.3)` 이상
- 3심볼 × 2타임프레임 = 6 조합 → 초기 bulk 수집 시 순차 처리

### 6.2 타임프레임별 데이터 관리

**4h, 1d 데이터는 별도 수집** (resample하지 않음):

```
data/raw/bybit/BTCUSDTUSDT/1h/2024-01.parquet   # 1h 봉 직접 수집
data/raw/bybit/BTCUSDTUSDT/4h/2024-01.parquet   # 4h 봉 직접 수집
data/processed/BTCUSDT_1h_features.parquet       # 1h 피처
data/processed/BTCUSDT_4h_features.parquet       # 4h 피처
```

### 6.3 실시간 데이터 동기화

메인 루프에서 타임프레임별 봉 마감 체크:

```python
# main.py 메인 루프 내

for strategy in active_strategies:
    if is_bar_closed(strategy.timeframe, current_time):
        df = fetch_latest_data(strategy.symbol, strategy.timeframe)
        signal, prob = strategy.generate_signal(df)
        signals[strategy.name] = (signal, prob)
```

1h 전략은 매 정시, 4h 전략은 4시간마다 시그널 생성.

---

## 7. 주문 실행 흐름

### 7.1 전체 흐름

```
1. 봉 마감 감지
2. 각 전략의 generate_signal() 호출 → (signal, probability)
3. PortfolioManager.allocate() → 주문 목록 생성
4. PortfolioRiskManager.check_portfolio() → 전체 리스크 체크
5. 전략별 RiskManager.check_all() → 개별 리스크 체크
6. VirtualPositionTracker 업데이트
7. VirtualPositionTracker.get_delta_orders() → 실제 주문 계산
8. OrderExecutor.execute() → 거래소 주문 제출
9. 상태 저장 (virtual positions, PnL, circuit breaker 등)
```

### 7.2 동일 심볼 포지션 처리

```
예시: BTC/1h 전략과 BTC/4h 전략이 둘 다 활성

시나리오 1: 둘 다 매수 시그널
  가상: 전략A 0.002 BTC 롱 + 전략B 0.001 BTC 롱
  실제: 0.003 BTC 롱 주문

시나리오 2: 전략A 매수, 전략B 비매수
  가상: 전략A 0.002 BTC 롱 + 전략B 포지션 없음
  실제: 0.002 BTC 롱 주문

시나리오 3: 전략A 청산 (SL), 전략B 유지
  가상: 전략A 포지션 없음 + 전략B 0.001 BTC 롱
  실제: 0.003 → 0.001로 줄이는 부분 청산 주문

시나리오 4: 합산이 max_symbol_exposure 초과
  전략A 요청 0.003 + 전략B 요청 0.002 = 0.005 BTC
  max_symbol_exposure 30% = 0.003 BTC 상한
  → 비례 축소: 전략A 0.0018, 전략B 0.0012
```

---

## 8. 학습/백테스트/검증 파이프라인

### 8.1 전략별 독립 파이프라인

```bash
# 학습
python train_lgbm.py --strategy btc_1h_momentum --no-optuna --upper-barrier 3.0 --lower-barrier 3.0
python train_lgbm.py --strategy eth_1h_momentum --no-optuna --upper-barrier 3.0 --lower-barrier 3.0

# 백테스트
python backtest.py --strategy btc_1h_momentum
python backtest.py --strategy eth_1h_momentum

# OOS 검증
python oos_validation.py --strategy btc_1h_momentum
python oos_validation.py --strategy eth_1h_momentum
```

### 8.2 포트폴리오 합산 백테스트

```bash
# 여러 전략의 시그널을 조합한 합산 성과 측정
python portfolio_backtest.py --strategies btc_1h_momentum,eth_1h_momentum
```

포트폴리오 백테스트 로직:
1. 각 전략의 시그널 시리즈를 독립 생성
2. 자본 배분 규칙 적용 (균등 비중, 심볼 합산 캡)
3. 가상 포지션 기반 합산 수익률 계산
4. 포트폴리오 레벨 메트릭 산출 (합산 PF, MDD, 상관관계)

### 8.3 검증 기준

**전략 레벨 (필수):**
- 보수적 비용 Post-Val PF ≥ 1.20
- Post-Val 수익률 > 0%
- 거래 수 ≥ 25건
- IS 대비 PF 하락률 ≤ 50%

**포트폴리오 레벨 (참고):**
- 포트폴리오 합산 MDD < 개별 전략 MDD의 최대값 (분산 효과 확인)
- 전략 간 수익률 상관관계 < 0.7 (독립성 확인)

---

## 9. 모니터링 및 운영

### 9.1 텔레그램 알림 구조

```
[전략별 알림] — 시그널/체결/청산 발생 시 즉시
  📈 btc_1h_momentum: 매수 시그널 (확률 0.65)
  ✅ btc_1h_momentum: BTC 0.002 롱 체결 @ $80,000
  🔴 btc_1h_momentum: SL 청산 -2.1% ($-336)

[포트폴리오 일일 요약] — 매일 1회 (UTC 00:00)
  📊 일일 포트폴리오 요약
  총 PnL: +$120 (+0.12%)
  활성 전략: 2/3
  활성 포지션: BTC 롱 0.003
  일일 거래: 3건 (승 2 / 패 1)

[긴급 알림] — Circuit Breaker 또는 포트폴리오 MDD 한도
  🚨 포트폴리오 MDD -10% 도달 — 신규 진입 차단
  🚨 btc_1h_momentum Circuit Breaker 발동 — 해당 전략 중단
```

### 9.2 전략 자동 비활성화

```python
# 매 거래 후 PortfolioRiskManager에서 체크
if strategy.total_trades >= 50 and strategy.profit_factor < 0.8:
    portfolio_manager.disable_strategy(strategy_name)
    notifier.send(f"⚠️ {strategy_name} 비활성화 (PF {strategy.pf:.2f} < 0.8)")
```

### 9.3 로그 분리

```
logs/
├── portfolio.log          # 포트폴리오 매니저 로그
├── btc_1h_momentum.log    # 전략별 로그
├── eth_1h_momentum.log
└── executor.log           # 주문 실행 로그
```

---

## 10. 실행 환경 제약 및 대응

### 10.1 Bybit API 제약

| 항목 | 제한 | 대응 |
|------|------|------|
| 분당 요청 수 | 120회 | 전략별 순차 처리, sleep(0.3) |
| 동시 미체결 주문 | 심볼당 25개 | 전략당 1개 주문 원칙 |
| 최소 주문 금액 | ~100 USDT | 포지션 비중 × 투자금이 100 USDT 초과 확인 |
| 포지션 모드 | 원웨이/헤지 | 원웨이 모드 사용 (가상 포지션으로 관리) |

### 10.2 자본 규모별 전략 수 가이드

| 투자금 | 전략당 20% | 최소 주문 충족 | 권장 전략 수 |
|--------|-----------|---------------|-------------|
| 50만원 | 10만원 | 미달 | 1전략 (30%) |
| 100만원 | 20만원 | 충족 | 2전략 |
| 200만원 | 40만원 | 충족 | 3전략 |
| 500만원 | 100만원 | 충족 | 3~5전략 |

### 10.3 단일 프로세스 실행 모델

```python
# main.py 메인 루프 (간소화)

while True:
    current_time = get_current_time()
    
    for strategy_name, strategy in portfolio_manager.strategies.items():
        if not is_bar_closed(strategy.timeframe, current_time):
            continue
        
        # 데이터 로드
        df = load_latest_data(strategy.symbol, strategy.timeframe)
        
        # 시그널 생성
        signal, prob = strategy.generate_signal(df)
        signals[strategy_name] = (signal, prob)
    
    if signals:
        # 포트폴리오 리스크 체크
        ok, reason = portfolio_risk.check_portfolio(portfolio_value, peak_value)
        if not ok:
            logger.warning(f"포트폴리오 리스크: {reason}")
            signals = {}  # 모든 시그널 무시
        
        # 자본 배분 + 주문 실행
        orders = portfolio_manager.allocate(signals, portfolio_value, positions)
        for order in orders:
            executor.execute(order)
        
        signals.clear()
    
    sleep(poll_interval)
```

---

## 11. 확장 로드맵

### Phase 1: 동일 모델, 다른 자산 (ETH)

- ETH/USDT 1h 데이터 수집
- 기존 BTC 모델과 동일 파이프라인으로 ETH 모델 학습
- OOS 검증 통과 후 포트폴리오에 추가
- 기대 효과: BTC-ETH 상관관계가 ~0.7이므로 부분적 분산 효과

### Phase 2: 동일 자산, 다른 타임프레임 (4h)

- BTC/USDT 4h 데이터 수집
- 4h 배리어는 더 넓게 설정 가능 → 비용 비율 개선
- 1h과 4h 전략의 시그널 상관관계가 낮으면 분산 효과
- 동일 심볼 포지션 합산 관리 검증

### Phase 3: 다른 접근법 (평균회귀)

- 현재 모멘텀/추세 기반과 다른 접근
- 횡보장에서 수익을 내는 전략 → 기존 전략과 상관관계 최소화
- 가장 작업량이 크지만 포트폴리오 효과 최대화

---

## 12. 의사결정 로그

| 항목 | 결정 | 근거 |
|------|------|------|
| 시그널 인터페이스 | `(int, float)` 반환 | 자본 배분에 확신도 활용 |
| 데이터 수집 | 타임프레임별 직접 수집 | resample 경계값 오류 방지 |
| 자본 배분 | 균등 고정 비중 | 초기 단계에서 동적 배분의 과적합 방지 |
| 시그널 충돌 | 독립 처리 + 합산 캡 | 전략 간 독립성 유지 |
| 리스크 관리 | 2계층 (전략 + 포트폴리오) | 개별 제어 + 전체 안전장치 |
| 포지션 관리 | 가상 포지션 분리 | 원웨이 모드 제약 대응 |
| 전략 비활성화 | PF < 0.8 (50거래 후) | 데이터 기반 자동 판단 |
| 실행 모델 | 단일 프로세스 동기 | 복잡도 최소화, 디버깅 용이 |
| 전략 수 상한 | 5개 | API, 자본, 유지보수 제약 |
| 멀티스레드 | 미사용 | 1h 타임프레임에서 불필요 |
