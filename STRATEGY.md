# 전략 개발 가이드

---

## 전략 인터페이스 규칙

모든 전략은 `BaseStrategy`를 상속하고 `generate_signal` 메서드를 구현한다.
백테스팅/실거래 엔진이 동일한 인터페이스로 전략을 호출하기 위함이다.

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> int:
        """
        Returns:
            1  = 매수 신호
           -1  = 매도 신호
            0  = 중립 (포지션 없음)
        """
        pass

    def get_params(self) -> dict:
        """백테스트 결과 저장 시 파라미터 기록용"""
        return self.config
```

---

## 전략 개발 로드맵

### 1단계 — 이동평균 크로스오버 (MA Crossover)
파이프라인 전체 검증 목적. 전략 자체보다 시스템이 제대로 작동하는지 확인.

```python
class MACrossoverStrategy(BaseStrategy):
    """
    Fast MA가 Slow MA를 상향 돌파 → 매수
    Fast MA가 Slow MA를 하향 돌파 → 매도
    """
    def generate_signal(self, df: pd.DataFrame) -> int:
        fast = self.config["fast_period"]  # 기본값: 10
        slow = self.config["slow_period"]  # 기본값: 30

        ma_fast = df["close"].rolling(fast).mean()
        ma_slow = df["close"].rolling(slow).mean()

        if ma_fast.iloc[-1] > ma_slow.iloc[-1] and \
           ma_fast.iloc[-2] <= ma_slow.iloc[-2]:
            return 1   # 골든 크로스
        elif ma_fast.iloc[-1] < ma_slow.iloc[-1] and \
             ma_fast.iloc[-2] >= ma_slow.iloc[-2]:
            return -1  # 데드 크로스
        return 0
```

### 2단계 — RSI 평균회귀
오실레이터 기반. 과매수/과매도 구간에서 평균 회귀 베팅.

```
RSI < 30 → 과매도 → 매수
RSI > 70 → 과매수 → 매도
보유 기간: 수시간 ~ 수일
```

### 3단계 — 페어 트레이딩 (BTC/ETH)
통계적 차익거래 입문.

```python
# 기본 개념
spread = price_BTC - hedge_ratio * price_ETH
z_score = (spread - spread.mean()) / spread.std()

if z_score > 2.0:    # 스프레드 과대 → BTC 숏, ETH 롱
if z_score < -2.0:   # 스프레드 과소 → BTC 롱, ETH 숏
```

### 4단계 — 멀티팩터
여러 신호를 결합해 전략 강건성 향상.

```
팩터 조합 예시:
  모멘텀 팩터 (MA 크로스오버) × 0.4
+ 평균회귀 팩터 (RSI) × 0.3
+ 변동성 팩터 (ATR 기반) × 0.3
= 최종 신호 점수
```

---

## 백테스팅 방법론

### 기본 흐름
```python
import vectorbt as vbt

# 데이터 로드
df = pd.read_parquet("data/processed/BTCUSDT_1h_features.parquet")

# 신호 생성
strategy = MACrossoverStrategy(config={"fast_period": 10, "slow_period": 30})
signals = [strategy.generate_signal(df.iloc[:i+1]) for i in range(len(df))]

# 백테스트 실행
portfolio = vbt.Portfolio.from_signals(
    close=df["close"],
    entries=(pd.Series(signals) == 1),
    exits=(pd.Series(signals) == -1),
    fees=0.0004,        # Bybit 테이커 수수료 0.04%
    slippage=0.001,     # 슬리피지 0.1%
    init_cash=1_000_000
)

print(portfolio.stats())
```

### 오버피팅 방지 — Walk-Forward Validation
```
전체 데이터: 2022-01 ~ 2024-12 (3년)

구간 분할:
  In-Sample  (학습): 2022-01 ~ 2023-06  (18개월)
  Out-of-Sample (검증): 2023-07 ~ 2024-12  (18개월)

규칙:
  - 파라미터 최적화는 반드시 In-Sample에서만
  - Out-of-Sample 결과로 최종 판단
  - Out-of-Sample에서 성과 없으면 전략 폐기
```

### 백테스트 결과 저장 형식
```json
{
  "strategy": "MACrossoverStrategy",
  "params": {"fast_period": 10, "slow_period": 30},
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "period": {"start": "2022-01-01", "end": "2024-12-31"},
  "metrics": {
    "total_return": 0.342,
    "sharpe_ratio": 1.87,
    "max_drawdown": -0.183,
    "win_rate": 0.54,
    "profit_factor": 1.62,
    "total_trades": 148
  },
  "run_at": "2024-01-15T10:30:00"
}
```

저장 경로: `strategies/{전략명}/backtest_results/{날짜}_run{N}.json`

---

## 성과 평가 기준

| 지표 | 계산 방법 | 기준 |
|------|-----------|------|
| **샤프 비율** | (수익률 - 무위험수익률) / 변동성 | 1.5 이상 우수 |
| **MDD** | 고점 대비 최대 손실 | -20% 이내 목표 |
| **승률** | 수익 거래 / 전체 거래 | 단독으론 의미 없음 |
| **Profit Factor** | 총 수익 / 총 손실 | 1.5 이상 |
| **칼마 비율** | 연간 수익률 / MDD | 1.0 이상 |

---

## 전략 파라미터 관리

각 전략의 설정은 `config.yaml`로 분리해 관리한다.

```yaml
# strategies/ma_crossover/config.yaml
strategy:
  name: MACrossoverStrategy
  symbol: BTCUSDT
  timeframe: 1h

params:
  fast_period: 10
  slow_period: 30

execution:
  order_type: limit       # limit | market
  limit_offset: 0.001     # 지정가 호가 오프셋 (0.1%)

risk:
  max_position_pct: 0.05  # 자산의 5%
  stop_loss_pct: 0.02     # 2% 손절
  take_profit_pct: 0.04   # 4% 익절
```
