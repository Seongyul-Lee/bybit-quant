"""PortfolioManager 테스트."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.portfolio.manager import PortfolioManager
from src.portfolio.virtual_position import VirtualPositionTracker
from src.strategies.base import BaseStrategy


class MockStrategy(BaseStrategy):
    """테스트용 Mock 전략."""

    def __init__(
        self,
        signal: int = 0,
        probability: float = 0.0,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
    ) -> None:
        config = {"strategy_name": "mock", "symbol": symbol, "timeframe": timeframe}
        super().__init__(config)
        self._signal = signal
        self._probability = probability

    def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        return self._signal, self._probability

    def generate_signals_vectorized(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        signals = pd.Series(self._signal, index=df.index)
        probs = pd.Series(self._probability, index=df.index)
        return signals, probs


class TestPortfolioManager:
    """PortfolioManager 테스트."""

    def setup_method(self) -> None:
        """각 테스트 전 초기화."""
        self.config = {
            "allocation": {
                "mode": "equal",
                "position_pct_per_strategy": 0.20,
            },
            "limits": {
                "max_total_exposure": 0.60,
                "max_symbol_exposure": 0.30,
                "max_concurrent_positions": 5,
            },
        }
        self.manager = PortfolioManager(self.config)

    def test_register_unregister(self) -> None:
        """전략 등록/해제."""
        strat = MockStrategy()
        self.manager.register_strategy("test", strat)

        assert "test" in self.manager.get_active_strategies()
        assert self.manager.get_strategy("test") is strat

        self.manager.unregister_strategy("test")
        assert "test" not in self.manager.get_active_strategies()
        assert self.manager.get_strategy("test") is None

    def test_collect_signals(self) -> None:
        """전략별 시그널 수집."""
        strat_a = MockStrategy(signal=1, probability=0.7)
        strat_b = MockStrategy(signal=0, probability=0.3)
        self.manager.register_strategy("a", strat_a)
        self.manager.register_strategy("b", strat_b)

        df = pd.DataFrame({"close": [100.0]})
        data = {"a": df, "b": df}

        signals = self.manager.collect_signals(data)
        assert signals["a"] == (1, 0.7)
        assert signals["b"] == (0, 0.3)

    def test_collect_signals_missing_data(self) -> None:
        """데이터 없는 전략은 시그널 생성 안 함."""
        strat = MockStrategy(signal=1, probability=0.7)
        self.manager.register_strategy("a", strat)

        signals = self.manager.collect_signals({})
        assert "a" not in signals

    def test_allocate_single_strategy(self) -> None:
        """단일 전략 매수 시그널 → 주문 생성."""
        strat = MockStrategy(signal=1, probability=0.65, symbol="BTCUSDT")
        strategy_config = {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}}
        self.manager.register_strategy("btc", strat, strategy_config)

        tracker = VirtualPositionTracker()
        signals = {"btc": (1, 0.65)}
        orders = self.manager.allocate(signals, 100000.0, tracker)

        assert len(orders) == 1
        assert orders[0]["strategy"] == "btc"
        assert orders[0]["symbol"] == "BTC/USDT:USDT"
        assert orders[0]["side"] == "buy"
        assert orders[0]["size_pct"] == 0.20

    def test_allocate_no_buy_signal(self) -> None:
        """매수 시그널 없음 → 빈 주문."""
        strat = MockStrategy(signal=0, probability=0.3)
        strategy_config = {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}}
        self.manager.register_strategy("btc", strat, strategy_config)

        tracker = VirtualPositionTracker()
        signals = {"btc": (0, 0.3)}
        orders = self.manager.allocate(signals, 100000.0, tracker)

        assert len(orders) == 0

    def test_allocate_skip_existing_position(self) -> None:
        """이미 포지션 보유 → 스킵."""
        strat = MockStrategy(signal=1, probability=0.65, symbol="BTCUSDT")
        strategy_config = {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}}
        self.manager.register_strategy("btc", strat, strategy_config)

        tracker = VirtualPositionTracker()
        tracker.open("btc", "BTC/USDT:USDT", "long", 0.002, 80000.0)

        signals = {"btc": (1, 0.65)}
        orders = self.manager.allocate(signals, 100000.0, tracker)

        assert len(orders) == 0

    def test_allocate_symbol_cap(self) -> None:
        """동일 심볼 합산 캡 → 비례 축소."""
        # max_symbol_exposure = 0.30, 두 전략이 각각 0.20 → 합산 0.40 → 축소
        strat_a = MockStrategy(signal=1, probability=0.65, symbol="BTCUSDT")
        strat_b = MockStrategy(signal=1, probability=0.60, symbol="BTCUSDT")
        cfg = {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}}
        self.manager.register_strategy("a", strat_a, cfg)
        self.manager.register_strategy("b", strat_b, cfg)

        tracker = VirtualPositionTracker()
        signals = {"a": (1, 0.65), "b": (1, 0.60)}
        orders = self.manager.allocate(signals, 100000.0, tracker)

        # 합산이 0.30 이하로 축소되어야 함
        total_pct = sum(o["size_pct"] for o in orders)
        assert total_pct <= 0.30 + 1e-10

    def test_allocate_total_cap(self) -> None:
        """전체 노출 캡 → 비례 축소."""
        # max_total_exposure = 0.60
        # 4개 전략 × 0.20 = 0.80 → 축소
        for i in range(4):
            sym = f"COIN{i}USDT"
            strat = MockStrategy(signal=1, probability=0.65, symbol=sym)
            cfg = {"strategy": {"symbol": sym, "timeframe": "1h"}}
            self.manager.register_strategy(f"s{i}", strat, cfg)

        tracker = VirtualPositionTracker()
        signals = {f"s{i}": (1, 0.65) for i in range(4)}
        orders = self.manager.allocate(signals, 100000.0, tracker)

        total_pct = sum(o["size_pct"] for o in orders)
        assert total_pct <= 0.60 + 1e-10

    def test_allocate_min_order_amount(self) -> None:
        """최소 주문 금액 미달 → 주문 제거."""
        strat = MockStrategy(signal=1, probability=0.65, symbol="BTCUSDT")
        cfg = {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}}
        self.manager.register_strategy("btc", strat, cfg)

        tracker = VirtualPositionTracker()
        signals = {"btc": (1, 0.65)}
        # 포트폴리오 가치 100 USDT → 20% = 20 USDT < 100 USDT 최소
        orders = self.manager.allocate(signals, 100.0, tracker)

        assert len(orders) == 0

    def test_allocate_concurrent_position_limit(self) -> None:
        """동시 포지션 한도 → 초과 주문 스킵."""
        self.config["limits"]["max_concurrent_positions"] = 1
        manager = PortfolioManager(self.config)

        strat_a = MockStrategy(signal=1, probability=0.65, symbol="BTCUSDT")
        strat_b = MockStrategy(signal=1, probability=0.60, symbol="ETHUSDT")
        manager.register_strategy("a", strat_a, {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}})
        manager.register_strategy("b", strat_b, {"strategy": {"symbol": "ETHUSDT", "timeframe": "1h"}})

        # 이미 1개 포지션 보유
        tracker = VirtualPositionTracker()
        tracker.open("existing", "SOL/USDT:USDT", "long", 1.0, 100.0)

        signals = {"a": (1, 0.65), "b": (1, 0.60)}
        orders = manager.allocate(signals, 100000.0, tracker)

        assert len(orders) == 0

    def test_allocate_short_signal(self) -> None:
        """숏 시그널 → sell 주문 생성."""
        strat = MockStrategy(signal=-1, probability=0.70, symbol="BTCUSDT")
        cfg = {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}}
        self.manager.register_strategy("btc_short", strat, cfg)

        tracker = VirtualPositionTracker()
        signals = {"btc_short": (-1, 0.70)}
        orders = self.manager.allocate(signals, 100000.0, tracker)

        assert len(orders) == 1
        assert orders[0]["side"] == "sell"
        assert orders[0]["direction"] == "short"
        assert orders[0]["signal"] == -1
        assert orders[0]["size_pct"] == 0.20

    def test_allocate_mixed_long_short(self) -> None:
        """롱 + 숏 혼합 시그널 → 양방향 주문 생성."""
        strat_long = MockStrategy(signal=1, probability=0.65, symbol="BTCUSDT")
        strat_short = MockStrategy(signal=-1, probability=0.70, symbol="ETHUSDT")
        self.manager.register_strategy(
            "btc_long", strat_long,
            {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}},
        )
        self.manager.register_strategy(
            "eth_short", strat_short,
            {"strategy": {"symbol": "ETHUSDT", "timeframe": "1h"}},
        )

        tracker = VirtualPositionTracker()
        signals = {"btc_long": (1, 0.65), "eth_short": (-1, 0.70)}
        orders = self.manager.allocate(signals, 100000.0, tracker)

        assert len(orders) == 2
        sides = {o["strategy"]: o["side"] for o in orders}
        directions = {o["strategy"]: o["direction"] for o in orders}
        assert sides["btc_long"] == "buy"
        assert sides["eth_short"] == "sell"
        assert directions["btc_long"] == "long"
        assert directions["eth_short"] == "short"

    def test_allocate_v1_backward_compat(self) -> None:
        """v1 전략(signal=1만) → 기존과 동일 동작 (direction=long)."""
        strat = MockStrategy(signal=1, probability=0.65, symbol="BTCUSDT")
        cfg = {"strategy": {"symbol": "BTCUSDT", "timeframe": "1h"}}
        self.manager.register_strategy("btc_v1", strat, cfg)

        tracker = VirtualPositionTracker()
        signals = {"btc_v1": (1, 0.65)}
        orders = self.manager.allocate(signals, 100000.0, tracker)

        assert len(orders) == 1
        assert orders[0]["side"] == "buy"
        assert orders[0]["direction"] == "long"

    def test_convert_symbol(self) -> None:
        """심볼 변환."""
        assert PortfolioManager._convert_symbol("BTCUSDT") == "BTC/USDT:USDT"
        assert PortfolioManager._convert_symbol("ETHUSDT") == "ETH/USDT:USDT"

    def test_convert_symbol_invalid(self) -> None:
        """잘못된 심볼 → ValueError."""
        with pytest.raises(ValueError):
            PortfolioManager._convert_symbol("INVALIDXYZ")
