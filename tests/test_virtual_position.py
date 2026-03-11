"""VirtualPositionTracker 테스트."""

import pytest

from src.portfolio.virtual_position import VirtualPositionTracker


class TestVirtualPositionTracker:
    """VirtualPositionTracker 테스트."""

    def setup_method(self) -> None:
        """각 테스트 전 초기화."""
        self.tracker = VirtualPositionTracker()

    def test_open_single_strategy(self) -> None:
        """단일 전략 open → 정상 추적."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)

        assert self.tracker.has_position("strat_a", "BTC/USDT:USDT")
        pos = self.tracker.get_position("strat_a", "BTC/USDT:USDT")
        assert pos["side"] == "long"
        assert pos["size"] == 0.002
        assert pos["entry_price"] == 80000.0

    def test_close_single_strategy(self) -> None:
        """단일 전략 close → 포지션 제거."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)
        self.tracker.close("strat_a", "BTC/USDT:USDT")

        assert not self.tracker.has_position("strat_a", "BTC/USDT:USDT")
        assert self.tracker.get_position("strat_a", "BTC/USDT:USDT") == {}

    def test_two_strategies_same_symbol(self) -> None:
        """두 전략이 동일 심볼에 open → 합산 포지션 정확."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)
        self.tracker.open("strat_b", "BTC/USDT:USDT", "long", 0.001, 81000.0)

        real = self.tracker.get_real_position("BTC/USDT:USDT")
        assert real["side"] == "long"
        assert abs(real["size"] - 0.003) < 1e-10

    def test_partial_close_delta(self) -> None:
        """한 전략만 close → 부분 청산 delta 주문 생성."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)
        self.tracker.open("strat_b", "BTC/USDT:USDT", "long", 0.001, 81000.0)

        # 전략A 청산
        self.tracker.close("strat_a", "BTC/USDT:USDT")

        # 현재 실제 포지션은 0.003 (아직 거래소 미반영)
        current_real = {"side": "long", "size": 0.003}
        deltas = self.tracker.get_delta_orders("BTC/USDT:USDT", current_real)

        assert len(deltas) == 1
        assert deltas[0]["side"] == "sell"
        assert abs(deltas[0]["amount"] - 0.002) < 1e-10

    def test_no_strategy_empty_position(self) -> None:
        """전략 없음 → 빈 포지션."""
        real = self.tracker.get_real_position("BTC/USDT:USDT")
        assert real == {}

    def test_delta_no_change(self) -> None:
        """가상 합산과 실제 일치 → 주문 없음."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)

        current_real = {"side": "long", "size": 0.002}
        deltas = self.tracker.get_delta_orders("BTC/USDT:USDT", current_real)
        assert deltas == []

    def test_delta_buy_more(self) -> None:
        """가상이 실제보다 큼 → 추가 매수 주문."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.003, 80000.0)

        current_real = {"side": "long", "size": 0.001}
        deltas = self.tracker.get_delta_orders("BTC/USDT:USDT", current_real)

        assert len(deltas) == 1
        assert deltas[0]["side"] == "buy"
        assert abs(deltas[0]["amount"] - 0.002) < 1e-10

    def test_delta_from_empty(self) -> None:
        """실제 포지션 없음 → 전체 매수."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)

        deltas = self.tracker.get_delta_orders("BTC/USDT:USDT", {})
        assert len(deltas) == 1
        assert deltas[0]["side"] == "buy"
        assert abs(deltas[0]["amount"] - 0.002) < 1e-10

    def test_to_dict_from_dict(self) -> None:
        """상태 직렬화/역직렬화 → 정확한 복원."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)
        self.tracker.open("strat_b", "ETH/USDT:USDT", "long", 0.05, 3000.0)

        state = self.tracker.to_dict()

        new_tracker = VirtualPositionTracker()
        new_tracker.from_dict(state)

        assert new_tracker.has_position("strat_a", "BTC/USDT:USDT")
        assert new_tracker.has_position("strat_b", "ETH/USDT:USDT")

        pos_a = new_tracker.get_position("strat_a", "BTC/USDT:USDT")
        assert pos_a["size"] == 0.002
        assert pos_a["entry_price"] == 80000.0

    def test_get_all_symbols(self) -> None:
        """모든 심볼 집합 반환."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)
        self.tracker.open("strat_b", "ETH/USDT:USDT", "long", 0.05, 3000.0)

        symbols = self.tracker.get_all_symbols()
        assert symbols == {"BTC/USDT:USDT", "ETH/USDT:USDT"}

    def test_get_strategies_for_symbol(self) -> None:
        """심볼에 포지션을 보유한 전략 목록."""
        self.tracker.open("strat_a", "BTC/USDT:USDT", "long", 0.002, 80000.0)
        self.tracker.open("strat_b", "BTC/USDT:USDT", "long", 0.001, 81000.0)

        strategies = self.tracker.get_strategies_for_symbol("BTC/USDT:USDT")
        assert set(strategies) == {"strat_a", "strat_b"}

    def test_close_nonexistent(self) -> None:
        """존재하지 않는 포지션 close → 에러 없음."""
        self.tracker.close("strat_a", "BTC/USDT:USDT")  # 에러 없이 무시
        assert not self.tracker.has_position("strat_a", "BTC/USDT:USDT")
