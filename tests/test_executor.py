"""OrderExecutor 단위 테스트."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import ccxt

from src.execution.executor import OrderExecutor


class TestOrderExecutor(unittest.TestCase):
    """OrderExecutor 테스트 클래스."""

    def setUp(self) -> None:
        """테스트 전 mock exchange로 executor 초기화."""
        self.mock_exchange = MagicMock(spec=ccxt.bybit)
        self.executor = OrderExecutor(exchange=self.mock_exchange)

    def test_execute_limit_order(self) -> None:
        """지정가 주문 실행 검증."""
        self.mock_exchange.create_limit_order.return_value = {
            "id": "order_001",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "type": "limit",
            "price": 42000.0,
            "amount": 0.01,
            "fee": {"cost": 0.168},
        }

        result = self.executor.execute(
            symbol="BTC/USDT:USDT",
            side="buy",
            amount=0.01,
            order_type="limit",
            price=42000.0,
            strategy_name="test_strategy",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "order_001")
        self.mock_exchange.create_limit_order.assert_called_once()

    def test_execute_market_order(self) -> None:
        """시장가 주문 실행 검증."""
        self.mock_exchange.create_market_order.return_value = {
            "id": "order_002",
            "symbol": "BTC/USDT:USDT",
            "side": "sell",
            "type": "market",
            "price": None,
            "average": 42100.0,
            "amount": 0.01,
            "fee": {"cost": 0.168},
        }

        result = self.executor.execute(
            symbol="BTC/USDT:USDT",
            side="sell",
            amount=0.01,
            order_type="market",
        )

        self.assertIsNotNone(result)
        self.mock_exchange.create_market_order.assert_called_once()

    def test_duplicate_order_prevention(self) -> None:
        """동일 심볼/방향 중복 주문 방지 검증."""
        self.executor.pending_orders["existing"] = {
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
        }

        result = self.executor.execute(
            symbol="BTC/USDT:USDT",
            side="buy",
            amount=0.01,
            order_type="market",
        )

        self.assertIsNone(result)
        self.mock_exchange.create_market_order.assert_not_called()

    def test_cancel_order(self) -> None:
        """주문 취소 검증."""
        self.executor.pending_orders["order_001"] = {"symbol": "BTC/USDT:USDT"}

        result = self.executor.cancel("order_001", "BTC/USDT:USDT")

        self.assertTrue(result)
        self.assertNotIn("order_001", self.executor.pending_orders)

    def test_cancel_order_failure(self) -> None:
        """주문 취소 실패 검증."""
        self.mock_exchange.cancel_order.side_effect = ccxt.BaseError("Not found")

        result = self.executor.cancel("nonexistent", "BTC/USDT:USDT")

        self.assertFalse(result)

    def test_sync_positions(self) -> None:
        """포지션 동기화 검증."""
        self.mock_exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": "0.05",
                "entryPrice": "42000",
                "unrealizedPnl": "150",
            },
        ]
        self.mock_exchange.fetch_open_orders.return_value = []

        positions = self.executor.sync_positions()

        self.assertIn("BTC/USDT:USDT", positions)
        self.assertEqual(positions["BTC/USDT:USDT"]["side"], "long")


if __name__ == "__main__":
    unittest.main()
