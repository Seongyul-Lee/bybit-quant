"""BybitDataCollector 단위 테스트."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.collector import BybitDataCollector


class TestBybitDataCollector(unittest.TestCase):
    """BybitDataCollector 테스트 클래스."""

    @patch("src.data.collector.ccxt.bybit")
    def setUp(self, mock_bybit: MagicMock) -> None:
        """테스트 전 mock exchange로 collector 초기화."""
        self.mock_exchange = MagicMock()
        mock_bybit.return_value = self.mock_exchange
        self.collector = BybitDataCollector(api_key="test", secret="test")
        self.collector.exchange = self.mock_exchange

    def test_fetch_ohlcv_returns_dataframe(self) -> None:
        """fetch_ohlcv가 올바른 컬럼의 DataFrame을 반환하는지 검증."""
        self.mock_exchange.fetch_ohlcv.return_value = [
            [1704067200000, 42000.0, 42500.0, 41800.0, 42300.0, 100.5],
            [1704070800000, 42300.0, 42700.0, 42100.0, 42600.0, 95.2],
        ]
        self.mock_exchange.parse8601.return_value = 1704067200000

        df = self.collector.fetch_ohlcv(
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            since="2024-01-01T00:00:00Z",
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(
            list(df.columns),
            ["timestamp", "open", "high", "low", "close", "volume"],
        )

    def test_fetch_ohlcv_empty_result(self) -> None:
        """API가 빈 결과를 반환할 때 빈 DataFrame을 반환하는지 검증."""
        self.mock_exchange.fetch_ohlcv.return_value = []

        df = self.collector.fetch_ohlcv(symbol="BTC/USDT:USDT", timeframe="1h")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)

    def test_fetch_funding_rate_returns_dataframe(self) -> None:
        """fetch_funding_rate가 올바른 DataFrame을 반환하는지 검증."""
        self.mock_exchange.fetch_funding_rate_history.return_value = [
            {"timestamp": 1704067200000, "symbol": "BTC/USDT:USDT", "fundingRate": 0.0001},
        ]

        df = self.collector.fetch_funding_rate(symbol="BTC/USDT:USDT")

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn("funding_rate", df.columns)

    def test_timeframe_to_ms(self) -> None:
        """타임프레임 문자열→밀리초 변환 검증."""
        self.assertEqual(BybitDataCollector._timeframe_to_ms("1m"), 60_000)
        self.assertEqual(BybitDataCollector._timeframe_to_ms("5m"), 300_000)
        self.assertEqual(BybitDataCollector._timeframe_to_ms("1h"), 3_600_000)
        self.assertEqual(BybitDataCollector._timeframe_to_ms("1d"), 86_400_000)


if __name__ == "__main__":
    unittest.main()
