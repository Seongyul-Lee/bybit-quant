"""DataProcessor 단위 테스트."""

import unittest

import numpy as np
import pandas as pd

from src.data.processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """DataProcessor 테스트 클래스."""

    def setUp(self) -> None:
        """테스트용 샘플 OHLCV 데이터 생성."""
        self.processor = DataProcessor(spike_threshold=0.1)
        np.random.seed(42)
        n = 250
        prices = 42000 + np.cumsum(np.random.randn(n) * 100)
        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": prices,
            "high": prices + np.random.rand(n) * 200,
            "low": prices - np.random.rand(n) * 200,
            "close": prices + np.random.randn(n) * 50,
            "volume": np.random.rand(n) * 1000 + 100,
        })

    def test_add_features_columns(self) -> None:
        """add_features가 모든 피처 컬럼을 추가하는지 검증."""
        result = self.processor.add_features(self.df)
        expected_cols = ["ma_10", "ma_20", "ma_50", "ma_200", "rsi_14",
                         "bb_upper", "bb_mid", "bb_lower", "volume_ma_20", "atr_14"]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"컬럼 {col} 누락")

    def test_add_features_preserves_rows(self) -> None:
        """add_features가 행 수를 변경하지 않는지 검증."""
        result = self.processor.add_features(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_detect_spike(self) -> None:
        """detect_spike가 is_spike 컬럼을 추가하는지 검증."""
        result = self.processor.detect_spike(self.df)
        self.assertIn("is_spike", result.columns)
        self.assertEqual(result["is_spike"].dtype, bool)

    def test_detect_spike_with_artificial_spike(self) -> None:
        """인위적 스파이크가 감지되는지 검증."""
        df = self.df.copy()
        df.loc[10, "close"] = df.loc[9, "close"] * 1.15  # 15% 급등
        result = self.processor.detect_spike(df, threshold=0.1)
        self.assertTrue(result.loc[10, "is_spike"])

    def test_fill_gaps(self) -> None:
        """fill_gaps가 누락 타임스탬프를 채우는지 검증."""
        # 중간 행 제거해서 갭 생성
        df = self.df.drop(index=[5, 6, 7]).reset_index(drop=True)
        result = self.processor.fill_gaps(df, timeframe_minutes=60)
        self.assertGreater(len(result), len(df))

    def test_fill_gaps_empty_df(self) -> None:
        """빈 DataFrame에 대해 fill_gaps가 빈 결과를 반환하는지 검증."""
        empty = pd.DataFrame(columns=self.df.columns)
        result = self.processor.fill_gaps(empty, timeframe_minutes=60)
        self.assertEqual(len(result), 0)

    def test_compute_rsi_range(self) -> None:
        """RSI 값이 0~100 범위인지 검증."""
        rsi = DataProcessor._compute_rsi(self.df["close"], period=14)
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())


if __name__ == "__main__":
    unittest.main()
