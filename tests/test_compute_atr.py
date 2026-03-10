"""_compute_atr 헬퍼 함수 단위 테스트."""

import numpy as np
import pandas as pd
import pytest

from main import _compute_atr


class TestComputeAtr:
    """_compute_atr 함수 테스트."""

    def _make_ohlcv(self, n: int = 30) -> pd.DataFrame:
        """테스트용 OHLCV DataFrame 생성."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return pd.DataFrame({
            "open": close + np.random.randn(n) * 0.1,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        })

    def test_returns_positive_for_valid_data(self) -> None:
        """충분한 데이터가 있으면 양수 ATR을 반환."""
        df = self._make_ohlcv(30)
        atr = _compute_atr(df, period=14)
        assert atr > 0

    def test_returns_zero_for_insufficient_data(self) -> None:
        """데이터가 부족하면 0.0을 반환."""
        df = self._make_ohlcv(10)
        atr = _compute_atr(df, period=14)
        assert atr == 0.0

    def test_returns_zero_for_empty_dataframe(self) -> None:
        """빈 DataFrame이면 0.0을 반환."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        atr = _compute_atr(df, period=14)
        assert atr == 0.0

    def test_custom_period(self) -> None:
        """커스텀 period 파라미터 동작 확인."""
        df = self._make_ohlcv(50)
        atr_7 = _compute_atr(df, period=7)
        atr_14 = _compute_atr(df, period=14)
        # 둘 다 양수여야 함
        assert atr_7 > 0
        assert atr_14 > 0
        # 값이 다를 수 있음 (같은 데이터, 다른 기간)
        assert atr_7 != atr_14

    def test_matches_manual_calculation(self) -> None:
        """수동 계산과 결과가 일치하는지 확인."""
        df = self._make_ohlcv(30)
        period = 14

        # 수동 계산
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        expected = float(tr.rolling(period).mean().iloc[-1])

        atr = _compute_atr(df, period=period)
        assert abs(atr - expected) < 1e-10
