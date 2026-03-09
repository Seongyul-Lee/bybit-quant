"""TripleBarrierLabeler 라벨링 검증 테스트."""

import numpy as np
import pandas as pd
import pytest

from strategies.lgbm_classifier.labeler import TripleBarrierLabeler


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성."""
    np.random.seed(42)
    n = 200
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": close + np.random.randn(n) * 10,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000) + 500,
    })

    # ATR 계산
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    return df


class TestTripleBarrierLabeler:
    """TripleBarrierLabeler 테스트."""

    def test_returns_series(self, sample_ohlcv: pd.DataFrame) -> None:
        """generate_labels가 Series를 반환하는지 확인."""
        labeler = TripleBarrierLabeler()
        labels = labeler.generate_labels(sample_ohlcv)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(sample_ohlcv)

    def test_label_values(self, sample_ohlcv: pd.DataFrame) -> None:
        """라벨 값이 -1, 0, 1, NaN 중 하나인지 확인."""
        labeler = TripleBarrierLabeler()
        labels = labeler.generate_labels(sample_ohlcv)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_last_bars_are_nan(self, sample_ohlcv: pd.DataFrame) -> None:
        """마지막 max_holding_period 봉이 NaN인지 확인."""
        holding = 24
        labeler = TripleBarrierLabeler(max_holding_period=holding)
        labels = labeler.generate_labels(sample_ohlcv)
        assert labels.iloc[-holding:].isna().all()

    def test_first_bars_are_nan_due_to_atr(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR 계산 전 (처음 ~14봉)은 NaN인지 확인."""
        labeler = TripleBarrierLabeler()
        labels = labeler.generate_labels(sample_ohlcv)
        # ATR 14봉 필요 → 처음 13봉은 NaN이어야 함
        assert labels.iloc[:13].isna().all()

    def test_clear_uptrend_labels_buy(self) -> None:
        """명확한 상승 추세에서 매수(1) 라벨이 생성되는지 확인."""
        n = 100
        close = np.arange(1000, 1000 + n * 10, 10, dtype=float)
        high = close + 5
        low = close - 5
        atr = np.full(n, 50.0)  # ATR=50 고정

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000,
            "atr_14": atr,
        })

        labeler = TripleBarrierLabeler(upper_multiplier=2.0, lower_multiplier=1.0)
        labels = labeler.generate_labels(df)
        valid = labels.dropna()
        # 강한 상승 → 대부분 매수 라벨
        buy_ratio = (valid == 1).sum() / len(valid)
        assert buy_ratio > 0.5, f"상승 추세에서 매수 비율이 너무 낮음: {buy_ratio:.2f}"

    def test_clear_downtrend_labels_sell(self) -> None:
        """명확한 하락 추세에서 매도(-1) 라벨이 생성되는지 확인."""
        n = 100
        close = np.arange(2000, 2000 - n * 10, -10, dtype=float)
        high = close + 5
        low = close - 5
        atr = np.full(n, 50.0)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000,
            "atr_14": atr,
        })

        labeler = TripleBarrierLabeler(upper_multiplier=2.0, lower_multiplier=1.0)
        labels = labeler.generate_labels(df)
        valid = labels.dropna()
        sell_ratio = (valid == -1).sum() / len(valid)
        assert sell_ratio > 0.5, f"하락 추세에서 매도 비율이 너무 낮음: {sell_ratio:.2f}"

    def test_custom_parameters(self, sample_ohlcv: pd.DataFrame) -> None:
        """커스텀 파라미터가 정상 적용되는지 확인."""
        labeler = TripleBarrierLabeler(
            upper_multiplier=3.0,
            lower_multiplier=2.0,
            max_holding_period=12,
        )
        labels = labeler.generate_labels(sample_ohlcv)
        # 배리어가 넓어지면 중립(0) 비율이 높아져야 함
        valid = labels.dropna()
        assert len(valid) > 0

    def test_computes_atr_if_missing(self) -> None:
        """atr_14 컬럼이 없어도 자체 계산하는지 확인."""
        n = 100
        np.random.seed(42)
        close = 40000 + np.cumsum(np.random.randn(n) * 100)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close + np.random.randn(n) * 10,
            "high": close + np.abs(np.random.randn(n) * 50),
            "low": close - np.abs(np.random.randn(n) * 50),
            "close": close,
            "volume": np.ones(n) * 1000,
        })

        labeler = TripleBarrierLabeler()
        labels = labeler.generate_labels(df)
        assert isinstance(labels, pd.Series)
        assert len(labels) == n
