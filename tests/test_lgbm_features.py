"""FeatureEngine 피처 계산 검증 테스트."""

import numpy as np
import pandas as pd
import pytest

from strategies.lgbm_classifier.features import FeatureEngine


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성 (500봉)."""
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000) + 500

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestFeatureEngine:
    """FeatureEngine 테스트."""

    def test_compute_all_features_returns_dataframe(self, sample_ohlcv: pd.DataFrame) -> None:
        """compute_all_features가 DataFrame을 반환하는지 확인."""
        engine = FeatureEngine(config={})
        result = engine.compute_all_features(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)

    def test_feature_count_minimum(self, sample_ohlcv: pd.DataFrame) -> None:
        """최소 40개 이상의 피처가 생성되는지 확인."""
        engine = FeatureEngine(config={})
        engine.compute_all_features(sample_ohlcv)
        names = engine.get_feature_names()
        assert len(names) >= 40, f"피처 수가 40개 미만: {len(names)}"

    def test_technical_indicators_present(self, sample_ohlcv: pd.DataFrame) -> None:
        """핵심 기술적 지표 컬럼이 존재하는지 확인."""
        engine = FeatureEngine(config={})
        result = engine.compute_all_features(sample_ohlcv)

        expected_cols = [
            "ma_10", "ma_30", "ma_50", "ma_200",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_width", "bb_position",
            "atr_14", "atr_14_pct", "adx_14",
            "stoch_k", "stoch_d", "obv",
        ]
        for col in expected_cols:
            assert col in result.columns, f"컬럼 누락: {col}"

    def test_price_features_present(self, sample_ohlcv: pd.DataFrame) -> None:
        """가격 파생 피처가 존재하는지 확인."""
        engine = FeatureEngine(config={})
        result = engine.compute_all_features(sample_ohlcv)

        for period in [1, 5, 10, 20]:
            assert f"return_{period}" in result.columns
        assert "log_return_1" in result.columns
        assert "high_low_ratio" in result.columns
        assert "body_ratio" in result.columns

    def test_volume_features_present(self, sample_ohlcv: pd.DataFrame) -> None:
        """거래량 파생 피처가 존재하는지 확인."""
        engine = FeatureEngine(config={})
        result = engine.compute_all_features(sample_ohlcv)

        assert "volume_ma_20" in result.columns
        assert "volume_change" in result.columns
        assert "volume_ratio" in result.columns

    def test_time_features_cyclical(self, sample_ohlcv: pd.DataFrame) -> None:
        """시간 피처가 cyclical encoding 범위 [-1, 1]인지 확인."""
        engine = FeatureEngine(config={})
        result = engine.compute_all_features(sample_ohlcv)

        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert col in result.columns
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0

    def test_multitimeframe_features(self, sample_ohlcv: pd.DataFrame) -> None:
        """멀티타임프레임 피처가 생성되는지 확인."""
        engine = FeatureEngine(config={})
        result = engine.compute_all_features(sample_ohlcv)

        # 500봉 = ~20일. 4h resample은 가능하나 1d는 50개 미만일 수 있음
        # 4h는 125봉 > 50이므로 생성 가능
        assert "rsi_14_4h" in result.columns
        assert "ma_50_4h" in result.columns
        assert "atr_14_4h" in result.columns

    def test_reuses_processor_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """processor가 이미 계산한 컬럼이 있으면 재활용하는지 확인."""
        df = sample_ohlcv.copy()
        # processor가 이미 ma_10을 계산했다고 가정
        df["ma_10"] = df["close"].rolling(10).mean()
        original_ma10 = df["ma_10"].copy()

        engine = FeatureEngine(config={})
        result = engine.compute_all_features(df)

        # 재활용했으므로 값이 동일해야 함
        pd.testing.assert_series_equal(result["ma_10"], original_ma10, check_names=False)

    def test_original_columns_preserved(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 OHLCV 컬럼이 보존되는지 확인."""
        engine = FeatureEngine(config={})
        result = engine.compute_all_features(sample_ohlcv)

        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_get_feature_names_excludes_ohlcv(self, sample_ohlcv: pd.DataFrame) -> None:
        """get_feature_names가 OHLCV 컬럼을 제외하는지 확인."""
        engine = FeatureEngine(config={})
        engine.compute_all_features(sample_ohlcv)
        names = engine.get_feature_names()

        excluded = {"timestamp", "open", "high", "low", "close", "volume"}
        assert not excluded.intersection(names), f"OHLCV 컬럼이 피처에 포함됨: {excluded.intersection(names)}"
