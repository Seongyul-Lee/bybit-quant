"""LGBMClassifierStrategy generate_signal 인터페이스 검증 테스트.

mock 모델을 사용하여 실제 LightGBM 학습 없이 전략 인터페이스를 검증한다.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from strategies.lgbm_classifier.strategy import LGBMClassifierStrategy


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성 (500봉)."""
    np.random.seed(42)
    n = 500
    close = 40000 + np.cumsum(np.random.randn(n) * 100)

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": close + np.random.randn(n) * 30,
        "high": close + np.abs(np.random.randn(n) * 50),
        "low": close - np.abs(np.random.randn(n) * 50),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000) + 500,
    })


@pytest.fixture
def mock_model_files(tmp_path):
    """mock 모델 파일을 임시 디렉토리에 생성."""
    model_path = str(tmp_path / "latest.txt")
    feature_names_path = str(tmp_path / "feature_names.json")

    # feature_names.json
    feature_names = ["ma_10", "ma_30", "rsi_14", "macd", "atr_14", "return_1"]
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)

    return {
        "model_path": model_path,
        "feature_names_path": feature_names_path,
        "feature_names": feature_names,
    }


def _create_mock_strategy(mock_model_files: dict, confidence: float = 0.5):
    """mock 모델이 주입된 전략 인스턴스 생성."""
    config = {
        "model_path": mock_model_files["model_path"],
        "feature_names_path": mock_model_files["feature_names_path"],
        "confidence_threshold": confidence,
    }

    # _load_model을 패치하여 mock 모델 반환
    mock_model = MagicMock()

    with patch.object(LGBMClassifierStrategy, "_load_model", return_value=mock_model):
        strategy = LGBMClassifierStrategy(config=config)

    return strategy


class TestLGBMClassifierStrategy:
    """LGBMClassifierStrategy 인터페이스 테스트."""

    def test_generate_signal_returns_int(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """generate_signal이 int를 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files)
        # mock 모델이 중립을 반환하도록 설정
        strategy.model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])

        signal = strategy.generate_signal(sample_ohlcv)
        assert isinstance(signal, (int, np.integer))
        assert signal in {-1, 0, 1}

    def test_generate_signal_buy(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """높은 매수 확률이면 1을 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files)
        # p_up(class 2) = 0.7 → 매수
        strategy.model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

        signal = strategy.generate_signal(sample_ohlcv)
        assert signal == 1

    def test_generate_signal_sell(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """높은 매도 확률이면 -1을 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files)
        # p_down(class 0) = 0.7 → 매도
        strategy.model.predict_proba.return_value = np.array([[0.7, 0.2, 0.1]])

        signal = strategy.generate_signal(sample_ohlcv)
        assert signal == -1

    def test_generate_signal_neutral_below_threshold(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """모든 확률이 threshold 미만이면 0을 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files, confidence=0.5)
        # 최대 확률 = 0.4 < 0.5 → 중립
        strategy.model.predict_proba.return_value = np.array([[0.3, 0.4, 0.3]])

        signal = strategy.generate_signal(sample_ohlcv)
        assert signal == 0

    def test_generate_signals_vectorized_returns_series(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """generate_signals_vectorized가 Series를 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files)

        # predict_proba가 입력 크기에 맞게 동적 반환
        def dynamic_proba(X):
            return np.tile([0.1, 0.8, 0.1], (len(X), 1))

        strategy.model.predict_proba.side_effect = dynamic_proba

        signals = strategy.generate_signals_vectorized(sample_ohlcv)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_generate_signals_vectorized_confidence_filter(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """벡터화 신호에서 confidence 필터가 적용되는지 확인."""
        strategy = _create_mock_strategy(mock_model_files, confidence=0.6)

        def dynamic_proba(X):
            return np.tile([0.3, 0.4, 0.3], (len(X), 1))

        strategy.model.predict_proba.side_effect = dynamic_proba

        signals = strategy.generate_signals_vectorized(sample_ohlcv)
        assert (signals == 0).all()

    def test_get_params_returns_config(self, mock_model_files: dict) -> None:
        """get_params가 config를 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files)
        params = strategy.get_params()
        assert "confidence_threshold" in params

    def test_inherits_base_strategy(self, mock_model_files: dict) -> None:
        """BaseStrategy를 상속하는지 확인."""
        from src.strategies.base import BaseStrategy
        strategy = _create_mock_strategy(mock_model_files)
        assert isinstance(strategy, BaseStrategy)
