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
        strategy.model.predict.return_value = np.array([[0.1, 0.8, 0.1]])

        signal = strategy.generate_signal(sample_ohlcv)
        assert isinstance(signal, (int, np.integer))
        assert signal in {-1, 0, 1}

    def test_generate_signal_buy(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """높은 매수 확률이면 1을 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files)
        # p_up(class 2) = 0.7 → 매수
        strategy.model.predict.return_value = np.array([[0.1, 0.2, 0.7]])

        signal = strategy.generate_signal(sample_ohlcv)
        assert signal == 1

    def test_generate_signal_sell(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """높은 매도 확률이면 -1을 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files)
        # p_down(class 0) = 0.7 → 매도
        strategy.model.predict.return_value = np.array([[0.7, 0.2, 0.1]])

        signal = strategy.generate_signal(sample_ohlcv)
        assert signal == -1

    def test_generate_signal_neutral_below_threshold(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """모든 확률이 threshold 미만이면 0을 반환하는지 확인."""
        strategy = _create_mock_strategy(mock_model_files, confidence=0.5)
        # 최대 확률 = 0.4 < 0.5 → 중립
        strategy.model.predict.return_value = np.array([[0.3, 0.4, 0.3]])

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

        strategy.model.predict.side_effect = dynamic_proba

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

        strategy.model.predict.side_effect = dynamic_proba

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

    def test_vectorized_matches_sequential(
        self, sample_ohlcv: pd.DataFrame, mock_model_files: dict
    ) -> None:
        """Defect #7: generate_signal() 순차 호출과 generate_signals_vectorized() 결과가 일치하는지 검증.

        warmup 구간(처음 200봉) 이후 구간에서 동일한 신호를 반환해야 한다.
        """
        strategy = _create_mock_strategy(mock_model_files, confidence=0.5)

        # mock 모델: 피처 값 기반으로 결정론적 결과를 반환하도록 설정
        # 첫 번째 피처의 NaN 여부와 합계로 분류를 결정
        def deterministic_predict(X):
            """피처 합계 기반으로 결정론적 확률 반환."""
            if isinstance(X, pd.DataFrame):
                X_arr = X.values
            else:
                X_arr = np.asarray(X)

            results = []
            for row in X_arr:
                if np.any(np.isnan(row)):
                    results.append([0.1, 0.8, 0.1])  # 중립
                else:
                    s = np.sum(row)
                    # 합계를 기반으로 결정론적 확률 생성
                    if s % 3 < 1:
                        results.append([0.7, 0.2, 0.1])  # 매도
                    elif s % 3 < 2:
                        results.append([0.1, 0.8, 0.1])  # 중립
                    else:
                        results.append([0.1, 0.2, 0.7])  # 매수
            return np.array(results)

        strategy.model.predict.side_effect = deterministic_predict

        # 벡터화 신호 생성
        vec_signals = strategy.generate_signals_vectorized(sample_ohlcv)

        # 순차 신호 생성 (warmup 200봉 이후부터 비교)
        warmup = 200
        mismatches = 0
        total_compared = 0

        for i in range(warmup, len(sample_ohlcv)):
            seq_signal = strategy.generate_signal(sample_ohlcv.iloc[: i + 1])
            if vec_signals.iloc[i] != seq_signal:
                mismatches += 1
            total_compared += 1

        # warmup 이후 구간에서 불일치율이 1% 이하여야 함
        mismatch_rate = mismatches / total_compared if total_compared > 0 else 0.0
        assert mismatch_rate <= 0.01, (
            f"벡터화 vs 순차 신호 불일치율 {mismatch_rate:.2%} "
            f"({mismatches}/{total_compared}건) — 1% 초과"
        )
