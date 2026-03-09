"""WalkForwardTrainer fold 분할 및 embargo 검증 테스트."""

import numpy as np
import pandas as pd
import pytest

from strategies.lgbm_classifier.trainer import WalkForwardTrainer, LABEL_MAP


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """12개월 분량의 테스트 데이터 생성."""
    np.random.seed(42)
    n = 24 * 365  # 1년 ≈ 8760봉 (1h)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": close,
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.randn(n),
        "label": np.random.choice([-1, 0, 1], size=n),
    })

    return df


class TestWalkForwardTrainer:
    """WalkForwardTrainer 테스트."""

    def test_generate_folds_count(self, sample_data: pd.DataFrame) -> None:
        """12개월 데이터에서 적절한 수의 fold가 생성되는지 확인.

        min_train=6, val=1 → 최대 6개 fold (6+1, 7+1, ..., 11+1).
        """
        trainer = WalkForwardTrainer(
            min_train_months=6, val_months=1, embargo_bars=24
        )
        folds = trainer.generate_folds(sample_data)
        assert len(folds) >= 4, f"fold 수가 4개 미만: {len(folds)}"
        assert len(folds) <= 7, f"fold 수가 7개 초과: {len(folds)}"

    def test_folds_no_overlap(self, sample_data: pd.DataFrame) -> None:
        """학습/검증 인덱스가 겹치지 않는지 확인."""
        trainer = WalkForwardTrainer(
            min_train_months=6, val_months=1, embargo_bars=24
        )
        folds = trainer.generate_folds(sample_data)

        for fold in folds:
            train_set = set(fold["train_idx"])
            val_set = set(fold["val_idx"])
            overlap = train_set.intersection(val_set)
            assert len(overlap) == 0, f"학습/검증 인덱스 겹침: {len(overlap)}건"

    def test_embargo_gap(self, sample_data: pd.DataFrame) -> None:
        """학습 끝 인덱스와 검증 시작 인덱스 사이에 embargo gap이 있는지 확인."""
        embargo = 24
        trainer = WalkForwardTrainer(
            min_train_months=6, val_months=1, embargo_bars=embargo
        )
        folds = trainer.generate_folds(sample_data)

        for fold in folds:
            train_end = max(fold["train_idx"])
            val_start = min(fold["val_idx"])
            gap = val_start - train_end
            assert gap >= embargo, (
                f"Embargo gap 부족: {gap} < {embargo} "
                f"(train_end={train_end}, val_start={val_start})"
            )

    def test_expanding_window(self, sample_data: pd.DataFrame) -> None:
        """학습 윈도우가 매 fold마다 확장되는지 확인."""
        trainer = WalkForwardTrainer(
            min_train_months=6, val_months=1, embargo_bars=24
        )
        folds = trainer.generate_folds(sample_data)

        for i in range(1, len(folds)):
            prev_train_size = len(folds[i - 1]["train_idx"])
            curr_train_size = len(folds[i]["train_idx"])
            assert curr_train_size > prev_train_size, (
                f"Fold {i} 학습 크기({curr_train_size})가 "
                f"Fold {i - 1}({prev_train_size})보다 크지 않음"
            )

    def test_insufficient_data_returns_empty(self) -> None:
        """데이터가 부족하면 빈 fold 리스트를 반환하는지 확인."""
        # 2개월 데이터 → min_train_months=6이면 fold 불가
        n = 24 * 60  # 60일
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "close": np.random.randn(n),
        })

        trainer = WalkForwardTrainer(min_train_months=6, val_months=1)
        folds = trainer.generate_folds(df)
        assert len(folds) == 0

    def test_label_mapping(self) -> None:
        """라벨 매핑 (-1→0, 0→1, 1→2)이 올바른지 확인."""
        assert LABEL_MAP[-1] == 0
        assert LABEL_MAP[0] == 1
        assert LABEL_MAP[1] == 2

    def test_fold_has_required_keys(self, sample_data: pd.DataFrame) -> None:
        """각 fold에 필수 키가 있는지 확인."""
        trainer = WalkForwardTrainer(
            min_train_months=6, val_months=1, embargo_bars=24
        )
        folds = trainer.generate_folds(sample_data)

        required_keys = {"train_start", "train_end", "val_start", "val_end",
                         "train_idx", "val_idx"}
        for fold in folds:
            assert required_keys.issubset(fold.keys()), (
                f"필수 키 누락: {required_keys - fold.keys()}"
            )
