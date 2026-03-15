"""모델 평가 모듈.

ML 메트릭(분류/회귀)과 트레이딩 메트릭을 통합 평가한다.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from src.analytics.reporter import Reporter


class ModelEvaluator:
    """ML 및 트레이딩 관점에서 모델 성능을 평가.

    Reporter.calculate_metrics()를 재활용하여 트레이딩 메트릭을 계산한다.
    """

    @staticmethod
    def ml_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict:
        """ML 분류 성능 지표 계산.

        Args:
            y_true: 실제 라벨 (0, 1).
            y_pred: 예측 라벨 (0, 1).
            y_proba: 예측 확률 (n_samples,) — positive class(1)의 확률.

        Returns:
            {"f1_binary", "auc_roc"} 딕셔너리.
        """
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)

        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.0

        return {
            "f1_binary": float(f1),
            "auc_roc": float(auc),
        }

    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """회귀 모델 성능 지표 계산.

        Args:
            y_true: 실제 수익률.
            y_pred: 예측 수익률.

        Returns:
            {"mae", "ic", "directional_accuracy"} 딕셔너리.
        """
        mae = float(np.mean(np.abs(y_pred - y_true)))
        ic, _ = spearmanr(y_pred, y_true)
        ic = float(ic) if not np.isnan(ic) else 0.0
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

        return {
            "mae": mae,
            "ic": ic,
            "directional_accuracy": dir_acc,
        }

    @staticmethod
    def check_overfitting_regression(
        train_mae: float,
        val_mae: float,
        threshold: float = 0.005,
    ) -> dict:
        """회귀 모델 과적합 판단.

        Args:
            train_mae: 학습 MAE.
            val_mae: 검증 MAE.
            threshold: 허용 MAE 갭.

        Returns:
            {"gap", "is_overfit"}.
        """
        gap = val_mae - train_mae
        return {
            "gap": float(gap),
            "is_overfit": gap > threshold,
        }

    @staticmethod
    def check_overfitting(
        train_f1: float,
        val_f1: float,
        threshold: float = 0.15,
    ) -> dict:
        """과적합 여부를 판단.

        Args:
            train_f1: 학습 F1 macro.
            val_f1: 검증 F1 macro.
            threshold: 허용 갭 임계값.

        Returns:
            {"gap", "is_overfit"} 딕셔너리.
        """
        gap = train_f1 - val_f1
        return {
            "gap": float(gap),
            "is_overfit": gap > threshold,
        }

    @staticmethod
    def walk_forward_stability(fold_sharpes: list[float]) -> dict:
        """Walk-Forward fold 간 성능 안정성 평가.

        Args:
            fold_sharpes: fold별 샤프 비율 리스트.

        Returns:
            {"mean_sharpe", "std_sharpe", "min_sharpe", "positive_ratio"} 딕셔너리.
        """
        arr = np.array(fold_sharpes)
        return {
            "mean_sharpe": float(arr.mean()),
            "std_sharpe": float(arr.std()),
            "min_sharpe": float(arr.min()),
            "positive_ratio": float((arr > 0).mean()),
        }

    @staticmethod
    def trading_metrics(returns: pd.Series, timeframe: str = "1d") -> dict:
        """트레이딩 성과 지표 계산 (Reporter.calculate_metrics 재활용).

        Args:
            returns: 수익률 시리즈.
            timeframe: 데이터 타임프레임 (연환산 계수 결정에 사용).

        Returns:
            Reporter.calculate_metrics() 결과.
        """
        return Reporter.calculate_metrics(returns, timeframe=timeframe)
