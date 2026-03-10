"""모델 평가 모듈.

ML 메트릭과 트레이딩 메트릭을 통합 평가한다.
"""

import numpy as np
import pandas as pd
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
            y_true: 실제 라벨 (0, 1, 2).
            y_pred: 예측 라벨 (0, 1, 2).
            y_proba: 예측 확률 (n_samples, 3).

        Returns:
            {"f1_macro", "auc_roc_ovr"} 딕셔너리.
        """
        f1 = f1_score(y_true, y_pred, average="macro")

        # AUC-ROC OvR (클래스가 2개 이상이어야 계산 가능)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0

        return {
            "f1_macro": float(f1),
            "auc_roc_ovr": float(auc),
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
