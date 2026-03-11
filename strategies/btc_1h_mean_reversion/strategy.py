"""LightGBM 이진 분류 평균회귀 전략 (롱 전용).

과매도 구간에서 평균 복귀를 예측하는 LightGBM 모델 기반 전략.
2클래스 분류: 매수(1), 비매수(0).
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from strategies._common.features import FeatureEngine


class MeanReversionStrategy(BaseStrategy):
    """LightGBM 기반 평균회귀 전략 (롱 전용).

    과매도 구간에서 반등 확률이 높은 시점을 포착한다.
    모멘텀 전략과 낮은 상관관계를 목표로 설계.

    Config 키:
        model_path: 학습된 모델 파일 경로 (.txt).
        feature_names_path: 피처 이름 JSON 경로.
        confidence_threshold: 최소 확률 임계값 (기본 0.5).
        ensemble_folds: 앙상블에 사용할 fold 인덱스 리스트.
        models_dir: fold 모델이 저장된 디렉토리.
    """

    def __init__(self, config: dict) -> None:
        """MeanReversionStrategy 초기화.

        Args:
            config: 전략 파라미터 딕셔너리.
        """
        super().__init__(config)
        self.feature_engine = FeatureEngine(config)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.ensemble_folds = config.get("ensemble_folds", None)
        self.models_dir = config.get(
            "models_dir", "strategies/btc_1h_mean_reversion/models"
        )

        if self.ensemble_folds:
            self.models = self._load_ensemble_models()
            self.model = None
        else:
            self.model = self._load_model()
            self.models = None

        self.feature_names = self._load_feature_names()

        # 펀딩비 적응형 threshold 설정
        funding_filter = config.get("funding_filter", {})
        self.funding_filter_enabled = funding_filter.get("enabled", False)
        self.zscore_thresholds = funding_filter.get("zscore_thresholds", [])

        # OI 필터 설정
        oi_filter = config.get("oi_filter", {})
        self.oi_filter_enabled = oi_filter.get("enabled", False)
        self.oi_block_zscore = oi_filter.get("block_zscore", None)

    def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """마지막 봉의 피처로 매매 신호 생성.

        Args:
            df: OHLCV + 피처 데이터프레임.

        Returns:
            (signal, probability) 튜플.
            signal: 1(매수) 또는 0(비매수).
            probability: 매수 확률 (0.0 ~ 1.0).
        """
        df_feat = self.feature_engine.compute_all_features(df)
        last_row = df_feat[self.feature_names].iloc[[-1]]

        if last_row.isna().any(axis=1).iloc[0]:
            return 0, 0.0

        proba = self._predict(last_row)[0]

        threshold = self._get_adaptive_threshold(df_feat.iloc[-1])

        # OI 필터: 과밀 포지션 구간 차단
        if self.oi_filter_enabled and self.oi_block_zscore is not None:
            oi_z = df_feat.iloc[-1].get("oi_zscore", np.nan)
            if not np.isnan(oi_z) and oi_z >= self.oi_block_zscore:
                return 0, float(proba)

        if proba >= threshold:
            return 1, float(proba)
        return 0, float(proba)

    def generate_signals_vectorized(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """전체 데이터에 대해 벡터화 신호 생성 (백테스트 전용).

        Args:
            df: OHLCV + 피처 데이터프레임.

        Returns:
            (signal_series, probability_series) 튜플.
        """
        df_feat = self.feature_engine.compute_all_features(df)

        X = df_feat[self.feature_names]
        valid_mask = ~X.isna().any(axis=1)

        signals = pd.Series(0, index=df.index, dtype=int)
        probabilities = pd.Series(0.0, index=df.index, dtype=float)

        if valid_mask.sum() == 0:
            return signals, probabilities

        X_valid = X[valid_mask]
        proba = self._predict(X_valid)
        probabilities.loc[valid_mask] = proba

        if self.funding_filter_enabled and "funding_rate_zscore" in df_feat.columns:
            fr_zscore = df_feat["funding_rate_zscore"].values
            adaptive_thr = np.full(len(df), 999.0)
            for rule in sorted(self.zscore_thresholds,
                               key=lambda x: x["zscore_below"], reverse=True):
                adaptive_thr[fr_zscore < rule["zscore_below"]] = rule["confidence"]
            adaptive_thr[np.isnan(fr_zscore)] = self.confidence_threshold
        else:
            adaptive_thr = np.full(len(df), self.confidence_threshold)

        # OI 필터: 과밀 포지션 구간 차단
        if (self.oi_filter_enabled and self.oi_block_zscore is not None
                and "oi_zscore" in df_feat.columns):
            oi_z = df_feat["oi_zscore"].values
            block_mask = (oi_z >= self.oi_block_zscore) & ~np.isnan(oi_z)
            adaptive_thr[block_mask] = 999.0

        signal_values = np.where(proba >= adaptive_thr[valid_mask], 1, 0)
        signals.loc[valid_mask] = signal_values

        return signals, probabilities

    def _get_adaptive_threshold(self, row: pd.Series) -> float:
        """펀딩비 z-score에 따라 적응형 confidence threshold 반환."""
        if not self.funding_filter_enabled:
            return self.confidence_threshold

        zscore = row.get("funding_rate_zscore", np.nan)
        if np.isnan(zscore):
            return self.confidence_threshold

        for rule in sorted(self.zscore_thresholds,
                           key=lambda x: x["zscore_below"]):
            if zscore < rule["zscore_below"]:
                return rule["confidence"]
        return 999.0

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """모델 예측. 앙상블이면 평균, 단일이면 직접 예측."""
        if self.models:
            preds = [m.predict(X) for m in self.models]
            return np.mean(preds, axis=0)
        return self.model.predict(X)

    def _load_model(self) -> lgb.Booster:
        """저장된 LightGBM Booster 모델 로드."""
        model_path = self.config.get(
            "model_path",
            "strategies/btc_1h_mean_reversion/models/latest.txt",
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_path}\n"
                f"먼저 train_lgbm.py로 모델을 학습하세요."
            )
        return lgb.Booster(model_file=model_path)

    def _load_ensemble_models(self) -> list[lgb.Booster]:
        """앙상블 fold 모델들을 로드."""
        models = []
        for fold_idx in self.ensemble_folds:
            path = os.path.join(self.models_dir, f"fold_{fold_idx:02d}.txt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Fold 모델 파일을 찾을 수 없습니다: {path}\n"
                    f"먼저 train_lgbm.py로 모델을 학습하세요."
                )
            models.append(lgb.Booster(model_file=path))
        return models

    def _load_feature_names(self) -> list[str]:
        """피처 이름 목록 로드."""
        path = self.config.get(
            "feature_names_path",
            "strategies/btc_1h_mean_reversion/models/feature_names.json",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"피처 이름 파일을 찾을 수 없습니다: {path}\n"
                f"먼저 train_lgbm.py로 모델을 학습하세요."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
