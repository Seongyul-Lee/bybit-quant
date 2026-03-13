"""LightGBM 회귀 기반 양방향 전략 기본 클래스.

학습된 LGBMRegressor 모델이 예상 수익률을 예측하고:
- 양수 + 임계값 초과 → 롱 (signal = 1)
- 음수 + 임계값 초과 → 숏 (signal = -1)
- 임계값 미만 → 대기 (signal = 0)
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from strategies._common.features import FeatureEngine


class LGBMRegressorStrategy(BaseStrategy):
    """LightGBM 회귀 기반 양방향 전략.

    Config 키:
        model_path, feature_names_path, models_dir, ensemble_folds  ← 기존과 동일
        min_pred_threshold: 최소 예측값 (기본 0.005 = 0.5%)
        max_position_scale: 포지션 스케일 상한 (기본 2.0)
        sl_atr_mult: 동적 SL ATR 배수 (기본 2.0)
        tp_atr_mult: 동적 TP ATR 배수 (기본 3.0)
        min_sl_pct: SL 하한 (기본 0.01)
        max_sl_pct: SL 상한 (기본 0.05)
        min_tp_pct: TP 하한 (기본 0.01)
        max_tp_pct: TP 상한 (기본 0.08)
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.feature_engine = FeatureEngine(config)
        self.min_pred_threshold = config.get("min_pred_threshold", 0.005)
        self.max_position_scale = config.get("max_position_scale", 2.0)
        self.sl_atr_mult = config.get("sl_atr_mult", 2.0)
        self.tp_atr_mult = config.get("tp_atr_mult", 3.0)
        self.min_sl_pct = config.get("min_sl_pct", 0.01)
        self.max_sl_pct = config.get("max_sl_pct", 0.05)
        self.min_tp_pct = config.get("min_tp_pct", 0.01)
        self.max_tp_pct = config.get("max_tp_pct", 0.08)

        # 펀딩비 필터
        funding_filter = config.get("funding_filter", {})
        self.funding_filter_enabled = funding_filter.get("enabled", False)
        self.zscore_thresholds = funding_filter.get("zscore_thresholds", [])

        # OI 필터
        oi_filter = config.get("oi_filter", {})
        self.oi_filter_enabled = oi_filter.get("enabled", False)
        self.oi_block_zscore = oi_filter.get("block_zscore", 1.0)

        # 모델 로드
        self.ensemble_folds = config.get("ensemble_folds", None)
        self.models_dir = config.get("models_dir", "")
        if self.ensemble_folds:
            self.models = self._load_ensemble_models()
        else:
            self.models = [self._load_single_model()]
        self.feature_names = self._load_feature_names()

    def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """예상 수익률 기반 양방향 시그널.

        Returns:
            (signal, confidence)
            signal: 1(롱) / -1(숏) / 0(대기)
            confidence: |예측값| 정규화 (0.0 ~ 1.0)
        """
        df_feat = self.feature_engine.compute_all_features(df)
        last_row = df_feat[self.feature_names].iloc[[-1]]

        if last_row.isna().any(axis=1).iloc[0]:
            return 0, 0.0

        pred = self._predict(last_row)[0]

        # 펀딩비 적응형 threshold
        threshold = self._get_adaptive_threshold(df_feat.iloc[-1], pred)

        # OI 필터
        if self.oi_filter_enabled and "oi_zscore" in df_feat.columns:
            oi_z = df_feat["oi_zscore"].iloc[-1]
            if not np.isnan(oi_z) and oi_z >= self.oi_block_zscore:
                return 0, 0.0

        abs_pred = abs(pred)
        if abs_pred < threshold:
            return 0, 0.0

        signal = 1 if pred > 0 else -1
        confidence = min(
            abs_pred / (self.min_pred_threshold * self.max_position_scale), 1.0
        )

        return signal, float(confidence)

    def generate_signals_vectorized(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """벡터화 양방향 시그널 (백테스트 전용).

        Returns:
            (signal_series, confidence_series)
            signal_series: 1(롱) / -1(숏) / 0(대기)
            confidence_series: 0.0 ~ 1.0
        """
        df_feat = self.feature_engine.compute_all_features(df)
        X = df_feat[self.feature_names]
        valid_mask = ~X.isna().any(axis=1)

        signals = pd.Series(0, index=df.index, dtype=int)
        confidences = pd.Series(0.0, index=df.index, dtype=float)

        if valid_mask.sum() == 0:
            return signals, confidences

        X_valid = X[valid_mask]
        preds = self._predict(X_valid)

        abs_preds = np.abs(preds)
        threshold = self.min_pred_threshold

        # 펀딩비 적응형 threshold (벡터화)
        if (
            self.funding_filter_enabled
            and "funding_rate_zscore" in df_feat.columns
        ):
            fr_zscore = df_feat.loc[valid_mask, "funding_rate_zscore"].values
            adaptive_thr = np.full(len(X_valid), 999.0)
            for rule in sorted(
                self.zscore_thresholds,
                key=lambda x: x["zscore_below"],
                reverse=True,
            ):
                adaptive_thr[fr_zscore < rule["zscore_below"]] = rule[
                    "confidence"
                ]
            adaptive_thr[np.isnan(fr_zscore)] = threshold
            threshold_arr = adaptive_thr
        else:
            threshold_arr = np.full(len(X_valid), threshold)

        # OI 필터 (벡터화)
        if self.oi_filter_enabled and "oi_zscore" in df_feat.columns:
            oi_z = df_feat.loc[valid_mask, "oi_zscore"].values
            oi_block = (~np.isnan(oi_z)) & (oi_z >= self.oi_block_zscore)
            threshold_arr[oi_block] = 999.0

        # 시그널 생성
        long_mask = (preds > 0) & (abs_preds >= threshold_arr)
        short_mask = (preds < 0) & (abs_preds >= threshold_arr)

        signal_values = np.zeros(len(X_valid), dtype=int)
        signal_values[long_mask] = 1
        signal_values[short_mask] = -1

        conf_values = np.minimum(
            abs_preds / (self.min_pred_threshold * self.max_position_scale),
            1.0,
        )
        conf_values[signal_values == 0] = 0.0

        signals.loc[valid_mask] = signal_values
        confidences.loc[valid_mask] = conf_values

        return signals, confidences

    def get_dynamic_sl_tp(
        self, df: pd.DataFrame, signal: int
    ) -> tuple[float, float]:
        """ATR 기반 동적 SL/TP.

        Args:
            df: OHLCV + atr_14 데이터프레임.
            signal: 1(롱) 또는 -1(숏).

        Returns:
            (sl_pct, tp_pct) — 0.0~1.0 비율.
        """
        if "atr_14" not in df.columns or df["atr_14"].iloc[-1] == 0:
            return self.min_sl_pct, self.min_tp_pct

        atr_pct = df["atr_14"].iloc[-1] / df["close"].iloc[-1]
        sl_pct = max(
            self.min_sl_pct, min(atr_pct * self.sl_atr_mult, self.max_sl_pct)
        )
        tp_pct = max(
            self.min_tp_pct, min(atr_pct * self.tp_atr_mult, self.max_tp_pct)
        )

        return sl_pct, tp_pct

    # --- 내부 메서드 ---

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """앙상블 예측. 여러 모델의 예측값 평균."""
        preds = [m.predict(X) for m in self.models]
        return np.mean(preds, axis=0)

    def _load_single_model(self) -> lgb.Booster:
        """단일 모델 로드."""
        path = self.config.get("model_path", "")
        return lgb.Booster(model_file=path)

    def _load_ensemble_models(self) -> list[lgb.Booster]:
        """앙상블 fold 모델 로드."""
        models = []
        for fold_idx in self.ensemble_folds:
            path = os.path.join(self.models_dir, f"fold_{fold_idx:02d}.txt")
            models.append(lgb.Booster(model_file=path))
        return models

    def _load_feature_names(self) -> list[str]:
        """피처 이름 로드."""
        path = self.config.get("feature_names_path", "")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_adaptive_threshold(self, row: pd.Series, pred: float) -> float:
        """펀딩비 z-score 기반 적응형 threshold."""
        if not self.funding_filter_enabled:
            return self.min_pred_threshold
        zscore = row.get("funding_rate_zscore", np.nan)
        if np.isnan(zscore):
            return self.min_pred_threshold
        for rule in sorted(
            self.zscore_thresholds, key=lambda x: x["zscore_below"]
        ):
            if zscore < rule["zscore_below"]:
                return rule["confidence"]
        return 999.0  # 모든 threshold 초과 → 차단
