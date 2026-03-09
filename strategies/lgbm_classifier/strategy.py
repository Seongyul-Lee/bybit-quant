"""LightGBM Leaf-Wise 분류 전략.

학습된 LightGBM 모델로 predict_proba 기반 매매 신호를 생성한다.
3클래스 분류: 매수(1), 중립(0), 매도(-1).
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from strategies.lgbm_classifier.features import FeatureEngine


class LGBMClassifierStrategy(BaseStrategy):
    """LightGBM 기반 3클래스 분류 전략.

    학습된 모델의 predict_proba 결과에서
    confidence_threshold 이상인 클래스의 신호를 반환한다.

    Config 키:
        model_path: 학습된 모델 파일 경로 (.txt).
        feature_names_path: 피처 이름 JSON 경로.
        confidence_threshold: 최소 확률 임계값 (기본 0.5).
    """

    # 라벨 역매핑: 모델 출력(0,1,2) → 시그널(-1,0,1)
    _LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}

    def __init__(self, config: dict) -> None:
        """LGBMClassifierStrategy 초기화.

        Args:
            config: 전략 파라미터 딕셔너리.
        """
        super().__init__(config)
        self.feature_engine = FeatureEngine(config)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()

    def generate_signal(self, df: pd.DataFrame) -> int:
        """마지막 봉의 피처로 매매 신호 생성.

        1. compute_all_features(df)
        2. 마지막 봉 피처 추출
        3. model.predict_proba → [p_down, p_neutral, p_up]
        4. argmax 확률 >= confidence_threshold → 해당 시그널, 아니면 0

        Args:
            df: OHLCV + 피처 데이터프레임.

        Returns:
            1(매수), -1(매도), 0(중립).
        """
        df_feat = self.feature_engine.compute_all_features(df)

        # 마지막 봉 피처 추출
        last_row = df_feat[self.feature_names].iloc[[-1]]

        # NaN 체크
        if last_row.isna().any(axis=1).iloc[0]:
            return 0

        proba = self.model.predict_proba(last_row)[0]  # [p_down, p_neutral, p_up]
        max_idx = int(np.argmax(proba))
        max_prob = proba[max_idx]

        if max_prob >= self.confidence_threshold:
            return self._LABEL_MAP_INV[max_idx]
        return 0

    def generate_signals_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """전체 데이터에 대해 벡터화 신호 생성 (백테스트 전용).

        Args:
            df: OHLCV + 피처 데이터프레임.

        Returns:
            신호 시리즈 (1=매수, -1=매도, 0=중립).
        """
        df_feat = self.feature_engine.compute_all_features(df)

        X = df_feat[self.feature_names]
        valid_mask = ~X.isna().any(axis=1)

        signals = pd.Series(0, index=df.index, dtype=int)

        if valid_mask.sum() == 0:
            return signals

        X_valid = X[valid_mask]
        proba = self.model.predict_proba(X_valid)  # (n, 3)

        max_idx = np.argmax(proba, axis=1)
        max_prob = np.max(proba, axis=1)

        # confidence_threshold 이상인 경우만 시그널 생성
        confident = max_prob >= self.confidence_threshold
        signal_values = np.array([self._LABEL_MAP_INV[i] for i in max_idx])
        signal_values[~confident] = 0

        signals.loc[valid_mask] = signal_values

        return signals

    def _load_model(self) -> lgb.LGBMClassifier:
        """저장된 LightGBM 모델 로드.

        Returns:
            LGBMClassifier 인스턴스.

        Raises:
            FileNotFoundError: 모델 파일이 없는 경우.
        """
        model_path = self.config.get(
            "model_path",
            "strategies/lgbm_classifier/models/latest.txt",
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_path}\n"
                f"먼저 train_lgbm.py로 모델을 학습하세요."
            )

        booster = lgb.Booster(model_file=model_path)
        model = lgb.LGBMClassifier()
        model._Booster = booster
        model._n_classes = 3
        model.fitted_ = True

        return model

    def _load_feature_names(self) -> list[str]:
        """피처 이름 목록 로드.

        Returns:
            피처 이름 리스트.

        Raises:
            FileNotFoundError: 피처 이름 파일이 없는 경우.
        """
        path = self.config.get(
            "feature_names_path",
            "strategies/lgbm_classifier/models/feature_names.json",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"피처 이름 파일을 찾을 수 없습니다: {path}\n"
                f"먼저 train_lgbm.py로 모델을 학습하세요."
            )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
