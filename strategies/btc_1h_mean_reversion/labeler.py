"""평균회귀 라벨링 모듈.

과매도 조건을 감지하고, 이후 평균으로 복귀하는지 여부로 2클래스 라벨을 생성한다.
라벨: 1(매수 = 평균 복귀 성공), 0(비매수 = 과매도 아님 or 추가 하락/타임아웃).
"""

import numpy as np
import pandas as pd


class MeanReversionLabeler:
    """평균회귀 방식으로 2클래스 라벨을 생성.

    1. 과매도 조건 감지 (bb_position < threshold OR rsi < threshold)
    2. 과매도인 봉에서만 Triple Barrier 라벨링 수행
    3. 과매도가 아닌 봉은 0(비매수)

    Attributes:
        oversold_bb_threshold: 볼린저밴드 position 과매도 기준.
        oversold_rsi_threshold: RSI 과매도 기준.
        profit_atr_mult: 상단 배리어 ATR 배수 (평균 복귀 목표).
        loss_atr_mult: 하단 배리어 ATR 배수 (추가 하락 손절).
        max_holding_period: 최대 보유 기간 (봉 수).
        oversold_mode: 과매도 조건 결합 방식 ("or" 또는 "and").
    """

    def __init__(
        self,
        oversold_bb_threshold: float = 0.2,
        oversold_rsi_threshold: float = 30.0,
        profit_atr_mult: float = 2.0,
        loss_atr_mult: float = 3.0,
        max_holding_period: int = 16,
        oversold_mode: str = "or",
    ) -> None:
        """MeanReversionLabeler 초기화.

        Args:
            oversold_bb_threshold: bb_position이 이 값 미만이면 과매도.
            oversold_rsi_threshold: RSI가 이 값 미만이면 과매도.
            profit_atr_mult: 상단 배리어 = close + profit_atr_mult * ATR_14.
            loss_atr_mult: 하단 배리어 = close - loss_atr_mult * ATR_14.
            max_holding_period: 배리어 미터치 시 최대 대기 봉 수.
            oversold_mode: "or"이면 하나만 충족해도 과매도, "and"이면 모두 충족.
        """
        self.oversold_bb_threshold = oversold_bb_threshold
        self.oversold_rsi_threshold = oversold_rsi_threshold
        self.profit_atr_mult = profit_atr_mult
        self.loss_atr_mult = loss_atr_mult
        self.max_holding_period = max_holding_period
        self.oversold_mode = oversold_mode

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """평균회귀 라벨을 생성.

        Args:
            df: OHLCV + 피처 데이터프레임.
                bb_position, rsi_14, atr_14 컬럼이 필요.

        Returns:
            라벨 시리즈 (1=매수, 0=비매수, NaN=라벨 불가).
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        if "atr_14" in df.columns:
            atr = df["atr_14"].values
        else:
            atr = self._compute_atr(df).values

        # 과매도 조건 감지
        oversold = self._detect_oversold(df)

        n = len(df)
        labels = np.full(n, np.nan)

        for i in range(n - self.max_holding_period):
            if np.isnan(atr[i]):
                continue

            # 과매도가 아니면 비매수
            if not oversold[i]:
                labels[i] = 0
                continue

            # 과매도일 때 Triple Barrier 적용
            upper_barrier = close[i] + self.profit_atr_mult * atr[i]
            lower_barrier = close[i] - self.loss_atr_mult * atr[i]

            label = 0  # 기본: 비매수 (추가 하락 또는 타임아웃)

            for j in range(i + 1, i + 1 + self.max_holding_period):
                hit_upper = high[j] >= upper_barrier
                hit_lower = low[j] <= lower_barrier

                if hit_upper and hit_lower:
                    if abs(high[j] - close[i]) <= abs(low[j] - close[i]):
                        label = 1
                    else:
                        label = 0
                    break
                elif hit_upper:
                    label = 1
                    break
                elif hit_lower:
                    label = 0
                    break

            labels[i] = label

        return pd.Series(labels, index=df.index, name="label")

    def _detect_oversold(self, df: pd.DataFrame) -> np.ndarray:
        """과매도 조건 감지.

        Args:
            df: 피처가 포함된 데이터프레임.

        Returns:
            과매도 여부 boolean 배열.
        """
        n = len(df)
        conditions = []

        if "bb_position" in df.columns:
            conditions.append(df["bb_position"].values < self.oversold_bb_threshold)

        if "rsi_14" in df.columns:
            conditions.append(df["rsi_14"].values < self.oversold_rsi_threshold)

        if not conditions:
            return np.ones(n, dtype=bool)

        if self.oversold_mode == "and":
            result = np.ones(n, dtype=bool)
            for cond in conditions:
                result &= cond
        else:  # "or"
            result = np.zeros(n, dtype=bool)
            for cond in conditions:
                result |= cond

        return result

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR 계산 (fallback)."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
