"""라벨링 모듈.

Triple Barrier (2클래스 분류) 라벨러를 제공한다.
"""

import numpy as np
import pandas as pd


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR 계산 (공통 유틸).

    Args:
        df: OHLCV 데이터프레임.
        period: ATR 기간.

    Returns:
        ATR 시리즈.
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


class TripleBarrierLabeler:
    """Triple Barrier 방식으로 2클래스 라벨을 생성.

    각 봉에서 ATR 기반 상한/하한 배리어를 설정하고,
    max_holding_period 내에 어느 배리어를 먼저 터치하는지에 따라
    라벨을 결정한다. 상단 배리어 터치 → 1(매수), 그 외 → 0(비매수).

    Attributes:
        upper_multiplier: 상단 배리어 ATR 배수.
        lower_multiplier: 하단 배리어 ATR 배수.
        max_holding_period: 최대 보유 기간 (봉 수).
    """

    def __init__(
        self,
        upper_multiplier: float = 1.5,
        lower_multiplier: float = 1.5,
        max_holding_period: int = 24,
    ) -> None:
        """TripleBarrierLabeler 초기화.

        Args:
            upper_multiplier: 상단 배리어 = close + upper_multiplier * ATR_14. 대칭 권장.
            lower_multiplier: 하단 배리어 = close - lower_multiplier * ATR_14. 대칭 권장.
            max_holding_period: 배리어 미터치 시 최대 대기 봉 수.
        """
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.max_holding_period = max_holding_period

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """Triple Barrier 라벨을 생성.

        Args:
            df: OHLCV 데이터프레임. atr_14 컬럼이 필요하며,
                없으면 자체 계산한다.

        Returns:
            라벨 시리즈 (1=매수, 0=비매수, NaN=라벨 불가).
            마지막 max_holding_period 봉은 NaN.
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        if "atr_14" in df.columns:
            atr = df["atr_14"].values
        else:
            atr = self._compute_atr(df).values

        n = len(df)
        labels = np.full(n, np.nan)

        for i in range(n - self.max_holding_period):
            if np.isnan(atr[i]):
                continue

            upper_barrier = close[i] + self.upper_multiplier * atr[i]
            lower_barrier = close[i] - self.lower_multiplier * atr[i]

            label = 0  # 기본: 비매수 (하단 배리어 터치 또는 타임아웃)

            for j in range(i + 1, i + 1 + self.max_holding_period):
                hit_upper = high[j] >= upper_barrier
                hit_lower = low[j] <= lower_barrier

                if hit_upper and hit_lower:
                    # 동시 터치: 상단이 가까우면 매수, 하단이 가까우면 비매수
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

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR 계산 (fallback)."""
        return _compute_atr(df, period)


