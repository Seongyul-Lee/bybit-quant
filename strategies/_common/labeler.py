"""라벨링 모듈.

Triple Barrier (2클래스 분류)와 Forward Return (회귀) 라벨러를 제공한다.
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


class ShortTripleBarrierLabeler:
    """Triple Barrier 방식으로 숏 전용 2클래스 라벨을 생성.

    각 봉에서 ATR 기반 상한/하한 배리어를 설정하고,
    max_holding_period 내에 어느 배리어를 먼저 터치하는지에 따라
    라벨을 결정한다. 하단 배리어 터치 → 1(매도 기회), 그 외 → 0(비매도).

    기존 TripleBarrierLabeler와 상단/하단 판정만 반전.

    Attributes:
        upper_multiplier: 상단 배리어 ATR 배수 (숏의 SL — 가격 상승).
        lower_multiplier: 하단 배리어 ATR 배수 (숏의 TP — 가격 하락).
        max_holding_period: 최대 보유 기간 (봉 수).
    """

    def __init__(
        self,
        upper_multiplier: float = 3.0,
        lower_multiplier: float = 3.0,
        max_holding_period: int = 24,
    ) -> None:
        """ShortTripleBarrierLabeler 초기화.

        Args:
            upper_multiplier: 상단 배리어 = close + upper_multiplier * ATR_14.
                숏에서는 SL 방향 (가격 상승 = 손실).
            lower_multiplier: 하단 배리어 = close - lower_multiplier * ATR_14.
                숏에서는 TP 방향 (가격 하락 = 이익).
            max_holding_period: 배리어 미터치 시 최대 대기 봉 수.
        """
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.max_holding_period = max_holding_period

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """숏 라벨 생성.

        로직 (TripleBarrierLabeler와 비교):
        - 기존 롱: hit_upper → 1(매수), hit_lower → 0
        - 숏:      hit_lower → 1(매도 기회), hit_upper → 0(숏 실패)
        - 동시 터치: 하단이 가까우면 1, 상단이 가까우면 0
        - 타임아웃: 0 (비매도)

        Args:
            df: OHLCV 데이터프레임. atr_14 컬럼이 필요하며,
                없으면 자체 계산한다.

        Returns:
            라벨 시리즈 (1=매도 기회, 0=비매도, NaN=라벨 불가).
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

            label = 0  # 기본: 비매도 (상단 배리어 터치 또는 타임아웃)

            for j in range(i + 1, i + 1 + self.max_holding_period):
                hit_upper = high[j] >= upper_barrier
                hit_lower = low[j] <= lower_barrier

                if hit_upper and hit_lower:
                    # 동시 터치: 하단이 가까우면 매도(숏 성공), 상단이 가까우면 비매도
                    if abs(low[j] - close[i]) <= abs(high[j] - close[i]):
                        label = 1
                    else:
                        label = 0
                    break
                elif hit_lower:
                    label = 1  # 하단 터치 = 가격 하락 = 숏 성공
                    break
                elif hit_upper:
                    label = 0  # 상단 터치 = 가격 상승 = 숏 실패
                    break

            labels[i] = label

        return pd.Series(labels, index=df.index, name="label")

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR 계산 (fallback)."""
        return _compute_atr(df, period)


class ForwardReturnLabeler:
    """미래 N봉 수익률을 연속값으로 라벨링.

    각 봉에서 미래 forward_period 봉 동안의 수익률을 계산하되,
    ATR 기반 배리어로 클리핑하여 극단값을 제한한다.

    Attributes:
        forward_period: 미래 수익률 계산 기간 (봉 수).
        barrier_atr_mult: 수익률 클리핑 ATR 배수.
        use_log_return: True면 로그 수익률 사용.
    """

    def __init__(
        self,
        forward_period: int = 24,
        barrier_atr_mult: float = 3.0,
        use_log_return: bool = False,
    ) -> None:
        """ForwardReturnLabeler 초기화.

        Args:
            forward_period: 미래 수익률 계산 기간 (봉 수). 기본 24.
            barrier_atr_mult: 수익률 클리핑 ATR 배수. 기본 3.0.
            use_log_return: True면 로그 수익률. 기본 False.
        """
        self.forward_period = forward_period
        self.barrier_atr_mult = barrier_atr_mult
        self.use_log_return = use_log_return

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """미래 수익률 라벨 생성.

        로직:
        1. future_return = (close[t+forward_period] - close[t]) / close[t]
           (use_log_return이면 log(close[t+forward_period] / close[t]))
        2. atr_pct = atr_14 / close
        3. clip_bound = atr_pct * barrier_atr_mult
        4. future_return = future_return.clip(-clip_bound, +clip_bound)
        5. 마지막 forward_period 봉은 NaN

        Args:
            df: OHLCV + atr_14 데이터프레임.

        Returns:
            연속값 라벨 시리즈 (이름: "label").
        """
        close = df["close"]
        future_close = close.shift(-self.forward_period)

        if self.use_log_return:
            future_return = np.log(future_close / close)
        else:
            future_return = (future_close - close) / close

        # ATR 기반 클리핑
        if "atr_14" in df.columns:
            atr = df["atr_14"]
        else:
            atr = _compute_atr(df)

        atr_pct = atr / close
        clip_bound = atr_pct * self.barrier_atr_mult
        future_return = future_return.clip(lower=-clip_bound, upper=clip_bound)

        return pd.Series(future_return.values, index=df.index, name="label")
