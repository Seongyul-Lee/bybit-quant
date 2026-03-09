"""이동평균 크로스오버 전략 (v2).

Fast MA가 Slow MA를 상향 돌파하면 매수,
하향 돌파하면 매도 신호를 생성한다.
트렌드/RSI/거래량/ATR 필터를 통해 whipsaw를 억제한다.
"""

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy


class MACrossoverStrategy(BaseStrategy):
    """이동평균 크로스오버 전략.

    골든 크로스(Fast MA > Slow MA 상향 돌파) → 매수 신호
    데드 크로스(Fast MA < Slow MA 하향 돌파) → 매도 신호

    v2 필터:
        - 트렌드 필터 (MA200): 장기 추세 방향과 일치하는 신호만 허용
        - RSI 필터: 과매수/과매도 구간에서 역방향 신호 억제
        - 거래량 필터: 평균 거래량 대비 충분한 거래량 동반 시에만 유효
        - ATR 필터: 최소 변동성 미달 시 가짜 크로스로 판단

    Config 키:
        fast_period: 빠른 이동평균 기간 (기본 20).
        slow_period: 느린 이동평균 기간 (기본 50).
        trend_filter: 트렌드 필터 활성화 (기본 True).
        rsi_filter: RSI 필터 활성화 (기본 True).
        rsi_overbought: RSI 과매수 임계값 (기본 70).
        rsi_oversold: RSI 과매도 임계값 (기본 30).
        volume_filter: 거래량 필터 활성화 (기본 True).
        volume_multiplier: 거래량 배수 임계값 (기본 1.0).
        atr_filter: ATR 필터 활성화 (기본 True).
        atr_min_pct: ATR/close 최소 비율 (기본 0.003).
    """

    def generate_signal(self, df: pd.DataFrame) -> int:
        """이동평균 크로스오버 기반 매매 신호 생성 (필터 적용).

        Args:
            df: OHLCV 데이터프레임 (최소 컬럼: close).

        Returns:
            1  = 골든 크로스 (매수)
            -1 = 데드 크로스 (매도)
            0  = 크로스오버 없음 또는 필터에 의해 억제
        """
        fast_period: int = self.config.get("fast_period", 20)
        slow_period: int = self.config.get("slow_period", 50)

        if len(df) < slow_period + 1:
            return 0

        # MA 계산 (processor 피처 재활용 또는 fallback)
        ma_fast = self._get_or_compute_ma(df, fast_period)
        ma_slow = self._get_or_compute_ma(df, slow_period)

        # 직전 2개 시점 비교로 크로스오버 감지
        curr_fast = ma_fast.iloc[-1]
        curr_slow = ma_slow.iloc[-1]
        prev_fast = ma_fast.iloc[-2]
        prev_slow = ma_slow.iloc[-2]

        if pd.isna(curr_fast) or pd.isna(curr_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return 0

        # 크로스오버 감지
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            signal = 1
        elif curr_fast < curr_slow and prev_fast >= prev_slow:
            signal = -1
        else:
            return 0

        # 필터 적용
        close = df["close"].iloc[-1]

        # 1. 트렌드 필터: close > ma_200이면 롱만, close < ma_200이면 숏만
        if self.config.get("trend_filter", True):
            ma_200 = self._get_or_compute_ma(df, 200)
            ma_200_val = ma_200.iloc[-1]
            if not pd.isna(ma_200_val):
                if signal == 1 and close < ma_200_val:
                    return 0
                if signal == -1 and close > ma_200_val:
                    return 0

        # 2. RSI 필터: RSI > overbought일 때 롱 억제, RSI < oversold일 때 숏 억제
        if self.config.get("rsi_filter", True):
            rsi = self._get_or_compute_rsi(df)
            rsi_val = rsi.iloc[-1]
            overbought = self.config.get("rsi_overbought", 70)
            oversold = self.config.get("rsi_oversold", 30)
            if not pd.isna(rsi_val):
                if signal == 1 and rsi_val > overbought:
                    return 0
                if signal == -1 and rsi_val < oversold:
                    return 0

        # 3. 거래량 필터: volume > volume_ma_20 * multiplier
        if self.config.get("volume_filter", True) and "volume" in df.columns:
            vol_ma = self._get_or_compute_volume_ma(df)
            vol_ma_val = vol_ma.iloc[-1]
            multiplier = self.config.get("volume_multiplier", 1.0)
            if not pd.isna(vol_ma_val) and vol_ma_val > 0:
                if df["volume"].iloc[-1] < vol_ma_val * multiplier:
                    return 0

        # 4. ATR 필터: atr_14/close < atr_min_pct이면 가짜 크로스
        if self.config.get("atr_filter", True):
            atr = self._get_or_compute_atr(df)
            atr_val = atr.iloc[-1]
            atr_min_pct = self.config.get("atr_min_pct", 0.003)
            if not pd.isna(atr_val) and close > 0:
                if atr_val / close < atr_min_pct:
                    return 0

        return signal

    def generate_signals_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """벡터화된 신호 생성 (백테스트 전용).

        generate_signal과 동일한 로직을 벡터 연산으로 구현하여
        백테스트 속도를 O(n²) → O(n)으로 개선한다.

        Args:
            df: OHLCV + 피처 데이터프레임.

        Returns:
            신호 시리즈 (1=매수, -1=매도, 0=중립).
        """
        fast_period: int = self.config.get("fast_period", 20)
        slow_period: int = self.config.get("slow_period", 50)

        ma_fast = self._get_or_compute_ma(df, fast_period)
        ma_slow = self._get_or_compute_ma(df, slow_period)

        # 크로스오버 감지
        cross_up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
        cross_down = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[cross_up] = 1
        signals[cross_down] = -1

        close = df["close"]

        # 1. 트렌드 필터
        if self.config.get("trend_filter", True):
            ma_200 = self._get_or_compute_ma(df, 200)
            signals[(signals == 1) & (close < ma_200)] = 0
            signals[(signals == -1) & (close > ma_200)] = 0

        # 2. RSI 필터
        if self.config.get("rsi_filter", True):
            rsi = self._get_or_compute_rsi(df)
            overbought = self.config.get("rsi_overbought", 70)
            oversold = self.config.get("rsi_oversold", 30)
            signals[(signals == 1) & (rsi > overbought)] = 0
            signals[(signals == -1) & (rsi < oversold)] = 0

        # 3. 거래량 필터
        if self.config.get("volume_filter", True) and "volume" in df.columns:
            vol_ma = self._get_or_compute_volume_ma(df)
            multiplier = self.config.get("volume_multiplier", 1.0)
            signals[(signals != 0) & (df["volume"] < vol_ma * multiplier)] = 0

        # 4. ATR 필터
        if self.config.get("atr_filter", True):
            atr = self._get_or_compute_atr(df)
            atr_min_pct = self.config.get("atr_min_pct", 0.003)
            atr_pct = atr / close
            signals[(signals != 0) & (atr_pct < atr_min_pct)] = 0

        return signals

    def _get_or_compute_ma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """processor 피처가 있으면 재활용, 없으면 계산."""
        col = f"ma_{period}"
        if col in df.columns:
            return df[col]
        return df["close"].rolling(period).mean()

    def _get_or_compute_rsi(self, df: pd.DataFrame) -> pd.Series:
        """processor의 rsi_14가 있으면 재활용, 없으면 계산."""
        if "rsi_14" in df.columns:
            return df["rsi_14"]
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _get_or_compute_volume_ma(self, df: pd.DataFrame) -> pd.Series:
        """processor의 volume_ma_20이 있으면 재활용, 없으면 계산."""
        if "volume_ma_20" in df.columns:
            return df["volume_ma_20"]
        return df["volume"].rolling(20).mean()

    def _get_or_compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """processor의 atr_14가 있으면 재활용, 없으면 계산."""
        if "atr_14" in df.columns:
            return df["atr_14"]
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(14).mean()
