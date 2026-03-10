"""데이터 처리 및 피처 엔지니어링 모듈.

원본 OHLCV 데이터에 기술적 지표를 추가하고,
이상치 탐지 및 갭 처리를 수행한다.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("processor")


class DataProcessor:
    """OHLCV 데이터 처리 및 피처 엔지니어링 담당.

    이상치 탐지, 갭 처리, 기술적 지표 계산을 수행하고
    결과를 data/processed/에 Parquet으로 저장한다.
    """

    def __init__(self, spike_threshold: float = 0.1) -> None:
        """DataProcessor 초기화.

        Args:
            spike_threshold: 가격 스파이크 탐지 임계값 (기본 10%).
        """
        self.spike_threshold = spike_threshold

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV 데이터에 기술적 지표 피처를 추가.

        추가되는 지표:
        - 이동평균 (MA 10, 20, 50, 200)
        - RSI (14)
        - 볼린저밴드 (20, 2σ)
        - 거래량 이동평균 (20)
        - ATR (14)

        Args:
            df: OHLCV 데이터프레임.

        Returns:
            피처가 추가된 데이터프레임.
        """
        df = df.copy()

        # 이동평균
        for period in [10, 20, 50, 200]:
            df[f"ma_{period}"] = df["close"].rolling(period).mean()

        # RSI
        df["rsi_14"] = self._compute_rsi(df["close"], period=14)

        # 볼린저밴드
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std

        # 거래량 이동평균
        df["volume_ma_20"] = df["volume"].rolling(20).mean()

        # ATR
        df["atr_14"] = self._compute_atr(df, period=14)

        logger.info(f"피처 추가 완료: {len(df)}행, 컬럼 {list(df.columns)}")
        return df

    def detect_spike(self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        """가격 스파이크 이상치를 탐지해 플래그 컬럼 추가.

        직전 캔들 대비 threshold 이상 급등락하면 is_spike=True로 표시.
        원본 데이터는 제거하지 않고 플래그만 추가한다.

        Args:
            df: OHLCV 데이터프레임.
            threshold: 스파이크 판정 임계값. None이면 인스턴스 기본값 사용.

        Returns:
            is_spike 컬럼이 추가된 데이터프레임.
        """
        df = df.copy()
        th = threshold or self.spike_threshold
        pct_change = df["close"].pct_change().abs()
        df["is_spike"] = pct_change > th

        spike_count = df["is_spike"].sum()
        if spike_count > 0:
            logger.warning(f"스파이크 {spike_count}건 감지 (임계값: {th:.1%})")
        return df

    def fill_gaps(self, df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        """거래소 점검 등으로 발생한 시간 갭을 forward fill로 처리.

        예상 타임스탬프를 생성하고 누락 구간을 직전 종가로 채운다.

        Args:
            df: OHLCV 데이터프레임 (timestamp 컬럼 필수).
            timeframe_minutes: 타임프레임 분 단위 (예: 60 = 1시간).

        Returns:
            갭이 채워진 데이터프레임.
        """
        df = df.copy()
        if df.empty:
            return df

        expected_index = pd.date_range(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
            freq=f"{timeframe_minutes}min",
        )

        original_len = len(df)
        original_timestamps = set(df["timestamp"])
        df = df.set_index("timestamp").reindex(expected_index).ffill()
        df = df.reset_index().rename(columns={"index": "timestamp"})

        # gap으로 forward fill된 행을 표시
        df["is_gap_filled"] = ~df["timestamp"].isin(original_timestamps)

        gap_count = len(df) - original_len
        if gap_count > 0:
            logger.warning(f"갭 {gap_count}건 forward fill 처리됨")
        return df

    def process_and_save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        timeframe_minutes: int = 60,
    ) -> str:
        """전체 처리 파이프라인 실행 후 Parquet 저장.

        이상치 탐지 → 갭 처리 → 피처 추가 → 저장.

        Args:
            df: 원본 OHLCV 데이터프레임.
            symbol: 심볼 (예: "BTCUSDT").
            timeframe: 타임프레임 (예: "1h").
            timeframe_minutes: 타임프레임 분 단위.

        Returns:
            저장된 파일 경로.
        """
        df = self.detect_spike(df)
        df = self.fill_gaps(df, timeframe_minutes)
        df = self.add_features(df)

        path = f"data/processed/{symbol}_{timeframe}_features.parquet"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")
        logger.info(f"처리 데이터 저장 완료: {path}")
        return path

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """RSI(Relative Strength Index) 계산.

        Args:
            series: 종가 시리즈.
            period: RSI 기간.

        Returns:
            RSI 시리즈 (0~100).
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR(Average True Range) 계산.

        Args:
            df: OHLCV 데이터프레임 (high, low, close 필요).
            period: ATR 기간.

        Returns:
            ATR 시리즈.
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
