"""LightGBM 전략용 피처 엔지니어링 모듈.

약 50개 피처를 계산한다:
- 기술적 지표 (MA, RSI, MACD, BB, ATR, ADX, Stochastic, OBV)
- 가격 파생 (수익률, 로그수익률, 변동폭)
- 거래량 파생 (변화율, 비율)
- 시간 피처 (cyclical encoding)
- 멀티타임프레임 (4h, 1d resample)
"""

import numpy as np
import pandas as pd


class FeatureEngine:
    """OHLCV 데이터에서 ML 피처를 계산하는 엔진.

    processor가 이미 계산한 컬럼(ma_10, rsi_14, atr_14 등)은
    _get_or_compute_* 패턴으로 재활용한다.

    Attributes:
        config: 피처 관련 설정 딕셔너리.
    """

    def __init__(self, config: dict) -> None:
        """FeatureEngine 초기화.

        Args:
            config: 피처 설정 (ma_periods, rsi_period 등).
        """
        self.config = config
        self._feature_names: list[str] = []

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 피처 그룹을 계산하여 데이터프레임에 추가.

        Args:
            df: OHLCV 데이터프레임 (timestamp, open, high, low, close, volume).

        Returns:
            피처가 추가된 데이터프레임 (원본 컬럼 + 피처 컬럼).
        """
        df = df.copy()

        df = self._add_technical_indicators(df)
        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_time_features(df)
        df = self._add_multitimeframe_features(df)

        self._feature_names = [
            c for c in df.columns
            if c not in ("timestamp", "open", "high", "low", "close", "volume",
                         "is_spike", "label")
        ]

        return df

    def get_feature_names(self) -> list[str]:
        """계산된 피처 이름 목록 반환.

        Returns:
            피처 컬럼 이름 리스트.
        """
        return self._feature_names

    # ── 기술적 지표 ──────────────────────────────────────────

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """MA, RSI, MACD, BB, ATR, ADX, Stochastic, OBV 계산."""
        # 이동평균
        for period in [10, 30, 50, 200]:
            df[f"ma_{period}"] = self._get_or_compute_ma(df, period)

        # MA 비율 (가격 대비)
        for period in [10, 30, 50, 200]:
            df[f"ma_{period}_ratio"] = df["close"] / df[f"ma_{period}"]

        # RSI
        df["rsi_14"] = self._get_or_compute_rsi(df)

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 볼린저밴드
        bb_mid = self._get_or_compute_ma(df, 20)
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR
        df["atr_14"] = self._get_or_compute_atr(df)
        df["atr_14_pct"] = df["atr_14"] / df["close"]

        # ADX (14)
        df["adx_14"] = self._compute_adx(df, 14)

        # Stochastic (14, 3)
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # OBV
        obv_sign = np.sign(df["close"].diff())
        df["obv"] = (obv_sign * df["volume"]).cumsum()
        df["obv_ma_20"] = df["obv"].rolling(20).mean()

        return df

    # ── 가격 파생 피처 ───────────────────────────────────────

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """수익률, 로그수익률, 변동폭 등 가격 파생 피처 계산."""
        # 수익률
        for period in [1, 5, 10, 20]:
            df[f"return_{period}"] = df["close"].pct_change(period)

        # 로그수익률
        df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

        # 변동폭 비율
        df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]

        # 캔들 바디 비율
        df["body_ratio"] = (df["close"] - df["open"]) / (df["high"] - df["low"]).replace(0, np.nan)

        return df

    # ── 거래량 파생 피처 ─────────────────────────────────────

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 변화율, 비율 등 계산."""
        df["volume_ma_20"] = self._get_or_compute_volume_ma(df)
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

        return df

    # ── 시간 피처 ────────────────────────────────────────────

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 cyclical encoding (hour, dayofweek)."""
        if "timestamp" not in df.columns:
            return df

        ts = pd.to_datetime(df["timestamp"])
        hour = ts.dt.hour
        dow = ts.dt.dayofweek

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        return df

    # ── 멀티타임프레임 피처 ──────────────────────────────────

    def _add_multitimeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """1h → 4h/1d resample하여 RSI, MA(50), ATR 피처 생성.

        미래 정보 유출 방지: label='right', closed='right' 사용.
        """
        if "timestamp" not in df.columns:
            return df

        df_ts = df.set_index(pd.to_datetime(df["timestamp"]))

        for tf_label, rule in [("4h", "4h"), ("1d", "1D")]:
            ohlcv_resampled = df_ts.resample(rule, label="right", closed="right").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

            if len(ohlcv_resampled) < 50:
                continue

            # RSI
            rsi = self._compute_rsi_series(ohlcv_resampled["close"], 14)
            rsi = rsi.reindex(df_ts.index, method="ffill")
            df[f"rsi_14_{tf_label}"] = rsi.values

            # MA(50)
            ma50 = ohlcv_resampled["close"].rolling(50).mean()
            ma50 = ma50.reindex(df_ts.index, method="ffill")
            df[f"ma_50_{tf_label}"] = ma50.values
            df[f"ma_50_{tf_label}_ratio"] = df["close"] / df[f"ma_50_{tf_label}"]

            # ATR
            atr = self._compute_atr_series(ohlcv_resampled, 14)
            atr = atr.reindex(df_ts.index, method="ffill")
            df[f"atr_14_{tf_label}"] = atr.values

        return df

    # ── 헬퍼 메서드 ──────────────────────────────────────────

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
        return self._compute_rsi_series(df["close"], 14)

    def _get_or_compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """processor의 atr_14가 있으면 재활용, 없으면 계산."""
        if "atr_14" in df.columns:
            return df["atr_14"]
        return self._compute_atr_series(df, 14)

    def _get_or_compute_volume_ma(self, df: pd.DataFrame) -> pd.Series:
        """processor의 volume_ma_20이 있으면 재활용, 없으면 계산."""
        if "volume_ma_20" in df.columns:
            return df["volume_ma_20"]
        return df["volume"].rolling(20).mean()

    @staticmethod
    def _compute_rsi_series(series: pd.Series, period: int) -> pd.Series:
        """RSI 계산."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr_series(df: pd.DataFrame, period: int) -> pd.Series:
        """ATR 계산."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX(Average Directional Index) 계산."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = FeatureEngine._compute_atr_series(df, period)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(period).mean()

        return adx
