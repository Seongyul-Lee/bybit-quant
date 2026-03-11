"""LightGBM 전략용 피처 엔지니어링 모듈.

약 65개 피처를 계산한다:
- 기술적 지표 (MA, RSI, MACD, BB, ATR, ADX, Stochastic, OBV)
- 가격 파생 (수익률, 로그수익률, 변동폭)
- 거래량 파생 (변화율, 비율)
- 변동성 구조 (ATR 비율, skewness, kurtosis, Parkinson)
- 거래량 미세구조 (VWAP, 가격-거래량 상관)
- 시간 피처 (cyclical encoding)
- 멀티타임프레임 (4h, 1d resample — RSI, MA, ATR, ADX, BB)
"""

import os

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
                    symbol: 심볼 (예: "BTCUSDT"). 펀딩비 경로 결정에 사용.
        """
        self.config = config
        self._feature_names: list[str] = []
        self.symbol: str = config.get("symbol", "BTCUSDT")

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
        df = self._add_volatility_structure_features(df)
        df = self._add_volume_microstructure_features(df)
        df = self._add_time_features(df)
        df = self._add_multitimeframe_features(df)
        df = self._add_funding_features(df)

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

    def get_selected_features(self) -> list[str]:
        """과적합 방지를 위해 선별된 피처만 반환.

        47개 전체 피처에서 중복·가격수준의존 피처를 제거하고
        정규화된 ~18개 피처만 선택한다.

        Returns:
            선별된 피처 컬럼 이름 리스트.
        """
        selected = [
            # 정규화된 MA 비율 (3개 — 원본 모델 기준)
            "ma_10_ratio", "ma_30_ratio", "ma_200_ratio",
            # 모멘텀 (3개)
            "rsi_14", "macd_hist", "stoch_k",
            # 변동성 (3개)
            "atr_14_pct", "bb_width", "bb_position",
            # 추세 (1개)
            "adx_14",
            # 가격 파생 (1개)
            "return_1",
            # 거래량 (1개)
            "volume_ratio",
            # 시간 (2개)
            "hour_sin", "hour_cos",
            # 멀티타임프레임 (2개)
            "rsi_14_1d", "ma_50_1d_ratio",
            # 변동성 구조 (2개)
            "vol_ratio_7_30", "return_skew_20",
            # 펀딩비 (1개 — 극단값 감지)
            "funding_rate_zscore",
        ]
        return [f for f in selected if f in self._feature_names]

    @staticmethod
    def remove_correlated_features(
        X: pd.DataFrame,
        feature_names: list[str],
        threshold: float = 0.9,
    ) -> list[str]:
        """상관관계가 높은 피처를 자동 제거.

        피처 쌍의 Pearson 상관계수가 threshold 이상이면
        중요도가 낮은 쪽(뒤에 나오는 쪽)을 제거한다.

        Args:
            X: 피처 데이터프레임.
            feature_names: 대상 피처 이름 목록.
            threshold: 상관계수 제거 기준 (기본 0.9).

        Returns:
            상관관계 필터링 후 남은 피처 이름 리스트.
        """
        corr = X[feature_names].corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape, dtype=bool), k=1)
        )
        to_drop = set()
        for col in upper.columns:
            if any(upper[col] > threshold):
                to_drop.add(col)
        remaining = [f for f in feature_names if f not in to_drop]
        return remaining

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
        bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range
        df["bb_position"] = df["bb_position"].fillna(0.5)  # 밴드폭 0이면 중앙

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

    # ── 변동성 구조 피처 ────────────────────────────────────

    def _add_volatility_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR 비율, 수익률 skewness/kurtosis, Parkinson 변동성 등 계산."""
        # ATR(7) / ATR(30) — 변동성 확장/축소
        atr_7 = self._compute_atr_series(df, 7)
        atr_30 = self._compute_atr_series(df, 30)
        df["vol_ratio_7_30"] = atr_7 / atr_30.replace(0, np.nan)

        # 수익률 skewness / kurtosis (rolling 20봉)
        ret = df["close"].pct_change()
        df["return_skew_20"] = ret.rolling(20).skew()
        df["return_kurt_20"] = ret.rolling(20).kurt()

        # Parkinson volatility (high-low 기반, rolling 20봉)
        log_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
        df["parkinson_vol_20"] = np.sqrt(
            (log_hl ** 2).rolling(20).mean() / (4 * np.log(2))
        )

        # 상단/하단 그림자 비율 (매수/매도 압력 프록시)
        atr = self._get_or_compute_atr(df).replace(0, np.nan)
        df["upper_shadow_ratio"] = (df["high"] - df[["close", "open"]].max(axis=1)) / atr
        df["lower_shadow_ratio"] = (df[["close", "open"]].min(axis=1) - df["low"]) / atr

        return df

    # ── 거래량 미세구조 피처 ──────────────────────────────

    def _add_volume_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """VWAP 편차, 거래량 skewness, 가격-거래량 상관관계 계산."""
        # VWAP(20) 대비 현재가 편차 (ATR 정규화)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumvol = df["volume"].rolling(20).sum().replace(0, np.nan)
        vwap = (typical_price * df["volume"]).rolling(20).sum() / cumvol
        atr = self._get_or_compute_atr(df).replace(0, np.nan)
        df["vwap_deviation"] = (df["close"] - vwap) / atr

        # 거래량 skewness (rolling 20봉)
        df["volume_skew_20"] = df["volume"].rolling(20).skew()

        # 가격-거래량 상관관계 (rolling 20봉)
        ret = df["close"].pct_change()
        df["price_volume_corr_20"] = ret.rolling(20).corr(df["volume"])

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

        ts_index = pd.to_datetime(df["timestamp"])
        df_ts = df.set_index(ts_index)
        ohlcv_cols = df[["open", "high", "low", "close", "volume"]].copy()
        ohlcv_cols.index = ts_index

        for tf_label, rule in [("4h", "4h"), ("1d", "1D")]:
            ohlcv_resampled = ohlcv_cols.resample(rule, label="right", closed="right").agg({
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

            # ADX
            adx = self._compute_adx(ohlcv_resampled, 14)
            adx = adx.reindex(df_ts.index, method="ffill")
            df[f"adx_14_{tf_label}"] = adx.values

            # BB position
            bb_mid = ohlcv_resampled["close"].rolling(20).mean()
            bb_std = ohlcv_resampled["close"].rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_range = (bb_upper - bb_lower).replace(0, np.nan)
            bb_pos = (ohlcv_resampled["close"] - bb_lower) / bb_range
            bb_pos = bb_pos.fillna(0.5)
            bb_pos = bb_pos.reindex(df_ts.index, method="ffill")
            df[f"bb_position_{tf_label}"] = bb_pos.values

        # ATR 비율: 1h ATR / 1d ATR (크로스타임프레임 변동성 비교)
        if "atr_14_1d" in df.columns:
            atr_1h = self._get_or_compute_atr(df)
            df["atr_ratio_1h_1d"] = atr_1h / df["atr_14_1d"].replace(0, np.nan)

        return df

    # ── 펀딩비 피처 ──────────────────────────────────────────

    def _add_funding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """펀딩비 파생 피처 생성.

        Bybit 펀딩비는 8시간 간격 (UTC 0:00, 8:00, 16:00).
        1h 봉에 merge할 때 merge_asof(direction='backward')로
        현재 시점 이전의 가장 최근 펀딩비만 사용 (look-ahead bias 방지).

        Args:
            df: OHLCV + 기존 피처가 추가된 데이터프레임.

        Returns:
            펀딩비 피처가 추가된 데이터프레임.
        """
        # 심볼에서 펀딩비 파일 경로 생성 (BTCUSDT → BTCUSDTUSDT)
        clean_symbol = self.symbol.replace("/", "").replace(":", "")
        if not clean_symbol.endswith("USDT"):
            clean_symbol = clean_symbol + "USDT"
        elif clean_symbol.count("USDT") == 1:
            clean_symbol = clean_symbol + "USDT"
        funding_path = f"data/raw/bybit/{clean_symbol}/funding_rate.parquet"
        if not os.path.exists(funding_path):
            return df

        funding = pd.read_parquet(funding_path)
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
        funding = funding.sort_values("timestamp").reset_index(drop=True)

        # merge_asof: 각 1h 봉에 대해 해당 시점 이전의 가장 최근 펀딩비를 매칭
        df_ts = pd.to_datetime(df["timestamp"], utc=True)
        merge_df = pd.DataFrame({"timestamp": df_ts}).reset_index()
        # timestamp 해상도 통일 (us/ms 불일치 방지)
        merge_df["timestamp"] = merge_df["timestamp"].astype("datetime64[ns, UTC]")
        funding_merge = funding[["timestamp", "funding_rate"]].rename(
            columns={"funding_rate": "_fr"}
        ).copy()
        funding_merge["timestamp"] = funding_merge["timestamp"].astype("datetime64[ns, UTC]")
        merged = pd.merge_asof(
            merge_df.sort_values("timestamp"),
            funding_merge,
            on="timestamp",
            direction="backward",
        ).sort_values("index").set_index("index")

        fr = merged["_fr"]
        fr.index = df.index

        # 기본 피처: 현재 펀딩비
        df["funding_rate"] = fr

        # 이동평균 (8회 = ~2.7일, 24회 = ~8일)
        # 펀딩비는 8h 간격이지만 1h봉에서는 같은 값이 8개씩 반복되므로
        # rolling은 1h 봉 기준으로 적용 (8봉 = 1회 펀딩비, 64봉 = 8회)
        df["funding_rate_ma_8"] = fr.rolling(64, min_periods=8).mean()
        df["funding_rate_ma_24"] = fr.rolling(192, min_periods=24).mean()

        # z-score: (현재 - MA_24) / std_24
        fr_std = fr.rolling(192, min_periods=24).std().replace(0, np.nan)
        df["funding_rate_zscore"] = (fr - df["funding_rate_ma_24"]) / fr_std

        # 펀딩비 변화: 현재 vs 이전 (8봉 전 = 이전 펀딩비 정산)
        df["funding_rate_change"] = fr - fr.shift(8)

        # 연속 양수/음수 횟수 (펀딩비 정산 단위, 부호 변경 시 리셋)
        # 펀딩비가 실제로 변하는 시점에서만 카운트 (8h 간격)
        fr_changed = fr != fr.shift(1)
        fr_unique = fr[fr_changed]  # 실제 펀딩비 변경 시점만 추출
        sign_unique = np.sign(fr_unique)
        sign_change = sign_unique != sign_unique.shift(1)
        streak_groups = sign_change.cumsum()
        streak_unique = sign_unique.groupby(streak_groups).cumcount() + 1
        streak_unique = streak_unique * sign_unique
        # 전체 1h 봉으로 forward fill
        df["funding_rate_streak"] = streak_unique.reindex(df.index).ffill()

        return df

    # ── 헬퍼 메서드 ──────────────────────────────────────────

    def _get_or_compute_ma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """processor 피처가 있으면 재활용, 없으면 계산."""
        col = f"ma_{period}"
        if col in df.columns:
            return df[col]
        return df["close"].rolling(period).mean()

    def _get_or_compute_rsi(self, df: pd.DataFrame) -> pd.Series:
        """processor의 rsi_14가 있으면 재활용, NaN이 과다하면 재계산."""
        if "rsi_14" in df.columns:
            nan_ratio = df["rsi_14"].isna().mean()
            if nan_ratio < 0.05:
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
        """RSI 계산.

        loss=0 (하락 없음)이면 RSI=100, gain과 loss 모두 0이면 RSI=50.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # loss=0 & gain>0 → RSI=100, 둘 다 0 → RSI=50
        both_zero = (gain == 0) & (loss == 0)
        only_gain = (gain > 0) & (loss == 0)
        rsi = rsi.fillna(rsi)  # keep NaN from rolling warmup
        rsi.loc[only_gain] = 100.0
        rsi.loc[both_zero] = 50.0
        return rsi

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
        """ADX(Average Directional Index) 계산.

        DI 합이 0인 구간(gap fill 등)에서는 DX=0으로 처리.
        """
        high = df["high"]
        low = df["low"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = FeatureEngine._compute_atr_series(df, period)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
        dx = dx.fillna(0.0)  # DI 합이 0이면 방향성 없음 → DX=0
        adx = dx.rolling(period).mean()

        return adx
