"""Bybit 거래소 데이터 수집 모듈.

REST API를 통한 과거 데이터 수집과 WebSocket을 통한 실시간 스트리밍을 담당한다.
"""

import os
import time
from datetime import datetime
from typing import Optional

import ccxt
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("collector")


class BybitDataCollector:
    """Bybit 거래소 데이터 수집기.

    ccxt 라이브러리를 통해 Bybit API에 접근하며,
    OHLCV, 펀딩비, 미결제약정 등의 데이터를 수집한다.
    """

    OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        testnet: bool = False,
    ) -> None:
        """BybitDataCollector 초기화.

        Args:
            api_key: Bybit API 키. None이면 환경변수에서 로드.
            secret: Bybit API 시크릿. None이면 환경변수에서 로드.
            testnet: True이면 Bybit testnet (sandbox) 모드 사용.
        """
        if testnet:
            key = api_key or os.getenv("BYBIT_TESTNET_API_KEY")
            sec = secret or os.getenv("BYBIT_TESTNET_SECRET")
        else:
            key = api_key or os.getenv("BYBIT_API_KEY")
            sec = secret or os.getenv("BYBIT_SECRET")

        self.exchange = ccxt.bybit({
            "apiKey": key,
            "secret": sec,
            "options": {"defaultType": "linear"},
            "enableRateLimit": True,
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("[TESTNET] BybitDataCollector 초기화 완료 (sandbox)")
        else:
            logger.info("BybitDataCollector 초기화 완료")

    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """OHLCV(시가/고가/저가/종가/거래량) 데이터 수집.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT:USDT").
            timeframe: 캔들 타임프레임 (예: "1m", "5m", "1h", "1d").
            since: 시작 시점 ISO 8601 문자열 (예: "2024-01-01T00:00:00Z").
            limit: 1회 요청 최대 캔들 수 (최대 1000).

        Returns:
            OHLCV 데이터프레임 (columns: timestamp, open, high, low, close, volume).
        """
        since_ts = self.exchange.parse8601(since) if since else None
        raw = self.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ts,
            limit=limit,
        )
        df = pd.DataFrame(raw, columns=self.OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        logger.info(f"OHLCV 수집 완료: {symbol} {timeframe} ({len(df)}건)")
        return df

    def fetch_ohlcv_bulk(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        since: str = "2024-01-01T00:00:00Z",
        sleep_sec: float = 0.2,
    ) -> pd.DataFrame:
        """대량 과거 OHLCV 데이터를 페이지네이션으로 수집.

        Args:
            symbol: 거래 심볼.
            timeframe: 캔들 타임프레임.
            since: 수집 시작 시점 ISO 8601 문자열.
            sleep_sec: 요청 간 대기 시간 (초). Rate limit 준수용.

        Returns:
            전체 기간의 OHLCV 데이터프레임.
        """
        all_data: list[pd.DataFrame] = []
        since_ts = self.exchange.parse8601(since)
        timeframe_ms = self._timeframe_to_ms(timeframe)

        while True:
            raw = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=1000,
            )
            if not raw:
                break

            df = pd.DataFrame(raw, columns=self.OHLCV_COLUMNS)
            all_data.append(df)

            since_ts = int(raw[-1][0]) + timeframe_ms
            if len(raw) < 1000:
                break

            time.sleep(sleep_sec)

        if not all_data:
            return pd.DataFrame(columns=self.OHLCV_COLUMNS)

        result = pd.concat(all_data, ignore_index=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        logger.info(f"OHLCV 대량 수집 완료: {symbol} {timeframe} ({len(result)}건)")
        return result

    def fetch_funding_rate(
        self,
        symbol: str = "BTC/USDT:USDT",
        since: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """펀딩비 이력 수집.

        Args:
            symbol: 거래 심볼.
            since: 시작 시점 ISO 8601 문자열.
            limit: 최대 건수.

        Returns:
            펀딩비 데이터프레임.
        """
        since_ts = self.exchange.parse8601(since) if since else None
        raw = self.exchange.fetch_funding_rate_history(
            symbol=symbol,
            since=since_ts,
            limit=limit,
        )
        records = [
            {
                "timestamp": r["timestamp"],
                "symbol": r["symbol"],
                "funding_rate": r["fundingRate"],
            }
            for r in raw
        ]
        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        logger.info(f"펀딩비 수집 완료: {symbol} ({len(df)}건)")
        return df

    def fetch_funding_rate_bulk(
        self,
        symbol: str = "BTC/USDT:USDT",
        since: str = "2024-01-01T00:00:00Z",
        sleep_sec: float = 0.3,
    ) -> pd.DataFrame:
        """대량 펀딩비 이력을 페이지네이션으로 수집.

        Bybit 펀딩비는 8시간 간격 (UTC 0:00, 8:00, 16:00).
        limit=100이면 약 33일분. 2024-01 ~ 현재 = ~2,400건 → 약 24회 호출.

        Args:
            symbol: 거래 심볼.
            since: 수집 시작 시점 ISO 8601 문자열.
            sleep_sec: 요청 간 대기 시간 (초). Rate limit 준수용.

        Returns:
            전체 기간의 펀딩비 데이터프레임 (timestamp, symbol, funding_rate).
        """
        all_data: list[pd.DataFrame] = []
        since_ts = self.exchange.parse8601(since)

        while True:
            raw = self.exchange.fetch_funding_rate_history(
                symbol=symbol,
                since=since_ts,
                limit=100,
            )
            if not raw:
                break

            records = [
                {
                    "timestamp": r["timestamp"],
                    "symbol": r["symbol"],
                    "funding_rate": r["fundingRate"],
                }
                for r in raw
            ]
            df = pd.DataFrame(records)
            all_data.append(df)

            since_ts = int(raw[-1]["timestamp"]) + 1
            if len(raw) < 100:
                break

            time.sleep(sleep_sec)

        if not all_data:
            return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

        result = pd.concat(all_data, ignore_index=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        logger.info(f"펀딩비 대량 수집 완료: {symbol} ({len(result)}건)")
        return result

    def fetch_open_interest_bulk(
        self,
        symbol: str = "BTC/USDT:USDT",
        timeframe: str = "1h",
        since: str = "2024-01-01T00:00:00Z",
        sleep_sec: float = 0.5,
    ) -> pd.DataFrame:
        """대량 미결제약정 이력을 페이지네이션으로 수집.

        ccxt의 fetch_open_interest_history()는 since를 무시하고 최신 200건만
        반환하므로, Bybit V5 API를 직접 호출하여 endTime 역순 페이지네이션.
        API는 역순(최신→과거) 반환하므로 endTime을 점점 과거로 이동.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT:USDT").
            timeframe: OI 간격 (1h 권장 — OHLCV와 동일 타임프레임).
            since: 수집 시작 시점 ISO 8601 문자열.
            sleep_sec: 요청 간 대기 시간 (초). Rate limit 준수용.

        Returns:
            미결제약정 데이터프레임 (timestamp, open_interest_amount).
        """
        all_data: list[pd.DataFrame] = []
        since_ts = self.exchange.parse8601(since)
        end_ts = int(datetime.utcnow().timestamp() * 1000)
        timeframe_ms = self._timeframe_to_ms(timeframe)
        # ccxt 심볼 → Bybit API 심볼 변환 (BTC/USDT:USDT → BTCUSDT)
        api_symbol = symbol.replace("/", "").replace(":USDT", "")

        # V5 intervalTime 매핑
        interval_map = {"5m": "5min", "15m": "15min", "30m": "30min",
                        "1h": "1h", "4h": "4h", "1d": "1D"}
        interval_time = interval_map.get(timeframe, timeframe)

        page_count = 0
        while end_ts > since_ts:
            resp = self.exchange.publicGetV5MarketOpenInterest({
                "category": "linear",
                "symbol": api_symbol,
                "intervalTime": interval_time,
                "limit": "200",
                "endTime": str(end_ts),
            })
            rows = resp.get("result", {}).get("list", [])
            if not rows:
                break

            records = [
                {
                    "timestamp": int(r["timestamp"]),
                    "open_interest_amount": float(r["openInterest"]),
                }
                for r in rows
            ]
            all_data.append(pd.DataFrame(records))

            # 역순이므로 마지막(가장 과거) row의 timestamp에서 1봉 전으로 이동
            oldest_ts = min(int(r["timestamp"]) for r in rows)
            end_ts = oldest_ts - timeframe_ms

            page_count += 1
            if page_count % 20 == 0:
                logger.info(f"OI 수집 진행: {page_count}페이지, "
                            f"현재 {datetime.utcfromtimestamp(oldest_ts/1000).strftime('%Y-%m-%d')}")

            if len(rows) < 200:
                break

            time.sleep(sleep_sec)

        if not all_data:
            return pd.DataFrame(columns=["timestamp", "open_interest_amount"])

        result = pd.concat(all_data, ignore_index=True)
        result["timestamp"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
        # since 이전 데이터 제거
        since_dt = pd.Timestamp(since, tz="UTC")
        result = result[result["timestamp"] >= since_dt]
        result = (
            result.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        logger.info(f"OI 대량 수집 완료: {symbol} {timeframe} ({len(result)}건, {page_count}페이지)")
        return result

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> list[str]:
        """OHLCV 데이터를 월별 Parquet 파일로 저장.

        Args:
            df: OHLCV 데이터프레임 (timestamp 컬럼이 datetime).
            symbol: 거래 심볼 (저장 시 슬래시/콜론 제거).
            timeframe: 타임프레임.

        Returns:
            저장된 파일 경로 목록.
        """
        clean_symbol = symbol.replace("/", "").replace(":", "")
        df["year_month"] = df["timestamp"].dt.strftime("%Y-%m")
        saved_paths: list[str] = []

        for ym, group in df.groupby("year_month"):
            path = f"data/raw/bybit/{clean_symbol}/{timeframe}/{ym}.parquet"
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if os.path.exists(path):
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, group.drop(columns=["year_month"])])
                combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            else:
                combined = group.drop(columns=["year_month"]).reset_index(drop=True)

            combined.to_parquet(path, index=False, compression="snappy")
            saved_paths.append(path)
            logger.info(f"저장 완료: {path} ({len(combined)}건)")

        return saved_paths

    def start_websocket(self, symbol: str = "BTC/USDT:USDT", callback=None) -> None:
        """WebSocket 실시간 데이터 스트리밍 시작.

        Args:
            symbol: 거래 심볼.
            callback: 새 데이터 수신 시 호출할 콜백 함수.
                      시그니처: callback(trade: dict) -> None

        Note:
            이 메서드는 블로킹이며, 별도 스레드에서 실행 권장.
        """
        logger.info(f"WebSocket 스트리밍 시작: {symbol}")
        while True:
            try:
                trades = self.exchange.watch_trades(symbol)
                for trade in trades:
                    if callback:
                        callback(trade)
            except Exception as e:
                logger.error(f"WebSocket 오류: {e}. 5초 후 재연결...")
                time.sleep(5)

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int:
        """타임프레임 문자열을 밀리초로 변환."""
        multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers[unit]
