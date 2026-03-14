"""현물 매수/매도 실행기.

기존 OrderExecutor(선물 전용)와 분리.
ccxt의 defaultType을 "spot"으로 초기화.
"""

import os
from typing import Optional

import ccxt

from src.utils.logger import setup_logger

logger = setup_logger("spot_executor")


class SpotExecutor:
    """Bybit 현물 주문 실행기.

    펀딩비 차익거래의 현물 레그(leg) 담당.

    Attributes:
        exchange: ccxt.bybit 인스턴스 (defaultType="spot").
    """

    def __init__(self, testnet: bool = False) -> None:
        """SpotExecutor 초기화.

        기존 BybitDataCollector와 동일한 환경변수 패턴 사용:
          BYBIT_API_KEY / BYBIT_SECRET (mainnet)
          BYBIT_TESTNET_API_KEY / BYBIT_TESTNET_SECRET (testnet)

        Args:
            testnet: True이면 sandbox 모드.
        """
        if testnet:
            key = os.getenv("BYBIT_TESTNET_API_KEY")
            sec = os.getenv("BYBIT_TESTNET_SECRET")
        else:
            key = os.getenv("BYBIT_API_KEY")
            sec = os.getenv("BYBIT_SECRET")

        self.exchange = ccxt.bybit({
            "apiKey": key,
            "secret": sec,
            "options": {"defaultType": "spot"},
            "enableRateLimit": True,
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)

        self._testnet = testnet
        logger.info(f"SpotExecutor 초기화 완료 (testnet={testnet})")

    def buy(
        self,
        symbol: str,
        amount: float,
        order_type: str = "market",
    ) -> Optional[dict]:
        """현물 매수.

        Args:
            symbol: "BTC/USDT" (현물 심볼, :USDT 없음).
            amount: 매수 수량 (BTC 단위).
            order_type: "market" 권장 (즉시 체결 보장).

        Returns:
            주문 결과 dict, 실패 시 None.
        """
        try:
            if order_type == "market":
                order = self.exchange.create_market_buy_order(symbol, amount)
            else:
                price = self.get_spot_price(symbol)
                order = self.exchange.create_limit_buy_order(symbol, amount, price)

            logger.info(
                f"현물 매수 완료: {symbol} {amount} @ "
                f"{order.get('average') or order.get('price', 'market')}"
            )
            return order

        except ccxt.BaseError as e:
            logger.error(f"현물 매수 실패: {symbol} {amount} — {e}")
            return None

    def sell(
        self,
        symbol: str,
        amount: float,
        order_type: str = "market",
    ) -> Optional[dict]:
        """현물 매도 (포지션 청산 시).

        Args:
            symbol: "BTC/USDT" (현물 심볼).
            amount: 매도 수량 (BTC 단위).
            order_type: "market" 권장.

        Returns:
            주문 결과 dict, 실패 시 None.
        """
        try:
            if order_type == "market":
                order = self.exchange.create_market_sell_order(symbol, amount)
            else:
                price = self.get_spot_price(symbol)
                order = self.exchange.create_limit_sell_order(symbol, amount, price)

            logger.info(
                f"현물 매도 완료: {symbol} {amount} @ "
                f"{order.get('average') or order.get('price', 'market')}"
            )
            return order

        except ccxt.BaseError as e:
            logger.error(f"현물 매도 실패: {symbol} {amount} — {e}")
            return None

    def get_balance(self, coin: str = "BTC") -> float:
        """현물 잔고 조회.

        Bybit UTA에서 현물 잔고 = wallet balance 중 해당 코인.

        Args:
            coin: 코인 심볼 (예: "BTC", "ETH", "USDT").

        Returns:
            사용 가능한(free) 잔고. 조회 실패 시 0.0.
        """
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get(coin, {}).get("free", 0))
        except ccxt.BaseError as e:
            logger.error(f"잔고 조회 실패 ({coin}): {e}")
            return 0.0

    def get_spot_price(self, symbol: str = "BTC/USDT") -> float:
        """현물 현재가 조회.

        Args:
            symbol: 현물 심볼 (예: "BTC/USDT").

        Returns:
            현재 가격. 조회 실패 시 0.0.
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except ccxt.BaseError as e:
            logger.error(f"현물 가격 조회 실패 ({symbol}): {e}")
            return 0.0
