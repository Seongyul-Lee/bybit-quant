"""현물+선물 동시 실행기 (펀딩비 차익거래 전용).

현물 매수와 선물 숏을 최대한 동시에 실행하여
델타 노출을 최소화.
"""

from datetime import datetime, timezone
from typing import Optional

from src.execution.executor import OrderExecutor
from src.execution.spot_executor import SpotExecutor
from src.utils.logger import setup_logger

logger = setup_logger("arb_executor")


class ArbExecutor:
    """펀딩비 차익거래 실행기.

    현물 매수 + 선물 숏을 동시 실행하고,
    델타(현물 보유량 - 선물 숏 수량)를 모니터링한다.

    Attributes:
        spot: SpotExecutor 인스턴스.
        perp: OrderExecutor 인스턴스 (기존).
    """

    def __init__(
        self,
        spot_executor: SpotExecutor,
        perp_executor: OrderExecutor,
    ) -> None:
        """ArbExecutor 초기화.

        Args:
            spot_executor: 현물 실행기.
            perp_executor: 선물 실행기 (기존 OrderExecutor).
        """
        self.spot = spot_executor
        self.perp = perp_executor
        logger.info("ArbExecutor 초기화 완료")

    def open_position(
        self,
        symbol_spot: str,
        symbol_perp: str,
        amount: float,
    ) -> dict:
        """차익거래 포지션 진입.

        실행 순서 (슬리피지 최소화):
        1. 선물 숏 먼저 (선물이 더 유동적)
        2. 현물 매수 직후
        3. 델타 검증

        선물을 먼저 실행하는 이유:
        - 선물이 현물보다 유동성이 높아 슬리피지 적음
        - 현물 매수 후 선물 숏이 실패하면 방향 노출 발생
        - 선물 숏 후 현물 매수가 실패하면 → 선물만 청산하면 됨

        Args:
            symbol_spot: 현물 심볼 (예: "BTC/USDT").
            symbol_perp: 선물 심볼 (예: "BTC/USDT:USDT").
            amount: 수량 (BTC 단위).

        Returns:
            진입 결과 딕셔너리.
        """
        result: dict = {
            "spot_order": None,
            "perp_order": None,
            "amount": amount,
            "delta": 0.0,
            "success": False,
            "entry_time": datetime.now(timezone.utc).isoformat(),
        }

        # Step 1: 선물 숏 (Hedge Mode: positionIdx=2 = Sell side)
        perp_order = self.perp.execute(
            symbol=symbol_perp,
            side="sell",
            amount=amount,
            order_type="market",
            strategy_name="funding_arb",
            signal_score=0,
            position_idx=2,
        )
        if not perp_order:
            logger.error("선물 숏 실패 — 진입 중단")
            return result
        result["perp_order"] = perp_order

        # Step 2: 현물 매수 (즉시)
        spot_order = self.spot.buy(
            symbol=symbol_spot,
            amount=amount,
            order_type="market",
        )
        if not spot_order:
            logger.error("현물 매수 실패 — 선물 숏 롤백")
            # 롤백: 실제 체결량 기준으로 선물 숏 청산 (Hedge Mode: positionIdx=2)
            actual_filled = float(perp_order.get("filled", 0))
            if actual_filled > 0:
                self.perp.execute(
                    symbol=symbol_perp,
                    side="buy",
                    amount=actual_filled,
                    order_type="market",
                    strategy_name="funding_arb_rollback",
                    signal_score=0,
                    position_idx=2,
                )
            else:
                logger.warning("선물 숏 체결 수량 0 — 롤백 불필요")
            return result
        result["spot_order"] = spot_order

        # Step 3: 델타 검증
        delta = self.get_delta(symbol_spot, symbol_perp)
        result["delta"] = delta
        result["success"] = True

        # Step 3 확장: 체결 가격 검증
        spot_fill = self._extract_fill_price(spot_order)
        perp_fill = self._extract_fill_price(perp_order)
        result["spot_fill_price"] = spot_fill
        result["perp_fill_price"] = perp_fill

        if spot_fill > 0 and perp_fill > 0:
            slippage_pct = (spot_fill - perp_fill) / spot_fill
            result["entry_slippage_pct"] = slippage_pct
            if abs(slippage_pct) > 0.005:
                logger.warning(
                    f"진입 슬리피지 경고: {slippage_pct:+.2%} "
                    f"(현물 {spot_fill:.2f} / 선물 {perp_fill:.2f})"
                )

        coin = symbol_spot.split("/")[0]
        logger.info(
            f"차익거래 진입 완료: {coin} 현물 {amount} + 선물 숏 {amount} "
            f"(delta={delta:.6f})"
        )
        return result

    def close_position(
        self,
        symbol_spot: str,
        symbol_perp: str,
        amount: float,
    ) -> dict:
        """차익거래 포지션 청산.

        실행 순서 (진입의 역순):
        1. 현물 매도
        2. 선물 숏 청산 (buy)

        Args:
            symbol_spot: 현물 심볼.
            symbol_perp: 선물 심볼.
            amount: 청산 수량.

        Returns:
            청산 결과 딕셔너리.
        """
        result: dict = {
            "spot_order": None,
            "perp_order": None,
            "amount": amount,
            "success": False,
            "close_time": datetime.now(timezone.utc).isoformat(),
        }

        # Step 1: 현물 매도
        spot_order = self.spot.sell(
            symbol=symbol_spot,
            amount=amount,
            order_type="market",
        )
        if not spot_order:
            logger.error("현물 매도 실패 — 수동 청산 필요")
            return result
        result["spot_order"] = spot_order

        # Step 2: 선물 숏 청산 (buy, Hedge Mode: positionIdx=2)
        perp_order = self.perp.execute(
            symbol=symbol_perp,
            side="buy",
            amount=amount,
            order_type="market",
            strategy_name="funding_arb_close",
            signal_score=0,
            position_idx=2,
        )
        if not perp_order:
            logger.error("선물 청산 실패 — 수동 청산 필요")
            return result
        result["perp_order"] = perp_order

        result["success"] = True
        coin = symbol_spot.split("/")[0]
        logger.info(f"차익거래 청산 완료: {coin} {amount}")
        return result

    def _extract_fill_price(self, order: dict) -> float:
        """주문에서 체결 가격 추출.

        ccxt 주문 결과에서 average(평균 체결가) 또는 price를 반환.

        Args:
            order: ccxt 주문 결과 딕셔너리.

        Returns:
            체결 가격. 추출 불가 시 0.0.
        """
        if not order:
            return 0.0
        avg = order.get("average")
        if avg and float(avg) > 0:
            return float(avg)
        price = order.get("price")
        if price and float(price) > 0:
            return float(price)
        return 0.0

    def get_delta(self, symbol_spot: str, symbol_perp: str) -> float:
        """현재 델타 = 현물 보유량 - 선물 숏 수량.

        이상적: 0 (완전 헤지)
        양수: 롱 노출 (현물 > 숏)
        음수: 숏 노출 (숏 > 현물)

        Args:
            symbol_spot: 현물 심볼 (예: "BTC/USDT").
            symbol_perp: 선물 심볼 (예: "BTC/USDT:USDT").

        Returns:
            현물 잔고 - 선물 숏 수량.
        """
        coin = symbol_spot.split("/")[0]
        spot_balance = self.spot.get_balance(coin)

        positions = self.perp.sync_positions()
        # Hedge Mode: positionIdx=2 키(symbol::2) 우선, 없으면 기본 키
        perp_key = f"{symbol_perp}::2"
        perp_pos = positions.get(perp_key, positions.get(symbol_perp, {}))
        perp_size = abs(float(perp_pos.get("size", 0)))

        return spot_balance - perp_size

    def get_funding_income(self, symbol_perp: str) -> Optional[float]:
        """최근 펀딩비 수취 금액 조회.

        Bybit V5 API의 거래 내역(transaction log)에서
        SETTLEMENT 유형을 필터링.

        Args:
            symbol_perp: 선물 심볼.

        Returns:
            최근 펀딩비 수취 금액. 조회 실패 시 None.
        """
        try:
            # ccxt의 fetch_funding_history 사용
            funding_history = self.perp.exchange.fetch_funding_history(
                symbol_perp, limit=3
            )
            if not funding_history:
                return None

            # 가장 최근 펀딩비
            latest = funding_history[0]
            amount = float(latest.get("amount", 0))
            logger.info(
                f"펀딩비 수취: {symbol_perp} {amount:+.6f} USDT"
            )
            return amount

        except Exception as e:
            logger.warning(f"펀딩비 조회 실패 ({symbol_perp}): {e}")
            return None
