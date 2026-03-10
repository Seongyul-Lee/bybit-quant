"""주문 실행 모듈.

거래소에 주문을 제출하고 체결을 확인하며,
모든 거래 내역을 기록한다.
"""

import json
import os
import tempfile
import shutil
import uuid
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("executor")

TRADE_LOG_PATH = "reports/trades/trade_log.csv"
STATE_PATH = "config/current_state.json"

TRADE_COLUMNS = [
    "trade_id", "timestamp", "strategy",
    "symbol", "side", "order_type",
    "price", "amount", "fee",
    "pnl", "cumulative_pnl",
    "signal_score", "notes",
]


class OrderExecutor:
    """주문 실행 및 관리 클래스.

    지정가 주문을 우선하며, 주문 멱등성을 보장하고,
    모든 거래 내역을 CSV에 append-only로 기록한다.

    Attributes:
        exchange: ccxt Exchange 인스턴스.
        pending_orders: 미체결 주문 추적 딕셔너리.
    """

    def __init__(self, exchange: ccxt.Exchange) -> None:
        """OrderExecutor 초기화.

        Args:
            exchange: 초기화된 ccxt Exchange 인스턴스.
        """
        self.exchange = exchange
        self.pending_orders: dict[str, dict] = {}
        self._cumulative_pnl = self._load_cumulative_pnl()
        logger.info("OrderExecutor 초기화 완료")

    @staticmethod
    def _load_cumulative_pnl() -> float:
        """trade_log.csv 마지막 행에서 cumulative_pnl 값을 로드.

        Returns:
            마지막 기록된 cumulative_pnl. 파일 없거나 비어있으면 0.0.
        """
        if not os.path.exists(TRADE_LOG_PATH):
            return 0.0
        try:
            df = pd.read_csv(TRADE_LOG_PATH)
            if df.empty or "cumulative_pnl" not in df.columns:
                return 0.0
            return float(df["cumulative_pnl"].iloc[-1])
        except Exception:
            return 0.0

    def execute(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "limit",
        price: Optional[float] = None,
        strategy_name: str = "",
        signal_score: int = 0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[dict]:
        """주문 실행.

        지정가 주문 우선 (메이커 수수료 등급).
        주문 멱등성: 동일 심볼/방향의 중복 주문 방지.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT:USDT").
            side: "buy" | "sell".
            amount: 주문 수량.
            order_type: "limit" | "market".
            price: 지정가 주문 시 가격. market 주문이면 None.
            strategy_name: 전략 이름 (기록용).
            signal_score: 신호 점수 (기록용).
            stop_loss: 손절 가격.
            take_profit: 익절 가격.

        Returns:
            거래소 주문 응답 딕셔너리, 실패 시 None.
        """
        # 멱등성 체크: 동일 심볼/방향 미체결 주문 존재 시 스킵
        for order_id, order in self.pending_orders.items():
            if order["symbol"] == symbol and order["side"] == side:
                logger.warning(f"중복 주문 방지: {symbol} {side} (기존 주문: {order_id})")
                return None

        try:
            params = {}
            if stop_loss is not None:
                params["stopLoss"] = {"triggerPrice": stop_loss}
            if take_profit is not None:
                params["takeProfit"] = {"triggerPrice": take_profit}

            if order_type == "limit" and price is not None:
                order = self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=price,
                    params=params,
                )
            else:
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    params=params,
                )

            self.pending_orders[order["id"]] = order
            logger.info(f"주문 제출: {order['id']} {symbol} {side} {amount} @ {price or 'market'}")

            self._record_trade(
                order=order,
                strategy_name=strategy_name,
                signal_score=signal_score,
            )
            return order

        except ccxt.BaseError as e:
            logger.error(f"주문 실패: {symbol} {side} {amount} — {e}")
            return None

    def cancel(self, order_id: str, symbol: str) -> bool:
        """미체결 주문 취소.

        Args:
            order_id: 취소할 주문 ID.
            symbol: 거래 심볼.

        Returns:
            취소 성공 여부.
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            self.pending_orders.pop(order_id, None)
            logger.info(f"주문 취소: {order_id}")
            return True
        except ccxt.BaseError as e:
            logger.error(f"주문 취소 실패: {order_id} — {e}")
            return False

    def sync_positions(self) -> dict:
        """거래소 실제 포지션과 로컬 상태를 동기화.

        거래소에서 현재 포지션을 조회하고 로컬 상태 파일을 업데이트한다.
        미체결 주문 목록도 갱신한다.

        Returns:
            현재 포지션 딕셔너리.
        """
        try:
            positions = self.exchange.fetch_positions()
            active = {}
            for pos in positions:
                if float(pos.get("contracts", 0)) > 0:
                    active[pos["symbol"]] = {
                        "side": pos["side"],
                        "size": float(pos["contracts"]),
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    }

            # 미체결 주문 갱신
            open_orders = self.exchange.fetch_open_orders()
            self.pending_orders = {o["id"]: o for o in open_orders}

            self._save_state(active)
            logger.info(f"포지션 동기화 완료: {len(active)}개 활성")
            return active

        except ccxt.BaseError as e:
            logger.error(f"포지션 동기화 실패: {e}")
            return {}

    def _record_trade(self, order: dict, strategy_name: str, signal_score: int) -> None:
        """거래 내역을 CSV에 append-only로 기록.

        Args:
            order: 거래소 주문 응답.
            strategy_name: 전략 이름.
            signal_score: 신호 점수.
        """
        trade = {
            "trade_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy_name,
            "symbol": order.get("symbol", ""),
            "side": order.get("side", ""),
            "order_type": order.get("type", ""),
            "price": order.get("price") or order.get("average", 0),
            "amount": order.get("amount", 0),
            "fee": order.get("fee", {}).get("cost", 0) if order.get("fee") else 0,
            "pnl": 0,
            "cumulative_pnl": self._cumulative_pnl,
            "signal_score": signal_score,
            "notes": "",
        }

        os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
        new_row = pd.DataFrame([trade])

        write_header = not os.path.exists(TRADE_LOG_PATH)
        new_row.to_csv(
            TRADE_LOG_PATH, mode="a", header=write_header, index=False
        )

    def close_position(
        self,
        symbol: str,
        position: dict,
        strategy_name: str = "",
    ) -> Optional[dict]:
        """기존 포지션을 시장가로 청산.

        Args:
            symbol: 거래 심볼.
            position: 포지션 정보 딕셔너리 (side, size 포함).
            strategy_name: 전략 이름 (기록용).

        Returns:
            거래소 주문 응답, 실패 시 None.
        """
        close_side = "sell" if position["side"] == "long" else "buy"
        # 같은 방향의 기존 미체결 주문 취소 (멱등성 체크 충돌 방지)
        for oid, order in list(self.pending_orders.items()):
            if order["symbol"] == symbol and order["side"] == close_side:
                self.cancel(oid, symbol)
        return self.execute(
            symbol=symbol,
            side=close_side,
            amount=position["size"],
            order_type="market",
            strategy_name=strategy_name,
            signal_score=0,
        )

    def record_closed_pnl(
        self,
        symbol: str,
        pnl: float,
        strategy_name: str = "",
        reason: str = "",
    ) -> None:
        """청산된 포지션의 PnL을 trade_log에 기록.

        SL/TP 자동 청산이나 반전 청산의 PnL을 별도 행으로 기록한다.

        Args:
            symbol: 거래 심볼.
            pnl: 실현 PnL.
            strategy_name: 전략 이름.
            reason: 청산 사유 (예: "SL/TP 자동 청산", "반전 청산").
        """
        self._cumulative_pnl += pnl

        trade = {
            "trade_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy_name,
            "symbol": symbol,
            "side": "close",
            "order_type": "auto",
            "price": 0,
            "amount": 0,
            "fee": 0,
            "pnl": pnl,
            "cumulative_pnl": self._cumulative_pnl,
            "signal_score": 0,
            "notes": reason,
        }

        os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
        new_row = pd.DataFrame([trade])

        write_header = not os.path.exists(TRADE_LOG_PATH)
        new_row.to_csv(
            TRADE_LOG_PATH, mode="a", header=write_header, index=False
        )
        logger.info(f"PnL 기록: {symbol} {pnl:+.4f} (누적: {self._cumulative_pnl:+.4f})")

    @staticmethod
    def _save_state(positions: dict, extra_state: Optional[dict] = None) -> None:
        """현재 상태를 Atomic Write로 JSON 파일에 저장.

        쓰다가 프로세스가 죽어도 파일이 깨지지 않도록
        임시 파일에 먼저 쓰고 이동한다.

        Args:
            positions: 현재 포지션 딕셔너리.
            extra_state: 추가 상태 (circuit_breaker, pnl_tracker 등).
        """
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "positions": positions,
        }
        if extra_state:
            state.update(extra_state)
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=os.path.dirname(STATE_PATH)
        ) as f:
            json.dump(state, f, indent=2, default=str)
            temp_path = f.name
        shutil.move(temp_path, STATE_PATH)
