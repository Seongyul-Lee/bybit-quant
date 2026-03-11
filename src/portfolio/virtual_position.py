"""전략별 가상 포지션 추적 모듈.

Bybit 원웨이 모드에서는 심볼당 하나의 포지션만 존재한다.
여러 전략이 동일 심볼에 포지션을 가지면 내부적으로 가상 분리하고
실제로는 합산 포지션을 유지한다.
"""

from src.utils.logger import setup_logger

logger = setup_logger("virtual_position")


class VirtualPositionTracker:
    """전략별 가상 포지션 추적.

    내부 구조:
        {strategy_name: {symbol: {"side": "long", "size": 0.002, "entry_price": 80000}}}

    Attributes:
        virtual_positions: 전략별 가상 포지션 딕셔너리.
    """

    def __init__(self) -> None:
        """VirtualPositionTracker 초기화."""
        self.virtual_positions: dict[str, dict[str, dict]] = {}

    def open(
        self,
        strategy_name: str,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
    ) -> None:
        """가상 포지션 생성.

        Args:
            strategy_name: 전략 이름.
            symbol: 거래 심볼 (예: "BTC/USDT:USDT").
            side: 포지션 방향 ("long" | "short").
            size: 포지션 수량.
            entry_price: 진입 가격.
        """
        if strategy_name not in self.virtual_positions:
            self.virtual_positions[strategy_name] = {}

        self.virtual_positions[strategy_name][symbol] = {
            "side": side,
            "size": size,
            "entry_price": entry_price,
        }
        logger.info(
            f"가상 포지션 생성: {strategy_name} | {symbol} {side} {size} @ {entry_price}"
        )

    def close(self, strategy_name: str, symbol: str) -> None:
        """전략의 가상 포지션 청산.

        Args:
            strategy_name: 전략 이름.
            symbol: 거래 심볼.
        """
        if strategy_name in self.virtual_positions:
            removed = self.virtual_positions[strategy_name].pop(symbol, None)
            if removed:
                logger.info(f"가상 포지션 청산: {strategy_name} | {symbol}")
            if not self.virtual_positions[strategy_name]:
                del self.virtual_positions[strategy_name]

    def has_position(self, strategy_name: str, symbol: str) -> bool:
        """전략이 해당 심볼에 가상 포지션을 보유하는지 확인.

        Args:
            strategy_name: 전략 이름.
            symbol: 거래 심볼.

        Returns:
            포지션 보유 여부.
        """
        return (
            strategy_name in self.virtual_positions
            and symbol in self.virtual_positions[strategy_name]
        )

    def get_position(self, strategy_name: str, symbol: str) -> dict:
        """특정 전략의 특정 심볼 가상 포지션 반환.

        Args:
            strategy_name: 전략 이름.
            symbol: 거래 심볼.

        Returns:
            포지션 딕셔너리 또는 빈 딕셔너리.
        """
        if self.has_position(strategy_name, symbol):
            return self.virtual_positions[strategy_name][symbol]
        return {}

    def get_real_position(self, symbol: str) -> dict:
        """심볼의 실제 포지션 (전략별 합산) 반환.

        Args:
            symbol: 거래 심볼.

        Returns:
            {"side": "long", "size": 0.003} 또는 빈 딕셔너리.
        """
        total_size = 0.0
        for strategy_positions in self.virtual_positions.values():
            if symbol in strategy_positions:
                pos = strategy_positions[symbol]
                if pos["side"] == "long":
                    total_size += pos["size"]
                else:
                    total_size -= pos["size"]

        if total_size > 0:
            return {"side": "long", "size": total_size}
        elif total_size < 0:
            return {"side": "short", "size": abs(total_size)}
        return {}

    def get_delta_orders(self, symbol: str, current_real: dict) -> list[dict]:
        """가상 합산과 실제 포지션의 차이를 주문으로 변환.

        Args:
            symbol: 거래 심볼.
            current_real: 거래소에서 조회한 현재 실제 포지션.
                {"side": "long", "size": 0.003} 또는 빈 딕셔너리.

        Returns:
            [{"symbol": sym, "side": "buy"|"sell", "amount": float}]
        """
        virtual = self.get_real_position(symbol)

        # 가상/실제를 부호 있는 수량으로 변환 (long=+, short=-)
        virtual_signed = 0.0
        if virtual:
            virtual_signed = virtual["size"] if virtual["side"] == "long" else -virtual["size"]

        real_signed = 0.0
        if current_real:
            real_signed = (
                current_real["size"]
                if current_real["side"] == "long"
                else -current_real["size"]
            )

        delta = virtual_signed - real_signed

        if abs(delta) < 1e-10:
            return []

        if delta > 0:
            return [{"symbol": symbol, "side": "buy", "amount": delta}]
        else:
            return [{"symbol": symbol, "side": "sell", "amount": abs(delta)}]

    def get_all_symbols(self) -> set[str]:
        """모든 가상 포지션의 심볼 집합 반환.

        Returns:
            심볼 집합.
        """
        symbols: set[str] = set()
        for strategy_positions in self.virtual_positions.values():
            symbols.update(strategy_positions.keys())
        return symbols

    def get_strategies_for_symbol(self, symbol: str) -> list[str]:
        """해당 심볼에 가상 포지션을 보유한 전략 목록.

        Args:
            symbol: 거래 심볼.

        Returns:
            전략 이름 리스트.
        """
        strategies = []
        for strategy_name, positions in self.virtual_positions.items():
            if symbol in positions:
                strategies.append(strategy_name)
        return strategies

    def to_dict(self) -> dict:
        """직렬화 (상태 저장용).

        Returns:
            가상 포지션 딕셔너리.
        """
        return self.virtual_positions

    def from_dict(self, data: dict) -> None:
        """역직렬화 (상태 복원용).

        Args:
            data: to_dict()로 생성된 딕셔너리.
        """
        self.virtual_positions = data
