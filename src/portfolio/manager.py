"""멀티 전략 포트폴리오 관리 모듈.

전략 등록, 시그널 수집, 자본 배분, 포지션 한도 제어를 담당한다.
"""

import importlib
import inspect
import os

import pandas as pd
import yaml

from src.strategies.base import BaseStrategy
from src.portfolio.virtual_position import VirtualPositionTracker
from src.utils.config import merge_strategy_params
from src.utils.logger import setup_logger

logger = setup_logger("portfolio_manager")


class PortfolioManager:
    """멀티 전략 포트폴리오 관리.

    config/portfolio.yaml을 읽어 초기화.
    전략별 시그널을 수집하고, 자본 배분 규칙에 따라 최종 주문 목록을 생성한다.

    Attributes:
        strategies: 등록된 전략 딕셔너리.
        position_pct: 전략당 포지션 비중.
        max_total_exposure: 전체 포지션 합산 상한.
        max_symbol_exposure: 동일 심볼 합산 상한.
        max_concurrent_positions: 동시 최대 포지션 수.
    """

    def __init__(self, config: dict) -> None:
        """PortfolioManager 초기화.

        Args:
            config: portfolio.yaml의 portfolio 섹션 딕셔너리.
        """
        self.strategies: dict[str, BaseStrategy] = {}
        self.strategy_configs: dict[str, dict] = {}

        allocation = config.get("allocation", {})
        self.position_pct: float = allocation.get("position_pct_per_strategy", 0.20)

        limits = config.get("limits", {})
        self.max_total_exposure: float = limits.get("max_total_exposure", 0.60)
        self.max_symbol_exposure: float = limits.get("max_symbol_exposure", 0.30)
        self.max_concurrent_positions: int = limits.get("max_concurrent_positions", 5)

        logger.info(
            f"PortfolioManager 초기화: position_pct={self.position_pct}, "
            f"max_total={self.max_total_exposure}, max_symbol={self.max_symbol_exposure}"
        )

    def register_strategy(
        self, name: str, strategy: BaseStrategy, config: dict | None = None
    ) -> None:
        """전략 등록.

        Args:
            name: 전략 이름.
            strategy: BaseStrategy 인스턴스.
            config: 전략 설정 딕셔너리 (strategy config.yaml 전체).
        """
        self.strategies[name] = strategy
        if config:
            self.strategy_configs[name] = config
        logger.info(f"전략 등록: {name} ({type(strategy).__name__})")

    def unregister_strategy(self, name: str) -> None:
        """전략 해제.

        Args:
            name: 전략 이름.
        """
        self.strategies.pop(name, None)
        self.strategy_configs.pop(name, None)
        logger.info(f"전략 해제: {name}")

    def get_strategy(self, name: str) -> BaseStrategy | None:
        """등록된 전략 반환.

        Args:
            name: 전략 이름.

        Returns:
            BaseStrategy 인스턴스 또는 None.
        """
        return self.strategies.get(name)

    def get_active_strategies(self) -> list[str]:
        """활성 전략 이름 목록.

        Returns:
            전략 이름 리스트.
        """
        return list(self.strategies.keys())

    def get_strategy_config(self, name: str) -> dict:
        """전략의 원본 config.yaml 설정 반환.

        Args:
            name: 전략 이름.

        Returns:
            전략 설정 딕셔너리.
        """
        return self.strategy_configs.get(name, {})

    def collect_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, tuple[int, float]]:
        """모든 전략의 시그널을 수집.

        Args:
            data: {전략이름: 해당 전략의 DataFrame} 딕셔너리.

        Returns:
            {전략이름: (signal, probability)} 딕셔너리.
        """
        signals: dict[str, tuple[int, float]] = {}
        for name, strategy in self.strategies.items():
            if name in data:
                try:
                    signal, prob = strategy.generate_signal(data[name])
                    signals[name] = (signal, prob)
                except Exception as e:
                    logger.error(f"전략 {name} 시그널 생성 실패: {e}")
        return signals

    def allocate(
        self,
        signals: dict[str, tuple[int, float]],
        portfolio_value: float,
        virtual_tracker: VirtualPositionTracker,
        portfolio_scale: float = 1.0,
        strategy_scales: dict[str, float] | None = None,
        current_prices: dict[str, float] | None = None,
    ) -> list[dict]:
        """시그널을 기반으로 주문 목록 생성.

        양방향 시그널 처리:
        signal == 1  → 롱 주문 (side="buy")
        signal == -1 → 숏 주문 (side="sell")
        signal == 0  → 무시

        자본 배분 규칙:
        1. 활성 시그널(signal != 0) 전략만 필터링
        2. 각 전략에 position_pct_per_strategy × portfolio_scale × strategy_scale 적용
        3. 이미 같은 방향 포지션 보유 중인 전략은 제외
        4. 동일 심볼 합산 캡 적용
        5. 전체 노출 캡 적용
        6. 최소 주문 금액(100 USDT) 미달 시 제거

        Args:
            signals: {전략이름: (signal, probability)} 딕셔너리.
            portfolio_value: 현재 포트폴리오 가치 (USDT).
            virtual_tracker: 현재 가상 포지션 추적기.
            portfolio_scale: MDD 기반 포트폴리오 레벨 스케일링 (0.0~1.0).
            strategy_scales: {전략이름: 스케일링 계수} Rolling PF 기반 (0.0~1.0).

        Returns:
            주문 목록. 각 주문은:
            {"strategy": name, "symbol": sym, "side": "buy"|"sell",
             "direction": "long"|"short", "size_pct": float, ...}
        """
        strategy_scales = strategy_scales or {}
        orders: list[dict] = []

        # 1. 활성 시그널 필터링 (롱 + 숏)
        active_signals = {
            name: (sig, prob)
            for name, (sig, prob) in signals.items()
            if sig != 0
        }

        if not active_signals:
            return orders

        # 동시 포지션 수 체크
        current_position_count = len(virtual_tracker.get_all_symbols())
        remaining_slots = self.max_concurrent_positions - current_position_count

        for name, (signal, prob) in active_signals.items():
            if remaining_slots <= 0:
                logger.info(f"동시 포지션 한도 도달 — {name} 스킵")
                break

            strategy = self.strategies.get(name)
            if not strategy:
                continue

            strategy_config = self.strategy_configs.get(name, {})
            symbol_raw = strategy_config.get("strategy", {}).get(
                "symbol", strategy.symbol
            )
            symbol = self._convert_symbol(symbol_raw)

            # 이미 포지션 보유 중인 전략은 제외
            if virtual_tracker.has_position(name, symbol):
                logger.info(f"이미 포지션 보유: {name} | {symbol} — 스킵")
                continue

            # 방향 결정
            side = "buy" if signal == 1 else "sell"
            direction = "long" if signal == 1 else "short"

            # 스케일링 적용: position_pct × portfolio_scale × strategy_scale
            strat_scale = strategy_scales.get(name, 1.0)
            effective_pct = self.position_pct * portfolio_scale * strat_scale
            order_value = portfolio_value * effective_pct

            # 최소 주문 금액 체크
            if order_value < 100:
                logger.info(
                    f"최소 주문 미달: {name} | {order_value:.0f} USDT < 100 USDT"
                )
                continue

            orders.append(
                {
                    "strategy": name,
                    "symbol": symbol,
                    "side": side,
                    "direction": direction,
                    "size_pct": effective_pct,
                    "signal": signal,
                    "probability": prob,
                }
            )
            remaining_slots -= 1

        # 동일 심볼 합산 캡 적용
        orders = self._apply_symbol_cap(
            orders, portfolio_value, virtual_tracker, current_prices
        )

        # 전체 노출 캡 적용
        orders = self._apply_total_cap(
            orders, portfolio_value, virtual_tracker, current_prices
        )

        return orders

    def _apply_symbol_cap(
        self,
        orders: list[dict],
        portfolio_value: float,
        virtual_tracker: VirtualPositionTracker,
        current_prices: dict[str, float] | None = None,
    ) -> list[dict]:
        """동일 심볼 합산이 max_symbol_exposure 초과 시 비례 축소.

        Args:
            orders: 주문 목록.
            portfolio_value: 포트폴리오 가치.
            virtual_tracker: 가상 포지션 추적기.
            current_prices: {심볼: 현재가} 딕셔너리. None이면 entry_price 사용.

        Returns:
            조정된 주문 목록.
        """
        current_prices = current_prices or {}

        # 심볼별 신규 주문 합산
        symbol_new_exposure: dict[str, float] = {}
        for order in orders:
            sym = order["symbol"]
            symbol_new_exposure[sym] = (
                symbol_new_exposure.get(sym, 0.0) + order["size_pct"]
            )

        for sym, new_pct in symbol_new_exposure.items():
            # 기존 가상 포지션의 노출 계산 (현재가 반영)
            existing_pct = 0.0
            for strat_positions in virtual_tracker.virtual_positions.values():
                if sym in strat_positions:
                    pos = strat_positions[sym]
                    if portfolio_value > 0:
                        price = current_prices.get(sym, pos["entry_price"])
                        existing_pct += (
                            pos["size"] * price
                        ) / portfolio_value

            total_pct = existing_pct + new_pct
            if total_pct > self.max_symbol_exposure and new_pct > 0:
                # 비례 축소
                scale = max(0, self.max_symbol_exposure - existing_pct) / new_pct
                for order in orders:
                    if order["symbol"] == sym:
                        order["size_pct"] *= scale
                if scale < 1.0:
                    logger.info(
                        f"심볼 캡 적용: {sym} 비례 축소 {scale:.2f}"
                    )

        return orders

    def _apply_total_cap(
        self,
        orders: list[dict],
        portfolio_value: float,
        virtual_tracker: VirtualPositionTracker,
        current_prices: dict[str, float] | None = None,
    ) -> list[dict]:
        """전체 합산이 max_total_exposure 초과 시 비례 축소.

        Args:
            orders: 주문 목록.
            portfolio_value: 포트폴리오 가치.
            virtual_tracker: 가상 포지션 추적기.
            current_prices: {심볼: 현재가} 딕셔너리. None이면 entry_price 사용.

        Returns:
            조정된 주문 목록.
        """
        current_prices = current_prices or {}

        # 기존 전체 노출 (현재가 반영)
        existing_pct = 0.0
        if portfolio_value > 0:
            for strat_positions in virtual_tracker.virtual_positions.values():
                for sym, pos in strat_positions.items():
                    price = current_prices.get(sym, pos["entry_price"])
                    existing_pct += (
                        pos["size"] * price
                    ) / portfolio_value

        new_pct = sum(o["size_pct"] for o in orders)
        total_pct = existing_pct + new_pct

        if total_pct > self.max_total_exposure and new_pct > 0:
            scale = max(0, self.max_total_exposure - existing_pct) / new_pct
            for order in orders:
                order["size_pct"] *= scale
            if scale < 1.0:
                logger.info(f"전체 노출 캡 적용: 비례 축소 {scale:.2f}")

        return orders

    def load_strategies_from_config(self, portfolio_config: dict) -> None:
        """portfolio.yaml의 active_strategies를 읽어 전략을 동적으로 로드/등록.

        각 전략의 config.yaml을 읽고, strategy.py를 importlib으로 동적 import하여
        BaseStrategy 서브클래스를 찾아 인스턴스 생성.

        Args:
            portfolio_config: portfolio.yaml의 portfolio 섹션.

        Raises:
            FileNotFoundError: 전략 폴더나 config.yaml이 없는 경우.
            ValueError: BaseStrategy 서브클래스를 찾을 수 없는 경우.
        """
        active_strategies = portfolio_config.get("active_strategies", [])

        for strategy_name in active_strategies:
            # 전략 폴더 존재 확인
            strategy_dir = os.path.join("strategies", strategy_name)
            if not os.path.isdir(strategy_dir):
                raise FileNotFoundError(
                    f"전략 폴더를 찾을 수 없습니다: {strategy_dir}"
                )

            # config.yaml 로드
            config_path = os.path.join(strategy_dir, "config.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"전략 설정 파일을 찾을 수 없습니다: {config_path}"
                )

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 동적 import
            module = importlib.import_module(f"strategies.{strategy_name}.strategy")

            # BaseStrategy 서브클래스 찾기
            strategy_class = None
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                    strategy_class = obj
                    break

            if strategy_class is None:
                raise ValueError(
                    f"전략 모듈에서 BaseStrategy 서브클래스를 찾을 수 없습니다: "
                    f"strategies/{strategy_name}/strategy.py"
                )

            strategy = strategy_class(config=merge_strategy_params(config))
            self.register_strategy(strategy_name, strategy, config)

        logger.info(
            f"전략 로드 완료: {len(active_strategies)}개 "
            f"({', '.join(active_strategies)})"
        )

    @staticmethod
    def _convert_symbol(symbol_raw: str) -> str:
        """파일명용 심볼(BTCUSDT)을 ccxt 심볼(BTC/USDT:USDT)로 변환.

        Args:
            symbol_raw: 파일명용 심볼 (예: "BTCUSDT").

        Returns:
            ccxt 형식 심볼 (예: "BTC/USDT:USDT").
        """
        for quote in ("USDT", "USDC", "BTC"):
            if symbol_raw.endswith(quote):
                base = symbol_raw[: -len(quote)]
                return f"{base}/{quote}:{quote}"
        raise ValueError(f"알 수 없는 심볼 형식: {symbol_raw}")
