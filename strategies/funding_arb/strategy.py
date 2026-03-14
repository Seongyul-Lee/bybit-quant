"""펀딩비 차익거래 전략.

Buy & Hold: 진입 후 계속 보유.
포지션 크기는 config에서 결정.
"""

import pandas as pd

from src.strategies.base import BaseStrategy


class FundingArbStrategy(BaseStrategy):
    """펀딩비 차익거래 Buy & Hold 전략.

    generate_signal():
      항상 signal=1 (진입/유지)
      확률은 0.0 (모니터링용 — 실제 펀딩비는 main.py에서 조회)

    이 전략은 v1 모멘텀과 다른 실행 경로를 따름:
    - v1: signal → PortfolioManager.allocate() → OrderExecutor.execute()
    - 펀딩비: signal → ArbExecutor.open_position() (현물+선물 동시)

    Attributes:
        symbol_spot: 현물 심볼 (예: "BTC/USDT").
        symbol_perp: 선물 심볼 (예: "BTC/USDT:USDT").
        leverage: 선물 레버리지.
        position_pct: 자본 대비 포지션 비율.
    """

    def __init__(self, config: dict) -> None:
        """FundingArbStrategy 초기화.

        Args:
            config: 전략 설정 (params 섹션).
        """
        super().__init__(config)
        self.symbol_spot = config.get("symbol_spot", "BTC/USDT")
        self.symbol_perp = config.get("symbol_perp", "BTC/USDT:USDT")
        self.leverage = config.get("leverage", 2)
        self.position_pct = config.get("position_pct", 0.40)
        self.symbol = self.symbol_perp

    def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """Buy & Hold: 항상 1(진입/유지).

        Args:
            df: OHLCV DataFrame (사용하지 않지만 인터페이스 준수).

        Returns:
            (1, 0.0) — 항상 진입 신호.
        """
        return 1, 0.0

    def generate_signals_vectorized(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """백테스트용: 항상 1.

        Args:
            df: OHLCV DataFrame.

        Returns:
            (signal_series, probability_series) — 모두 1, 0.0.
        """
        signals = pd.Series(1, index=df.index)
        probs = pd.Series(0.0, index=df.index)
        return signals, probs
