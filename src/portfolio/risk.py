"""포트폴리오 레벨 리스크 관리 모듈.

전략 레벨 RiskManager 위에 위치하여 포트폴리오 전체의 리스크를 제어한다.
"""

from src.utils.logger import setup_logger

logger = setup_logger("portfolio_risk")


class PortfolioRiskManager:
    """포트폴리오 레벨 리스크 관리.

    config/portfolio.yaml의 risk 섹션을 읽어 초기화.
    전략별 실전 성과를 추적하고, 포트폴리오 전체 MDD/일일 손실을 관리한다.

    Attributes:
        max_portfolio_mdd: 전체 MDD 한도 (음수, 예: -0.10).
        max_daily_loss: 일일 손실 한도 (음수, 예: -0.03).
        strategy_disable_threshold: 전략 비활성화 PF 기준.
        strategy_disable_min_trades: 비활성화 판단 최소 거래 수.
        strategy_stats: 전략별 실전 성과 추적 딕셔너리.
    """

    def __init__(self, config: dict) -> None:
        """PortfolioRiskManager 초기화.

        Args:
            config: portfolio.yaml의 risk 섹션 딕셔너리.
        """
        self.max_portfolio_mdd: float = config.get("max_portfolio_mdd", -0.10)
        self.max_daily_loss: float = config.get("max_daily_loss", -0.03)
        self.strategy_disable_threshold: float = config.get(
            "strategy_disable_threshold", 0.8
        )
        self.strategy_disable_min_trades: int = config.get(
            "strategy_disable_min_trades", 50
        )
        self.strategy_stats: dict[str, dict] = {}
        logger.info("PortfolioRiskManager 초기화 완료")

    def check_portfolio(
        self, portfolio_value: float, peak_value: float
    ) -> tuple[bool, str]:
        """포트폴리오 전체 리스크 체크.

        Args:
            portfolio_value: 현재 포트폴리오 가치.
            peak_value: 역대 최고 포트폴리오 가치.

        Returns:
            (통과 여부, 사유).
        """
        if peak_value <= 0:
            return True, "OK"

        current_dd = (portfolio_value - peak_value) / peak_value
        if current_dd < self.max_portfolio_mdd:
            msg = f"포트폴리오 MDD 한도 초과: {current_dd:.2%}"
            logger.warning(msg)
            return False, msg

        return True, "OK"

    def check_strategy_health(self, strategy_name: str) -> bool:
        """개별 전략의 실전 성과를 기반으로 활성 상태 판단.

        strategy_disable_min_trades 이상 거래 후 PF가 threshold 미만이면 비활성화.

        Args:
            strategy_name: 전략 이름.

        Returns:
            활성 상태 여부 (True=활성, False=비활성화 권고).
        """
        stats = self.strategy_stats.get(strategy_name)
        if not stats or stats["total_trades"] < self.strategy_disable_min_trades:
            return True  # 데이터 부족 → 활성 유지
        return stats["profit_factor"] >= self.strategy_disable_threshold

    def record_trade(self, strategy_name: str, pnl: float) -> None:
        """전략별 거래 결과 기록.

        PF, 승률 등을 실시간 업데이트한다.

        Args:
            strategy_name: 전략 이름.
            pnl: 해당 거래의 실현 손익.
        """
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
            }

        stats = self.strategy_stats[strategy_name]
        stats["total_trades"] += 1

        if pnl > 0:
            stats["wins"] += 1
            stats["gross_profit"] += pnl
        elif pnl < 0:
            stats["losses"] += 1
            stats["gross_loss"] += abs(pnl)

        if stats["gross_loss"] > 0:
            stats["profit_factor"] = stats["gross_profit"] / stats["gross_loss"]
        elif stats["gross_profit"] > 0:
            stats["profit_factor"] = float("inf")
        else:
            stats["profit_factor"] = 0.0

        logger.info(
            f"전략 성과 업데이트: {strategy_name} | "
            f"거래 {stats['total_trades']}건 | PF {stats['profit_factor']:.2f}"
        )

    def to_dict(self) -> dict:
        """직렬화 (상태 저장용).

        Returns:
            상태 딕셔너리.
        """
        return {"strategy_stats": self.strategy_stats}

    def from_dict(self, data: dict) -> None:
        """역직렬화 (상태 복원용).

        Args:
            data: to_dict()로 생성된 딕셔너리.
        """
        self.strategy_stats = data.get("strategy_stats", {})
