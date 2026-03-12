"""포트폴리오 레벨 리스크 관리 모듈.

전략 레벨 RiskManager 위에 위치하여 포트폴리오 전체의 리스크를 제어한다.

방어 계층:
    Level 0 (정상):  MDD > -3%        → 포지션 100%
    Level 1 (주의):  -3% ≥ MDD > -5%  → 포지션 75%
    Level 2 (경고):  -5% ≥ MDD > -7%  → 포지션 50%
    Level 3 (위험):  -7% ≥ MDD > -10% → 포지션 25%
    Level 4 (비상):  MDD ≤ -10%       → 포지션 0% (전체 차단)
"""

from datetime import datetime, timezone

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

        # 점진적 포지션 축소 설정
        dd_config = config.get("drawdown_scaling", {})
        self.drawdown_scaling_enabled: bool = dd_config.get("enabled", False)
        self.drawdown_levels: list[dict] = dd_config.get("levels", [])
        # mdd_threshold 내림차순 정렬 (덜 심한 것부터)
        self.drawdown_levels.sort(key=lambda x: x["mdd_threshold"], reverse=True)

        # Rolling PF 설정
        rpf_config = config.get("rolling_pf", {})
        self.rolling_pf_enabled: bool = rpf_config.get("enabled", False)
        self.rolling_pf_window: int = rpf_config.get("window", 20)
        self.rolling_scale_threshold: float = rpf_config.get("scale_threshold", 0.7)
        self.rolling_scale_factor: float = rpf_config.get("scale_factor", 0.50)
        self.rolling_disable_threshold: float = rpf_config.get(
            "disable_threshold", 0.5
        )

        # 복구 설정
        recovery_config = config.get("recovery", {})
        self.recovery_enabled: bool = recovery_config.get("enabled", False)
        self.consecutive_wins_to_upgrade: int = recovery_config.get(
            "consecutive_wins_to_upgrade", 3
        )
        self.min_hours_at_level: int = recovery_config.get("min_hours_at_level", 24)

        # 현재 방어 레벨 상태
        self._current_level: int = 0
        self._level_entered_at: str | None = None
        self._consecutive_wins: int = 0

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

    def get_position_scale(
        self, portfolio_value: float, peak_value: float
    ) -> float:
        """MDD 기반 점진적 포지션 스케일링 계수.

        drawdown_scaling 설정의 levels를 순회하며,
        현재 MDD에 해당하는 position_scale을 반환.

        Args:
            portfolio_value: 현재 포트폴리오 가치.
            peak_value: 역대 최고 포트폴리오 가치.

        Returns:
            0.0 ~ 1.0 사이의 스케일링 계수.
        """
        if not self.drawdown_scaling_enabled or peak_value <= 0:
            return 1.0

        current_dd = (portfolio_value - peak_value) / peak_value

        scale = 1.0
        new_level = 0
        for i, level in enumerate(self.drawdown_levels):
            if current_dd <= level["mdd_threshold"]:
                scale = level["position_scale"]
                new_level = i + 1

        # 레벨 변경 시 로깅 및 상태 업데이트
        if new_level != self._current_level:
            old_level = self._current_level
            self._current_level = new_level
            self._level_entered_at = datetime.now(timezone.utc).isoformat()
            self._consecutive_wins = 0
            if new_level > old_level:
                logger.warning(
                    f"방어 레벨 상승: {old_level} → {new_level} | "
                    f"MDD {current_dd:.2%} → 포지션 {scale:.0%}"
                )
            else:
                logger.info(
                    f"방어 레벨 하락: {old_level} → {new_level} | "
                    f"MDD {current_dd:.2%} → 포지션 {scale:.0%}"
                )

        return scale

    def check_daily_loss(
        self, daily_pnl: float, portfolio_value: float
    ) -> tuple[bool, str]:
        """일일 손실 한도 체크.

        Args:
            daily_pnl: 오늘의 누적 실현 손익.
            portfolio_value: 현재 포트폴리오 가치.

        Returns:
            (통과 여부, 사유).
        """
        if portfolio_value <= 0:
            return True, "OK"

        daily_loss_pct = daily_pnl / portfolio_value
        if daily_loss_pct < self.max_daily_loss:
            msg = (
                f"일일 손실 한도 초과: {daily_loss_pct:.2%} "
                f"(한도: {self.max_daily_loss:.2%})"
            )
            logger.warning(msg)
            return False, msg
        return True, "OK"

    def get_strategy_scale(self, strategy_name: str) -> float:
        """Rolling PF 기반 전략별 스케일링 계수.

        최근 N거래의 rolling PF를 계산하여:
        - PF >= scale_threshold → 1.0 (정상)
        - PF < scale_threshold → scale_factor (축소)
        - PF < disable_threshold → 0.0 (비활성화)

        rolling_pf.window 미만의 거래 수이면 1.0 반환 (데이터 부족).

        Args:
            strategy_name: 전략 이름.

        Returns:
            0.0 ~ 1.0 사이의 스케일링 계수.
        """
        if not self.rolling_pf_enabled:
            return 1.0

        stats = self.strategy_stats.get(strategy_name)
        if not stats:
            return 1.0

        recent_trades = stats.get("recent_trades", [])
        if len(recent_trades) < self.rolling_pf_window:
            return 1.0

        window = recent_trades[-self.rolling_pf_window :]
        gross_profit = sum(pnl for pnl in window if pnl > 0)
        gross_loss = sum(abs(pnl) for pnl in window if pnl < 0)

        if gross_loss <= 0:
            return 1.0

        rolling_pf = gross_profit / gross_loss

        if rolling_pf < self.rolling_disable_threshold:
            logger.warning(
                f"전략 비활성화 (Rolling PF): {strategy_name} "
                f"PF={rolling_pf:.2f} < {self.rolling_disable_threshold}"
            )
            return 0.0
        elif rolling_pf < self.rolling_scale_threshold:
            logger.info(
                f"전략 포지션 축소 (Rolling PF): {strategy_name} "
                f"PF={rolling_pf:.2f} → {self.rolling_scale_factor:.0%}"
            )
            return self.rolling_scale_factor
        return 1.0

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

        PF, 승률 등을 실시간 업데이트하고 recent_trades 리스트를 관리한다.

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
                "recent_trades": [],
            }

        stats = self.strategy_stats[strategy_name]
        stats["total_trades"] += 1

        if pnl > 0:
            stats["wins"] += 1
            stats["gross_profit"] += pnl
            self._consecutive_wins += 1
        elif pnl < 0:
            stats["losses"] += 1
            stats["gross_loss"] += abs(pnl)
            self._consecutive_wins = 0

        if stats["gross_loss"] > 0:
            stats["profit_factor"] = stats["gross_profit"] / stats["gross_loss"]
        elif stats["gross_profit"] > 0:
            stats["profit_factor"] = float("inf")
        else:
            stats["profit_factor"] = 0.0

        # recent_trades 관리 (최대 100건)
        if "recent_trades" not in stats:
            stats["recent_trades"] = []
        stats["recent_trades"].append(pnl)
        if len(stats["recent_trades"]) > 100:
            stats["recent_trades"] = stats["recent_trades"][-100:]

        # 복구 체크
        if self.recovery_enabled and self._current_level > 0:
            self._check_recovery()

        logger.info(
            f"전략 성과 업데이트: {strategy_name} | "
            f"거래 {stats['total_trades']}건 | PF {stats['profit_factor']:.2f}"
        )

    def _check_recovery(self) -> None:
        """복구 조건 체크 및 레벨 자동 조정.

        연속 승리 횟수가 기준 이상이고 최소 대기 시간이 경과하면
        한 단계 복구한다.
        """
        if self._current_level <= 0:
            return

        if self._consecutive_wins < self.consecutive_wins_to_upgrade:
            return

        # 최소 대기 시간 체크
        if self._level_entered_at:
            entered = datetime.fromisoformat(self._level_entered_at)
            now = datetime.now(timezone.utc)
            hours_elapsed = (now - entered).total_seconds() / 3600
            if hours_elapsed < self.min_hours_at_level:
                return

        old_level = self._current_level
        self._current_level = max(0, self._current_level - 1)
        self._consecutive_wins = 0
        self._level_entered_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"방어 레벨 복구: {old_level} → {self._current_level} "
            f"(연속 {self.consecutive_wins_to_upgrade}승 달성)"
        )

    @property
    def current_level(self) -> int:
        """현재 방어 레벨 (0=정상, 4=비상)."""
        return self._current_level

    def to_dict(self) -> dict:
        """직렬화 (상태 저장용).

        Returns:
            상태 딕셔너리.
        """
        return {
            "strategy_stats": self.strategy_stats,
            "current_level": self._current_level,
            "level_entered_at": self._level_entered_at,
            "consecutive_wins": self._consecutive_wins,
        }

    def from_dict(self, data: dict) -> None:
        """역직렬화 (상태 복원용).

        Args:
            data: to_dict()로 생성된 딕셔너리.
        """
        self.strategy_stats = data.get("strategy_stats", {})
        self._current_level = data.get("current_level", 0)
        self._level_entered_at = data.get("level_entered_at", None)
        self._consecutive_wins = data.get("consecutive_wins", 0)
