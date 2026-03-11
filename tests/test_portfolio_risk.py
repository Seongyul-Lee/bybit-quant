"""PortfolioRiskManager 테스트."""

import pytest

from src.portfolio.risk import PortfolioRiskManager


class TestPortfolioRiskManager:
    """PortfolioRiskManager 테스트."""

    def setup_method(self) -> None:
        """각 테스트 전 초기화."""
        self.config = {
            "max_portfolio_mdd": -0.10,
            "max_daily_loss": -0.03,
            "strategy_disable_threshold": 0.8,
            "strategy_disable_min_trades": 50,
        }
        self.risk = PortfolioRiskManager(self.config)

    def test_check_portfolio_within_limit(self) -> None:
        """MDD 한도 이내 → 통과."""
        ok, reason = self.risk.check_portfolio(
            portfolio_value=95000.0, peak_value=100000.0
        )
        assert ok is True
        assert reason == "OK"

    def test_check_portfolio_mdd_exceeded(self) -> None:
        """MDD 한도 초과 → 실패."""
        ok, reason = self.risk.check_portfolio(
            portfolio_value=89000.0, peak_value=100000.0
        )
        assert ok is False
        assert "MDD 한도 초과" in reason

    def test_check_portfolio_mdd_exact_boundary(self) -> None:
        """MDD 정확히 한도 → 통과 (경계값)."""
        ok, reason = self.risk.check_portfolio(
            portfolio_value=90000.0, peak_value=100000.0
        )
        # -10% exactly → should pass (not < -0.10)
        assert ok is True

    def test_check_portfolio_zero_peak(self) -> None:
        """피크 0 → 무조건 통과."""
        ok, reason = self.risk.check_portfolio(
            portfolio_value=50000.0, peak_value=0.0
        )
        assert ok is True

    def test_strategy_health_insufficient_data(self) -> None:
        """50거래 미만 → 항상 True."""
        # 10건의 손실 기록
        for _ in range(10):
            self.risk.record_trade("strat_a", -100.0)

        assert self.risk.check_strategy_health("strat_a") is True

    def test_strategy_health_pf_below_threshold(self) -> None:
        """50거래 이상 + PF < 0.8 → False."""
        # 40건 손실, 10건 이익 (PF = 500/4000 = 0.125)
        for _ in range(40):
            self.risk.record_trade("strat_a", -100.0)
        for _ in range(10):
            self.risk.record_trade("strat_a", 50.0)

        assert self.risk.check_strategy_health("strat_a") is False

    def test_strategy_health_pf_above_threshold(self) -> None:
        """50거래 이상 + PF >= 0.8 → True."""
        # 30건 이익, 20건 손실 (PF = 3000/2000 = 1.5)
        for _ in range(30):
            self.risk.record_trade("strat_a", 100.0)
        for _ in range(20):
            self.risk.record_trade("strat_a", -100.0)

        assert self.risk.check_strategy_health("strat_a") is True

    def test_strategy_health_unknown_strategy(self) -> None:
        """알 수 없는 전략 → True."""
        assert self.risk.check_strategy_health("unknown") is True

    def test_record_trade_pf_calculation(self) -> None:
        """PF 계산 정확성."""
        self.risk.record_trade("strat_a", 200.0)
        self.risk.record_trade("strat_a", -100.0)

        stats = self.risk.strategy_stats["strat_a"]
        assert stats["total_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["gross_profit"] == 200.0
        assert stats["gross_loss"] == 100.0
        assert stats["profit_factor"] == 2.0

    def test_record_trade_all_wins(self) -> None:
        """전부 이익 → PF = inf."""
        self.risk.record_trade("strat_a", 100.0)
        self.risk.record_trade("strat_a", 200.0)

        stats = self.risk.strategy_stats["strat_a"]
        assert stats["profit_factor"] == float("inf")

    def test_to_dict_from_dict(self) -> None:
        """상태 직렬화/역직렬화."""
        self.risk.record_trade("strat_a", 100.0)
        self.risk.record_trade("strat_a", -50.0)

        state = self.risk.to_dict()

        new_risk = PortfolioRiskManager(self.config)
        new_risk.from_dict(state)

        assert "strat_a" in new_risk.strategy_stats
        assert new_risk.strategy_stats["strat_a"]["total_trades"] == 2
        assert new_risk.strategy_stats["strat_a"]["profit_factor"] == 2.0
