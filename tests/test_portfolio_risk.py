"""PortfolioRiskManager 테스트."""

import pytest

from src.portfolio.risk import PortfolioRiskManager


def _make_config(**overrides) -> dict:
    """기본 설정 + 오버라이드."""
    config = {
        "max_portfolio_mdd": -0.10,
        "max_daily_loss": -0.03,
        "strategy_disable_threshold": 0.8,
        "strategy_disable_min_trades": 50,
        "drawdown_scaling": {
            "enabled": True,
            "levels": [
                {"mdd_threshold": -0.03, "position_scale": 0.75},
                {"mdd_threshold": -0.05, "position_scale": 0.50},
                {"mdd_threshold": -0.07, "position_scale": 0.25},
                {"mdd_threshold": -0.10, "position_scale": 0.00},
            ],
        },
        "rolling_pf": {
            "enabled": True,
            "window": 20,
            "scale_threshold": 0.7,
            "scale_factor": 0.50,
            "disable_threshold": 0.5,
        },
        "recovery": {
            "enabled": True,
            "consecutive_wins_to_upgrade": 3,
            "min_hours_at_level": 0,  # 테스트에서는 대기 없이 즉시 복구
        },
    }
    config.update(overrides)
    return config


class TestCheckPortfolio:
    """포트폴리오 MDD 체크 테스트."""

    def test_within_limit(self) -> None:
        """MDD 한도 이내 → 통과."""
        risk = PortfolioRiskManager(_make_config())
        ok, reason = risk.check_portfolio(95000.0, 100000.0)
        assert ok is True
        assert reason == "OK"

    def test_mdd_exceeded(self) -> None:
        """MDD 한도 초과 → 실패."""
        risk = PortfolioRiskManager(_make_config())
        ok, reason = risk.check_portfolio(89000.0, 100000.0)
        assert ok is False
        assert "MDD 한도 초과" in reason

    def test_mdd_exact_boundary(self) -> None:
        """MDD 정확히 한도 → 통과 (경계값)."""
        risk = PortfolioRiskManager(_make_config())
        ok, _ = risk.check_portfolio(90000.0, 100000.0)
        assert ok is True

    def test_zero_peak(self) -> None:
        """피크 0 → 무조건 통과."""
        risk = PortfolioRiskManager(_make_config())
        ok, _ = risk.check_portfolio(50000.0, 0.0)
        assert ok is True


class TestPositionScale:
    """점진적 포지션 스케일링 테스트."""

    def test_normal(self) -> None:
        """MDD > -3% → scale 1.0."""
        risk = PortfolioRiskManager(_make_config())
        scale = risk.get_position_scale(98000.0, 100000.0)  # -2%
        assert scale == 1.0

    def test_level1(self) -> None:
        """MDD -3% ~ -5% → scale 0.75."""
        risk = PortfolioRiskManager(_make_config())
        scale = risk.get_position_scale(96000.0, 100000.0)  # -4%
        assert scale == 0.75

    def test_level2(self) -> None:
        """MDD -5% ~ -7% → scale 0.50."""
        risk = PortfolioRiskManager(_make_config())
        scale = risk.get_position_scale(94000.0, 100000.0)  # -6%
        assert scale == 0.50

    def test_level3(self) -> None:
        """MDD -7% ~ -10% → scale 0.25."""
        risk = PortfolioRiskManager(_make_config())
        scale = risk.get_position_scale(92000.0, 100000.0)  # -8%
        assert scale == 0.25

    def test_level4(self) -> None:
        """MDD ≤ -10% → scale 0.0."""
        risk = PortfolioRiskManager(_make_config())
        scale = risk.get_position_scale(89000.0, 100000.0)  # -11%
        assert scale == 0.0

    def test_exact_boundary_3pct(self) -> None:
        """MDD 정확히 -3% → scale 0.75 (threshold 이하)."""
        risk = PortfolioRiskManager(_make_config())
        scale = risk.get_position_scale(97000.0, 100000.0)  # -3%
        assert scale == 0.75

    def test_exact_boundary_10pct(self) -> None:
        """MDD 정확히 -10% → scale 0.0."""
        risk = PortfolioRiskManager(_make_config())
        scale = risk.get_position_scale(90000.0, 100000.0)  # -10%
        assert scale == 0.0

    def test_disabled(self) -> None:
        """drawdown_scaling 비활성화 → 항상 1.0."""
        config = _make_config()
        config["drawdown_scaling"]["enabled"] = False
        risk = PortfolioRiskManager(config)
        scale = risk.get_position_scale(89000.0, 100000.0)  # -11%
        assert scale == 1.0

    def test_zero_peak(self) -> None:
        """피크 0 → 1.0."""
        risk = PortfolioRiskManager(_make_config())
        scale = risk.get_position_scale(50000.0, 0.0)
        assert scale == 1.0

    def test_level_tracking(self) -> None:
        """레벨 변경이 올바르게 추적되는지 확인."""
        risk = PortfolioRiskManager(_make_config())
        assert risk.current_level == 0

        risk.get_position_scale(96000.0, 100000.0)  # -4% → Level 1
        assert risk.current_level == 1

        risk.get_position_scale(93000.0, 100000.0)  # -7% → Level 3
        assert risk.current_level == 3


class TestDailyLoss:
    """일일 손실 한도 체크 테스트."""

    def test_within_limit(self) -> None:
        """일일 손실 -2% → 통과."""
        risk = PortfolioRiskManager(_make_config())
        ok, reason = risk.check_daily_loss(-2000.0, 100000.0)
        assert ok is True
        assert reason == "OK"

    def test_exceeds_limit(self) -> None:
        """일일 손실 -4% → 차단."""
        risk = PortfolioRiskManager(_make_config())
        ok, reason = risk.check_daily_loss(-4000.0, 100000.0)
        assert ok is False
        assert "일일 손실 한도 초과" in reason

    def test_exact_boundary(self) -> None:
        """일일 손실 정확히 -3% → 통과 (not <)."""
        risk = PortfolioRiskManager(_make_config())
        ok, _ = risk.check_daily_loss(-3000.0, 100000.0)
        assert ok is True

    def test_zero_portfolio(self) -> None:
        """포트폴리오 0 → 통과."""
        risk = PortfolioRiskManager(_make_config())
        ok, _ = risk.check_daily_loss(-5000.0, 0.0)
        assert ok is True

    def test_positive_pnl(self) -> None:
        """양수 PnL → 통과."""
        risk = PortfolioRiskManager(_make_config())
        ok, _ = risk.check_daily_loss(5000.0, 100000.0)
        assert ok is True


class TestRollingPF:
    """Rolling PF 기반 전략별 스케일링 테스트."""

    def test_insufficient_data(self) -> None:
        """20거래 미만 → scale 1.0."""
        risk = PortfolioRiskManager(_make_config())
        for _ in range(10):
            risk.record_trade("strat_a", 100.0)
        assert risk.get_strategy_scale("strat_a") == 1.0

    def test_healthy(self) -> None:
        """Rolling PF 1.5 → scale 1.0."""
        risk = PortfolioRiskManager(_make_config())
        # 15 wins, 5 losses → PF = 1500/500 = 3.0
        for _ in range(15):
            risk.record_trade("strat_a", 100.0)
        for _ in range(5):
            risk.record_trade("strat_a", -100.0)
        assert risk.get_strategy_scale("strat_a") == 1.0

    def test_weak(self) -> None:
        """Rolling PF 0.6 → scale 0.5."""
        risk = PortfolioRiskManager(_make_config())
        # 6 wins, 14 losses → PF = 600/1400 ≈ 0.43 < 0.5 → 0.0
        # Adjust: 8 wins, 12 losses → PF = 800/1200 ≈ 0.67 → scale 0.5
        for _ in range(8):
            risk.record_trade("strat_a", 100.0)
        for _ in range(12):
            risk.record_trade("strat_a", -100.0)
        assert risk.get_strategy_scale("strat_a") == 0.50

    def test_critical(self) -> None:
        """Rolling PF 0.4 → scale 0.0."""
        risk = PortfolioRiskManager(_make_config())
        # 5 wins, 15 losses → PF = 500/1500 ≈ 0.33 < 0.5
        for _ in range(5):
            risk.record_trade("strat_a", 100.0)
        for _ in range(15):
            risk.record_trade("strat_a", -100.0)
        assert risk.get_strategy_scale("strat_a") == 0.0

    def test_unknown_strategy(self) -> None:
        """알 수 없는 전략 → scale 1.0."""
        risk = PortfolioRiskManager(_make_config())
        assert risk.get_strategy_scale("unknown") == 1.0

    def test_all_wins(self) -> None:
        """전부 이익 (gross_loss=0) → scale 1.0."""
        risk = PortfolioRiskManager(_make_config())
        for _ in range(20):
            risk.record_trade("strat_a", 100.0)
        assert risk.get_strategy_scale("strat_a") == 1.0

    def test_disabled(self) -> None:
        """rolling_pf 비활성화 → 항상 1.0."""
        config = _make_config()
        config["rolling_pf"]["enabled"] = False
        risk = PortfolioRiskManager(config)
        for _ in range(5):
            risk.record_trade("strat_a", 100.0)
        for _ in range(15):
            risk.record_trade("strat_a", -100.0)
        assert risk.get_strategy_scale("strat_a") == 1.0


class TestStrategyHealth:
    """전략 건강 체크 테스트."""

    def test_insufficient_data(self) -> None:
        """50거래 미만 → 항상 True."""
        risk = PortfolioRiskManager(_make_config())
        for _ in range(10):
            risk.record_trade("strat_a", -100.0)
        assert risk.check_strategy_health("strat_a") is True

    def test_pf_below_threshold(self) -> None:
        """50거래 이상 + PF < 0.8 → False."""
        risk = PortfolioRiskManager(_make_config())
        for _ in range(40):
            risk.record_trade("strat_a", -100.0)
        for _ in range(10):
            risk.record_trade("strat_a", 50.0)
        assert risk.check_strategy_health("strat_a") is False

    def test_pf_above_threshold(self) -> None:
        """50거래 이상 + PF >= 0.8 → True."""
        risk = PortfolioRiskManager(_make_config())
        for _ in range(30):
            risk.record_trade("strat_a", 100.0)
        for _ in range(20):
            risk.record_trade("strat_a", -100.0)
        assert risk.check_strategy_health("strat_a") is True

    def test_unknown_strategy(self) -> None:
        """알 수 없는 전략 → True."""
        risk = PortfolioRiskManager(_make_config())
        assert risk.check_strategy_health("unknown") is True


class TestRecordTrade:
    """거래 기록 테스트."""

    def test_pf_calculation(self) -> None:
        """PF 계산 정확성."""
        risk = PortfolioRiskManager(_make_config())
        risk.record_trade("strat_a", 200.0)
        risk.record_trade("strat_a", -100.0)

        stats = risk.strategy_stats["strat_a"]
        assert stats["total_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["gross_profit"] == 200.0
        assert stats["gross_loss"] == 100.0
        assert stats["profit_factor"] == 2.0

    def test_all_wins(self) -> None:
        """전부 이익 → PF = inf."""
        risk = PortfolioRiskManager(_make_config())
        risk.record_trade("strat_a", 100.0)
        risk.record_trade("strat_a", 200.0)
        assert risk.strategy_stats["strat_a"]["profit_factor"] == float("inf")

    def test_recent_trades_tracking(self) -> None:
        """recent_trades가 올바르게 추적되는지 확인."""
        risk = PortfolioRiskManager(_make_config())
        for i in range(5):
            risk.record_trade("strat_a", float(i + 1))

        assert risk.strategy_stats["strat_a"]["recent_trades"] == [
            1.0, 2.0, 3.0, 4.0, 5.0
        ]

    def test_recent_trades_max_100(self) -> None:
        """recent_trades가 100건 초과 시 잘리는지 확인."""
        risk = PortfolioRiskManager(_make_config())
        for i in range(110):
            risk.record_trade("strat_a", float(i))

        assert len(risk.strategy_stats["strat_a"]["recent_trades"]) == 100
        assert risk.strategy_stats["strat_a"]["recent_trades"][0] == 10.0


class TestRecovery:
    """복구 메커니즘 테스트."""

    def test_recovery_after_consecutive_wins(self) -> None:
        """연속 3승 시 한 단계 복구."""
        risk = PortfolioRiskManager(_make_config())

        # Level 2로 이동 (MDD -6%)
        risk.get_position_scale(94000.0, 100000.0)
        assert risk.current_level == 2

        # 연속 3승 → Level 1로 복구
        risk.record_trade("strat_a", 100.0)
        risk.record_trade("strat_a", 100.0)
        risk.record_trade("strat_a", 100.0)
        assert risk.current_level == 1

    def test_no_recovery_with_loss(self) -> None:
        """손실이 끼면 연속 승 카운트 리셋."""
        risk = PortfolioRiskManager(_make_config())

        risk.get_position_scale(94000.0, 100000.0)  # Level 2
        assert risk.current_level == 2

        risk.record_trade("strat_a", 100.0)
        risk.record_trade("strat_a", 100.0)
        risk.record_trade("strat_a", -50.0)  # 리셋
        risk.record_trade("strat_a", 100.0)
        assert risk.current_level == 2  # 복구 안됨

    def test_recovery_one_step_at_a_time(self) -> None:
        """복구는 한 단계씩."""
        risk = PortfolioRiskManager(_make_config())

        risk.get_position_scale(92000.0, 100000.0)  # Level 3
        assert risk.current_level == 3

        # 3승 → Level 2
        for _ in range(3):
            risk.record_trade("strat_a", 100.0)
        assert risk.current_level == 2

        # 3승 더 → Level 1
        for _ in range(3):
            risk.record_trade("strat_a", 100.0)
        assert risk.current_level == 1

    def test_no_recovery_at_level0(self) -> None:
        """Level 0에서는 복구 안함."""
        risk = PortfolioRiskManager(_make_config())
        assert risk.current_level == 0

        for _ in range(10):
            risk.record_trade("strat_a", 100.0)
        assert risk.current_level == 0


class TestCombinedScaling:
    """통합 스케일링 테스트."""

    def test_portfolio_and_strategy_combined(self) -> None:
        """포트폴리오 scale 0.75 × 전략 scale 0.5 = 최종 0.375."""
        risk = PortfolioRiskManager(_make_config())

        # portfolio scale = 0.75 (Level 1, -4%)
        p_scale = risk.get_position_scale(96000.0, 100000.0)
        assert p_scale == 0.75

        # strategy scale = 0.5 (Rolling PF weak)
        for _ in range(8):
            risk.record_trade("strat_a", 100.0)
        for _ in range(12):
            risk.record_trade("strat_a", -100.0)
        s_scale = risk.get_strategy_scale("strat_a")
        assert s_scale == 0.50

        # 최종: 0.75 × 0.50 = 0.375
        # (실제 곱셈은 PortfolioManager.allocate()에서 수행)
        assert p_scale * s_scale == pytest.approx(0.375)


class TestSerialization:
    """상태 직렬화/역직렬화 테스트."""

    def test_basic_round_trip(self) -> None:
        """기본 직렬화/역직렬화."""
        risk = PortfolioRiskManager(_make_config())
        risk.record_trade("strat_a", 100.0)
        risk.record_trade("strat_a", -50.0)

        state = risk.to_dict()

        new_risk = PortfolioRiskManager(_make_config())
        new_risk.from_dict(state)

        assert "strat_a" in new_risk.strategy_stats
        assert new_risk.strategy_stats["strat_a"]["total_trades"] == 2
        assert new_risk.strategy_stats["strat_a"]["profit_factor"] == 2.0

    def test_recent_trades_preserved(self) -> None:
        """recent_trades가 직렬화/역직렬화 후 유지."""
        risk = PortfolioRiskManager(_make_config())
        risk.record_trade("strat_a", 100.0)
        risk.record_trade("strat_a", -50.0)
        risk.record_trade("strat_a", 200.0)

        state = risk.to_dict()

        new_risk = PortfolioRiskManager(_make_config())
        new_risk.from_dict(state)

        assert new_risk.strategy_stats["strat_a"]["recent_trades"] == [
            100.0, -50.0, 200.0
        ]

    def test_level_state_preserved(self) -> None:
        """방어 레벨 상태가 직렬화/역직렬화 후 유지."""
        risk = PortfolioRiskManager(_make_config())
        risk.get_position_scale(94000.0, 100000.0)  # Level 2

        state = risk.to_dict()
        assert state["current_level"] == 2
        assert state["level_entered_at"] is not None

        new_risk = PortfolioRiskManager(_make_config())
        new_risk.from_dict(state)
        assert new_risk.current_level == 2

    def test_consecutive_wins_preserved(self) -> None:
        """연속 승리 카운트가 직렬화/역직렬화 후 유지."""
        risk = PortfolioRiskManager(_make_config())
        risk.get_position_scale(94000.0, 100000.0)  # Level 2
        risk.record_trade("strat_a", 100.0)
        risk.record_trade("strat_a", 100.0)

        state = risk.to_dict()
        assert state["consecutive_wins"] == 2

        new_risk = PortfolioRiskManager(_make_config())
        new_risk.from_dict(state)
        assert new_risk._consecutive_wins == 2


class TestBackwardCompatibility:
    """하위 호환성 테스트."""

    def test_minimal_config(self) -> None:
        """최소 설정 (기존 형식)으로도 동작."""
        config = {
            "max_portfolio_mdd": -0.10,
            "max_daily_loss": -0.03,
            "strategy_disable_threshold": 0.8,
            "strategy_disable_min_trades": 50,
        }
        risk = PortfolioRiskManager(config)

        # 기존 기능 정상 동작
        ok, _ = risk.check_portfolio(95000.0, 100000.0)
        assert ok is True

        assert risk.check_strategy_health("unknown") is True

        # 스케일링 비활성화 → 항상 1.0
        assert risk.get_position_scale(89000.0, 100000.0) == 1.0
        assert risk.get_strategy_scale("unknown") == 1.0

        # 일일 손실 체크는 config 값으로 동작
        ok, _ = risk.check_daily_loss(-4000.0, 100000.0)
        assert ok is False

    def test_from_dict_without_new_fields(self) -> None:
        """기존 저장 데이터 (new fields 없음)에서 복원."""
        risk = PortfolioRiskManager(_make_config())
        old_state = {
            "strategy_stats": {
                "strat_a": {
                    "total_trades": 10,
                    "wins": 7,
                    "losses": 3,
                    "gross_profit": 700.0,
                    "gross_loss": 300.0,
                    "profit_factor": 2.33,
                }
            }
        }
        risk.from_dict(old_state)
        assert risk.current_level == 0
        assert risk._consecutive_wins == 0
