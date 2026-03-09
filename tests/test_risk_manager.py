"""RiskManager 단위 테스트."""

import os
import tempfile
import unittest

import yaml

from src.risk.manager import CircuitBreaker, RiskManager


class TestCircuitBreaker(unittest.TestCase):
    """CircuitBreaker 테스트 클래스."""

    def setUp(self) -> None:
        self.cb = CircuitBreaker(max_consecutive_losses=3, volatility_threshold=0.05)

    def test_initial_state(self) -> None:
        """초기 상태 검증."""
        self.assertFalse(self.cb.is_tripped)
        self.assertEqual(self.cb.consecutive_losses, 0)

    def test_consecutive_losses_trip(self) -> None:
        """연속 손실 시 Circuit Breaker 발동 검증."""
        self.cb.record_trade(-100)
        self.cb.record_trade(-50)
        self.assertFalse(self.cb.is_tripped)

        self.cb.record_trade(-30)
        self.assertTrue(self.cb.is_tripped)

    def test_win_resets_counter(self) -> None:
        """수익 거래가 연속 손실 카운터를 리셋하는지 검증."""
        self.cb.record_trade(-100)
        self.cb.record_trade(-50)
        self.assertEqual(self.cb.consecutive_losses, 2)

        self.cb.record_trade(200)
        self.assertEqual(self.cb.consecutive_losses, 0)

    def test_volatility_trip(self) -> None:
        """변동성 임계값 초과 시 발동 검증."""
        self.cb.check_volatility(0.03)
        self.assertFalse(self.cb.is_tripped)

        self.cb.check_volatility(0.06)
        self.assertTrue(self.cb.is_tripped)

    def test_manual_reset(self) -> None:
        """수동 리셋 검증."""
        self.cb.trip("테스트")
        self.assertTrue(self.cb.is_tripped)

        self.cb.reset()
        self.assertFalse(self.cb.is_tripped)
        self.assertEqual(self.cb.consecutive_losses, 0)


class TestRiskManager(unittest.TestCase):
    """RiskManager 테스트 클래스."""

    def setUp(self) -> None:
        """임시 리스크 파라미터 파일로 RiskManager 초기화."""
        self.params = {
            "position": {
                "max_position_pct": 0.05,
                "max_concurrent_positions": 3,
                "max_leverage": 3,
            },
            "loss_limits": {
                "daily_loss_limit_pct": 0.03,
                "monthly_loss_limit_pct": 0.10,
            },
            "circuit_breaker": {
                "max_consecutive_losses": 5,
                "volatility_threshold": 0.05,
            },
            "trade": {
                "default_stop_loss_pct": 0.02,
                "default_take_profit_pct": 0.04,
                "risk_per_trade_pct": 0.01,
            },
        }
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(self.params, self.tmp)
        self.tmp.close()
        self.rm = RiskManager(config_path=self.tmp.name)

    def tearDown(self) -> None:
        os.unlink(self.tmp.name)

    def test_calculate_position_size(self) -> None:
        """포지션 사이즈 계산 검증."""
        size = self.rm.calculate_position_size(
            portfolio_value=100_000,
            win_rate=0.55,
            reward_risk_ratio=2.0,
        )
        self.assertGreater(size, 0)
        self.assertLessEqual(size, 100_000 * 0.05)

    def test_position_size_cap(self) -> None:
        """포지션 사이즈가 최대 비율을 초과하지 않는지 검증."""
        size = self.rm.calculate_position_size(
            portfolio_value=100_000,
            win_rate=0.9,
            reward_risk_ratio=10.0,
        )
        self.assertLessEqual(size, 100_000 * 0.05)

    def test_check_all_passes(self) -> None:
        """정상 조건에서 check_all이 통과하는지 검증."""
        ok, reason = self.rm.check_all(
            daily_pnl=100,
            portfolio_value=100_000,
            current_positions=1,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "OK")

    def test_check_all_daily_loss(self) -> None:
        """일일 손실 한도 초과 시 거부 검증."""
        ok, reason = self.rm.check_all(
            daily_pnl=-5_000,
            portfolio_value=100_000,
            current_positions=0,
        )
        self.assertFalse(ok)
        self.assertIn("일일 손실 한도", reason)

    def test_check_all_max_positions(self) -> None:
        """최대 동시 포지션 초과 시 거부 검증."""
        ok, reason = self.rm.check_all(
            daily_pnl=0,
            portfolio_value=100_000,
            current_positions=3,
        )
        self.assertFalse(ok)
        self.assertIn("동시 포지션", reason)

    def test_stop_take_profit_long(self) -> None:
        """롱 포지션 손절/익절 가격 계산 검증."""
        sl, tp = self.rm.get_stop_take_profit(entry_price=40000, side="long")
        self.assertAlmostEqual(sl, 40000 * 0.98)
        self.assertAlmostEqual(tp, 40000 * 1.04)

    def test_stop_take_profit_short(self) -> None:
        """숏 포지션 손절/익절 가격 계산 검증."""
        sl, tp = self.rm.get_stop_take_profit(entry_price=40000, side="short")
        self.assertAlmostEqual(sl, 40000 * 1.02)
        self.assertAlmostEqual(tp, 40000 * 0.96)


if __name__ == "__main__":
    unittest.main()
