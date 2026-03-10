"""RiskManager 단위 테스트."""

import os
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src.risk.manager import CircuitBreaker, PnLTracker, RiskManager


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

    def test_serialization(self) -> None:
        """to_dict/from_dict 라운드트립 검증."""
        self.cb.record_trade(-100)
        self.cb.record_trade(-50)
        data = self.cb.to_dict()

        cb2 = CircuitBreaker()
        cb2.from_dict(data)
        self.assertEqual(cb2.consecutive_losses, 2)
        self.assertFalse(cb2.is_tripped)

        # 발동 상태에서도 복원
        self.cb.trip("test")
        data_tripped = self.cb.to_dict()
        cb3 = CircuitBreaker()
        cb3.from_dict(data_tripped)
        self.assertTrue(cb3.is_tripped)
        self.assertEqual(cb3.consecutive_losses, 2)


class TestPnLTracker(unittest.TestCase):
    """PnLTracker 테스트 클래스."""

    def test_record_pnl_accumulates(self) -> None:
        """PnL이 누적되는지 검증."""
        tracker = PnLTracker()
        tracker.record_pnl(100.0)
        tracker.record_pnl(-50.0)
        self.assertAlmostEqual(tracker.daily_pnl, 50.0)
        self.assertAlmostEqual(tracker.monthly_pnl, 50.0)

    def test_daily_reset_on_date_change(self) -> None:
        """날짜 변경 시 일일 PnL 리셋 검증."""
        tracker = PnLTracker()
        tracker.record_pnl(100.0)
        self.assertAlmostEqual(tracker.daily_pnl, 100.0)

        # 어제 날짜로 강제 설정 후 새 기록
        tracker._current_date = "2020-01-01"
        tracker.record_pnl(50.0)
        self.assertAlmostEqual(tracker.daily_pnl, 50.0)  # 리셋 후 50만 남음

    def test_monthly_reset_on_month_change(self) -> None:
        """월 변경 시 월간 PnL 리셋 검증."""
        tracker = PnLTracker()
        tracker.record_pnl(500.0)
        self.assertAlmostEqual(tracker.monthly_pnl, 500.0)

        # 이전 월로 강제 설정 후 새 기록
        tracker._current_month = "2020-01"
        tracker._current_date = "2020-01-31"
        tracker.record_pnl(100.0)
        self.assertAlmostEqual(tracker.monthly_pnl, 100.0)  # 리셋 후 100만 남음
        self.assertAlmostEqual(tracker.daily_pnl, 100.0)  # 일일도 리셋

    def test_serialization_roundtrip(self) -> None:
        """to_dict/from_dict 라운드트립 검증."""
        tracker = PnLTracker()
        tracker.record_pnl(200.0)
        tracker.record_pnl(-50.0)
        data = tracker.to_dict()

        tracker2 = PnLTracker()
        tracker2.from_dict(data)
        self.assertAlmostEqual(tracker2.daily_pnl, 150.0)
        self.assertAlmostEqual(tracker2.monthly_pnl, 150.0)
        self.assertEqual(tracker2._current_date, tracker._current_date)
        self.assertEqual(tracker2._current_month, tracker._current_month)


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

    def test_check_all_monthly_loss(self) -> None:
        """월간 손실 한도 초과 시 거부 검증."""
        ok, reason = self.rm.check_all(
            daily_pnl=0,
            portfolio_value=100_000,
            current_positions=0,
            monthly_pnl=-15_000,  # 15% > 10% 한도
        )
        self.assertFalse(ok)
        self.assertIn("월간 손실 한도", reason)

    def test_check_all_monthly_loss_within_limit(self) -> None:
        """월간 손실이 한도 이내일 때 통과 검증."""
        ok, reason = self.rm.check_all(
            daily_pnl=0,
            portfolio_value=100_000,
            current_positions=0,
            monthly_pnl=-5_000,  # 5% < 10% 한도
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "OK")


    def test_atr_position_size_max_cap_in_quantity(self) -> None:
        """ATR 포지션 사이즈가 max_position_pct 상한을 수량 단위로 적용하는지 검증.

        예시: portfolio $10,000, ATR 300, BTC $90,000
        dollar_risk = 10000 * 0.01 = 100
        position_size = 100 / 300 = 0.333 BTC
        max_size_usd = 10000 * 0.05 = 500
        max_size_qty = 500 / 90000 = 0.00556 BTC
        결과: min(0.333, 0.00556) = 0.00556 → 캡 적용됨
        """
        size = self.rm.calculate_atr_position_size(
            portfolio_value=10_000,
            atr=300,
            entry_price=90_000,
        )
        max_size_qty = 10_000 * 0.05 / 90_000  # 0.00556 BTC
        self.assertAlmostEqual(size, max_size_qty, places=6)
        # 캡이 없었다면 0.333 BTC가 되므로, 캡이 적용되었는지 확인
        uncapped = 10_000 * 0.01 / 300  # 0.333 BTC
        self.assertLess(size, uncapped)

    def test_atr_position_size_below_cap(self) -> None:
        """ATR 포지션 사이즈가 캡 이하일 때 그대로 반환되는지 검증."""
        # portfolio $100,000, ATR 5000, entry_price $50,000
        # dollar_risk = 100000 * 0.01 = 1000
        # position_size = 1000 / 5000 = 0.2
        # max_size_qty = 100000 * 0.05 / 50000 = 0.1
        # min(0.2, 0.1) = 0.1 → 캡 적용
        size = self.rm.calculate_atr_position_size(
            portfolio_value=100_000,
            atr=5000,
            entry_price=50_000,
        )
        expected = min(100_000 * 0.01 / 5000, 100_000 * 0.05 / 50_000)
        self.assertAlmostEqual(size, expected, places=6)

    def test_atr_position_size_entry_price_zero(self) -> None:
        """entry_price가 0 이하일 때 0.0 반환 검증."""
        size = self.rm.calculate_atr_position_size(
            portfolio_value=10_000,
            atr=300,
            entry_price=0,
        )
        self.assertEqual(size, 0.0)

        size_neg = self.rm.calculate_atr_position_size(
            portfolio_value=10_000,
            atr=300,
            entry_price=-100,
        )
        self.assertEqual(size_neg, 0.0)

    def test_atr_position_size_atr_zero(self) -> None:
        """ATR이 0 이하일 때 0.0 반환 검증."""
        size = self.rm.calculate_atr_position_size(
            portfolio_value=10_000,
            atr=0,
            entry_price=90_000,
        )
        self.assertEqual(size, 0.0)


if __name__ == "__main__":
    unittest.main()
