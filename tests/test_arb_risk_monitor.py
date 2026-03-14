"""ArbRiskMonitor 유닛 테스트.

사용법:
    python -m pytest tests/test_arb_risk_monitor.py -v
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.arb_monitor import ArbRiskMonitor


class TestCheckBasis:
    """check_basis() 테스트."""

    def test_normal_basis(self):
        """정상 베이시스: level=normal."""
        monitor = ArbRiskMonitor({})
        result = monitor.check_basis(100_000, 100_050)
        assert result["level"] == "normal"
        assert abs(result["basis_pct"] - 0.0005) < 1e-6

    def test_warn_basis(self):
        """경고 베이시스: ±1% 초과."""
        monitor = ArbRiskMonitor({"basis_warn_pct": 0.01})
        result = monitor.check_basis(100_000, 101_100)
        assert result["level"] == "warn"
        assert "경고" in result["message"]

    def test_critical_basis(self):
        """긴급 베이시스: ±2% 초과."""
        monitor = ArbRiskMonitor({"basis_critical_pct": 0.02})
        result = monitor.check_basis(100_000, 102_500)
        assert result["level"] == "critical"
        assert "긴급" in result["message"]

    def test_negative_basis_critical(self):
        """음수 베이시스 긴급 (현물 > 선물)."""
        monitor = ArbRiskMonitor({"basis_critical_pct": 0.02})
        result = monitor.check_basis(100_000, 97_000)
        assert result["level"] == "critical"

    def test_zero_spot_price(self):
        """현물 가격 0일 때 normal 반환."""
        monitor = ArbRiskMonitor({})
        result = monitor.check_basis(0, 100_000)
        assert result["level"] == "normal"
        assert result["basis_pct"] == 0

    def test_basis_history_accumulates(self):
        """베이시스 이력이 쌓이는지 확인."""
        monitor = ArbRiskMonitor({})
        for i in range(5):
            monitor.check_basis(100_000, 100_000 + i * 100)
        assert len(monitor._basis_history["BTC"]) == 5

    def test_basis_history_capped_at_100(self):
        """베이시스 이력이 100개로 제한."""
        monitor = ArbRiskMonitor({})
        for i in range(120):
            monitor.check_basis(100_000, 100_000 + i)
        assert len(monitor._basis_history["BTC"]) == 100


class TestCheckMargin:
    """check_margin() 테스트."""

    def test_no_positions(self):
        """포지션 없을 때 unknown."""
        class MockExchange:
            def fetch_positions(self, symbols):
                return []

        monitor = ArbRiskMonitor({})
        result = monitor.check_margin(MockExchange(), "BTC/USDT:USDT")
        assert result["level"] == "unknown"

    def test_normal_margin(self):
        """정상 마진."""
        class MockExchange:
            def fetch_positions(self, symbols):
                return [{
                    "marginRatio": 0.1,
                    "info": {"positionMM": "100", "positionBalance": "1000"},
                }]

        monitor = ArbRiskMonitor({})
        result = monitor.check_margin(MockExchange(), "BTC/USDT:USDT")
        assert result["level"] == "normal"
        assert result["margin_ratio"] == 0.1  # 100/1000

    def test_warn_margin(self):
        """마진 경고 (사용률 > 50%)."""
        class MockExchange:
            def fetch_positions(self, symbols):
                return [{
                    "marginRatio": 0,
                    "info": {"positionMM": "600", "positionBalance": "1000"},
                }]

        monitor = ArbRiskMonitor({"margin_warn_pct": 0.50})
        result = monitor.check_margin(MockExchange(), "BTC/USDT:USDT")
        assert result["level"] == "warn"

    def test_critical_margin(self):
        """마진 긴급 (사용률 > 70%)."""
        class MockExchange:
            def fetch_positions(self, symbols):
                return [{
                    "marginRatio": 0,
                    "info": {"positionMM": "800", "positionBalance": "1000"},
                }]

        monitor = ArbRiskMonitor({"margin_critical_pct": 0.30})
        result = monitor.check_margin(MockExchange(), "BTC/USDT:USDT")
        assert result["level"] == "critical"
        assert "청산 임박" in result["message"]

    def test_exception_handling(self):
        """API 오류 시 unknown."""
        class MockExchange:
            def fetch_positions(self, symbols):
                raise Exception("API error")

        monitor = ArbRiskMonitor({})
        result = monitor.check_margin(MockExchange(), "BTC/USDT:USDT")
        assert result["level"] == "unknown"


class TestCheckFundingTrend:
    """check_funding_trend() 테스트."""

    def test_positive_funding(self):
        """양수 펀딩비 → 카운터 리셋."""
        monitor = ArbRiskMonitor({})
        result = monitor.check_funding_trend(0.001)
        assert result["consecutive_negative"] == 0
        assert result["level"] == "normal"

    def test_consecutive_negative_warn(self):
        """연속 음수 3회 → 경고 (max_consecutive_negative=6의 절반)."""
        monitor = ArbRiskMonitor({"max_consecutive_negative": 6})
        for _ in range(3):
            result = monitor.check_funding_trend(-0.001)
        assert result["level"] == "warn"
        assert result["consecutive_negative"] == 3

    def test_consecutive_negative_critical(self):
        """연속 음수 6회 → 긴급."""
        monitor = ArbRiskMonitor({"max_consecutive_negative": 6})
        for _ in range(6):
            result = monitor.check_funding_trend(-0.001)
        assert result["level"] == "critical"
        assert "48시간" in result["message"]

    def test_reset_on_positive(self):
        """양수 펀딩비 수신 시 카운터 리셋."""
        monitor = ArbRiskMonitor({"max_consecutive_negative": 6})
        for _ in range(4):
            monitor.check_funding_trend(-0.001)
        result = monitor.check_funding_trend(0.001)
        assert result["consecutive_negative"] == 0
        assert result["level"] == "normal"

    def test_cumulative_funding(self):
        """누적 펀딩비 계산."""
        monitor = ArbRiskMonitor({})
        monitor.check_funding_trend(0.01)
        monitor.check_funding_trend(-0.005)
        monitor.check_funding_trend(0.003)
        assert abs(monitor._cumulative_funding["BTC"] - 0.008) < 1e-10


class TestCheckEntrySlippage:
    """check_entry_slippage() 테스트."""

    def test_normal_slippage(self):
        """정상 슬리피지."""
        monitor = ArbRiskMonitor({})
        result = monitor.check_entry_slippage(100_000, 100_010)
        assert result["level"] == "normal"

    def test_warn_slippage(self):
        """슬리피지 경고 (0.5% 초과)."""
        monitor = ArbRiskMonitor({"max_entry_slippage_pct": 0.005})
        # 현물 100000, 선물 99400 → slippage = 0.6%
        result = monitor.check_entry_slippage(100_000, 99_400)
        assert result["level"] == "warn"
        assert "슬리피지" in result["message"]

    def test_zero_spot_price(self):
        """현물 가격 0일 때 unknown."""
        monitor = ArbRiskMonitor({})
        result = monitor.check_entry_slippage(0, 100_000)
        assert result["level"] == "unknown"


class TestCheckAll:
    """check_all() 종합 테스트."""

    def test_all_normal(self):
        """모두 정상이면 빈 리스트."""
        class MockExchange:
            def fetch_positions(self, symbols):
                return [{
                    "marginRatio": 0,
                    "info": {"positionMM": "10", "positionBalance": "1000"},
                }]

        monitor = ArbRiskMonitor({})
        alerts = monitor.check_all(100_000, 100_050, MockExchange(), "BTC/USDT:USDT")
        assert alerts == []

    def test_returns_alerts_only(self):
        """경고/긴급만 반환."""
        class MockExchange:
            def fetch_positions(self, symbols):
                return [{
                    "marginRatio": 0,
                    "info": {"positionMM": "800", "positionBalance": "1000"},
                }]

        monitor = ArbRiskMonitor({"basis_critical_pct": 0.02, "margin_critical_pct": 0.30})
        alerts = monitor.check_all(100_000, 103_000, MockExchange(), "BTC/USDT:USDT")
        assert len(alerts) == 2
        levels = {a["level"] for a in alerts}
        assert "critical" in levels


class TestShouldSendAlert:
    """알림 빈도 제한 테스트."""

    def test_critical_always_sent(self):
        """critical은 항상 전송."""
        monitor = ArbRiskMonitor({})
        assert monitor.should_send_alert("basis", "critical") is True
        assert monitor.should_send_alert("basis", "critical") is True

    def test_warn_cooldown(self):
        """warn은 1시간 쿨다운."""
        monitor = ArbRiskMonitor({})
        monitor._alert_cooldown_sec = 0.1  # 테스트용 짧은 쿨다운

        assert monitor.should_send_alert("basis", "warn") is True
        assert monitor.should_send_alert("basis", "warn") is False
        time.sleep(0.15)
        assert monitor.should_send_alert("basis", "warn") is True

    def test_different_types_independent(self):
        """다른 종류의 알림은 독립적."""
        monitor = ArbRiskMonitor({})
        assert monitor.should_send_alert("basis", "warn") is True
        assert monitor.should_send_alert("margin", "warn") is True
        # 같은 종류는 쿨다운
        assert monitor.should_send_alert("basis", "warn") is False


class TestSerialization:
    """상태 직렬화/복원 테스트."""

    def test_to_dict(self):
        """to_dict 기본."""
        monitor = ArbRiskMonitor({})
        monitor.check_funding_trend(-0.001)
        monitor.check_funding_trend(-0.002)
        monitor.check_basis(100_000, 100_100)

        state = monitor.to_dict()
        btc = state["per_coin"]["BTC"]
        assert btc["consecutive_negative"] == 2
        assert abs(btc["cumulative_funding"] - (-0.003)) < 1e-10
        assert len(btc["basis_history_last10"]) == 1

    def test_from_dict_restores_state(self):
        """from_dict로 상태 복원."""
        monitor = ArbRiskMonitor({})
        state = {
            "per_coin": {
                "BTC": {
                    "consecutive_negative": 4,
                    "cumulative_funding": -0.01,
                    "basis_history_last10": [0.001, -0.002],
                }
            }
        }
        monitor.from_dict(state)

        assert monitor._consecutive_negative["BTC"] == 4
        assert monitor._cumulative_funding["BTC"] == -0.01
        assert len(monitor._basis_history["BTC"]) == 2

    def test_roundtrip(self):
        """직렬화 → 복원 라운드트립."""
        m1 = ArbRiskMonitor({})
        for i in range(5):
            m1.check_funding_trend(-0.001)
        m1.check_basis(100_000, 100_500)

        state = m1.to_dict()

        m2 = ArbRiskMonitor({})
        m2.from_dict(state)

        assert m2._consecutive_negative["BTC"] == m1._consecutive_negative["BTC"]
        assert m2._cumulative_funding["BTC"] == m1._cumulative_funding["BTC"]
        assert m2._basis_history["BTC"] == m1._basis_history["BTC"]


class TestPerCoinState:
    """코인별 상태 독립성 테스트."""

    def test_basis_per_coin_isolated(self):
        """BTC/ETH 베이시스가 독립."""
        monitor = ArbRiskMonitor({})
        monitor.check_basis(100_000, 100_100, coin="BTC")
        monitor.check_basis(3_000, 3_010, coin="ETH")
        monitor.check_basis(100_000, 100_200, coin="BTC")

        assert len(monitor._basis_history["BTC"]) == 2
        assert len(monitor._basis_history["ETH"]) == 1

    def test_funding_trend_per_coin_isolated(self):
        """BTC 연속음수가 ETH 양수로 리셋 안 됨."""
        monitor = ArbRiskMonitor({"max_consecutive_negative": 6})
        for _ in range(4):
            monitor.check_funding_trend(-0.001, coin="BTC")
        monitor.check_funding_trend(0.01, coin="ETH")

        assert monitor._consecutive_negative["BTC"] == 4
        assert monitor._consecutive_negative["ETH"] == 0

    def test_cumulative_per_coin(self):
        """코인별 누적 펀딩비 독립."""
        monitor = ArbRiskMonitor({})
        monitor.check_funding_trend(0.01, coin="BTC")
        monitor.check_funding_trend(-0.005, coin="ETH")

        assert abs(monitor._cumulative_funding["BTC"] - 0.01) < 1e-10
        assert abs(monitor._cumulative_funding["ETH"] - (-0.005)) < 1e-10

    def test_serialization_multi_coin(self):
        """멀티코인 직렬화/복원."""
        m1 = ArbRiskMonitor({})
        m1.check_funding_trend(-0.001, coin="BTC")
        m1.check_funding_trend(0.002, coin="ETH")
        m1.check_basis(100_000, 100_100, coin="BTC")
        m1.check_basis(3_000, 3_010, coin="ETH")

        state = m1.to_dict()
        assert "BTC" in state["per_coin"]
        assert "ETH" in state["per_coin"]

        m2 = ArbRiskMonitor({})
        m2.from_dict(state)

        assert m2._consecutive_negative["BTC"] == 1
        assert m2._consecutive_negative["ETH"] == 0
        assert abs(m2._cumulative_funding["ETH"] - 0.002) < 1e-10


class TestLegacyDeserialization:
    """레거시 상태 복원 테스트."""

    def test_legacy_format_loads_as_btc(self):
        """이전 형식(per_coin 키 없음)이 BTC로 복원."""
        monitor = ArbRiskMonitor({})
        legacy_state = {
            "consecutive_negative": 3,
            "cumulative_funding": -0.005,
            "basis_history_last10": [0.001, 0.002],
        }
        monitor.from_dict(legacy_state)

        assert monitor._consecutive_negative["BTC"] == 3
        assert monitor._cumulative_funding["BTC"] == -0.005
        assert len(monitor._basis_history["BTC"]) == 2


class TestCumulativeLossCheck:
    """누적 손실 체크 테스트."""

    def test_cumulative_loss_critical(self):
        """누적 손실 초과 시 critical."""
        monitor = ArbRiskMonitor({"max_cumulative_loss_pct": 0.03, "max_consecutive_negative": 100})
        # 양수 펀딩비 사이에 큰 음수 → 연속 음수는 아니지만 누적은 음수
        monitor.check_funding_trend(-0.02)
        monitor.check_funding_trend(0.005)
        result = monitor.check_funding_trend(-0.02)

        assert result["level"] == "critical"
        assert "누적 펀딩비 손실 긴급" in result["message"]
        assert abs(result["cumulative_funding"] - (-0.035)) < 1e-10

    def test_cumulative_loss_normal(self):
        """누적 손실 미만 시 normal."""
        monitor = ArbRiskMonitor({"max_cumulative_loss_pct": 0.03})
        monitor.check_funding_trend(-0.01)
        result = monitor.check_funding_trend(0.005)

        assert result["level"] == "normal"
        assert abs(result["cumulative_funding"] - (-0.005)) < 1e-10


class TestExtractFillPrice:
    """ArbExecutor._extract_fill_price() 테스트."""

    def test_average_price(self):
        """average 필드 사용."""
        from src.execution.arb_executor import ArbExecutor
        executor = ArbExecutor.__new__(ArbExecutor)
        assert executor._extract_fill_price({"average": 100.5, "price": 100.0}) == 100.5

    def test_fallback_to_price(self):
        """average 없으면 price 사용."""
        from src.execution.arb_executor import ArbExecutor
        executor = ArbExecutor.__new__(ArbExecutor)
        assert executor._extract_fill_price({"average": None, "price": 100.0}) == 100.0

    def test_empty_order(self):
        """빈 주문이면 0.0."""
        from src.execution.arb_executor import ArbExecutor
        executor = ArbExecutor.__new__(ArbExecutor)
        assert executor._extract_fill_price({}) == 0.0
        assert executor._extract_fill_price(None) == 0.0
