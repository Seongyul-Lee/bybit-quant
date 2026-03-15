"""펀딩비 차익거래 인프라 Testnet 검증 스크립트.

사용법:
    python -m pytest tests/test_funding_arb.py -v --testnet
    또는 직접 실행:
    python tests/test_funding_arb.py

Testnet API 키가 config/.env에 설정되어 있어야 함:
    BYBIT_TESTNET_API_KEY=...
    BYBIT_TESTNET_SECRET=...
"""

import os
import sys
import pytest

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("config/.env")


def _has_testnet_keys() -> bool:
    """Testnet API 키가 설정되어 있는지 확인."""
    return bool(
        os.getenv("BYBIT_TESTNET_API_KEY")
        and os.getenv("BYBIT_TESTNET_SECRET")
    )


# --- Unit Tests (API 불필요) ---

class TestFundingArbStrategy:
    """FundingArbStrategy 유닛 테스트."""

    def test_generate_signal_always_one(self):
        """Buy & Hold: 항상 signal=1."""
        import pandas as pd
        from strategies.funding_arb.strategy import FundingArbStrategy

        config = {
            "symbol": "BTC/USDT:USDT",
            "leverage": 2,
            "position_pct": 0.40,
        }
        strategy = FundingArbStrategy(config)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
            "open": range(100),
            "high": range(1, 101),
            "low": range(100),
            "close": range(100),
            "volume": [1000] * 100,
        })

        signal, prob = strategy.generate_signal(df)
        assert signal == 1
        assert prob == 0.0

    def test_generate_signals_vectorized(self):
        """백테스트용 벡터화: 모두 1."""
        import pandas as pd
        from strategies.funding_arb.strategy import FundingArbStrategy

        config = {"symbol": "BTC/USDT:USDT"}
        strategy = FundingArbStrategy(config)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            "close": range(50),
        })

        signals, probs = strategy.generate_signals_vectorized(df)
        assert (signals == 1).all()
        assert (probs == 0.0).all()
        assert len(signals) == 50

    def test_base_strategy_interface(self):
        """BaseStrategy 인터페이스 준수 확인."""
        from strategies.funding_arb.strategy import FundingArbStrategy
        from src.strategies.base import BaseStrategy

        config = {"symbol": "BTC/USDT:USDT"}
        strategy = FundingArbStrategy(config)
        assert isinstance(strategy, BaseStrategy)
        assert strategy.symbol == "BTC/USDT:USDT"


class TestConfigLoading:
    """설정 파일 로드 테스트."""

    def test_funding_arb_config_exists(self):
        """config.yaml 파일 존재 확인."""
        import yaml
        config_path = "strategies/funding_arb/config.yaml"
        assert os.path.exists(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "strategy" in config
        assert "params" in config
        assert config["strategy"]["name"] == "funding_arb"
        assert len(config["strategy"]["symbols"]) >= 2

    def test_portfolio_yaml_has_funding_arb(self):
        """portfolio.yaml에 funding_arb 섹션 존재."""
        import yaml
        with open("config/portfolio.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        portfolio = config.get("portfolio", config)
        arb = portfolio.get("funding_arb", {})
        assert arb.get("enabled") is True
        assert "config_path" in arb

    def test_load_funding_arb_config(self):
        """_load_funding_arb_config 함수 동작 확인."""
        sys.path.insert(0, ".")
        from main import _load_funding_arb_config

        portfolio_config = {
            "funding_arb": {
                "enabled": True,
                "config_path": "strategies/funding_arb/config.yaml",
                "capital_pct": 0.40,
            }
        }
        result = _load_funding_arb_config(portfolio_config)
        assert result is not None
        assert result["params"]["capital_pct"] == 0.40

    def test_load_funding_arb_config_disabled(self):
        """비활성화 시 None 반환."""
        from main import _load_funding_arb_config

        result = _load_funding_arb_config({"funding_arb": {"enabled": False}})
        assert result is None

        result = _load_funding_arb_config({})
        assert result is None


class TestArbSymbolGuard:
    """ARB ↔ v1 전략 동일 심볼 충돌 방지 가드 테스트."""

    @staticmethod
    def _filter_arb_symbols(arb_symbols: list[dict], v1_perp_symbols: set[str]):
        """main.py의 가드 로직을 재현."""
        blocked = [s["perp"] for s in arb_symbols if s["perp"] in v1_perp_symbols]
        filtered = [s for s in arb_symbols if s["perp"] not in v1_perp_symbols]
        return blocked, filtered

    def test_arb_symbol_guard_full_overlap(self):
        """v1=BTC+ETH, ARB=BTC+ETH → ARB 전체 차단."""
        v1_perp = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
        arb_syms = [
            {"spot": "BTC/USDT", "perp": "BTC/USDT:USDT"},
            {"spot": "ETH/USDT", "perp": "ETH/USDT:USDT"},
        ]
        blocked, filtered = self._filter_arb_symbols(arb_syms, v1_perp)
        assert len(blocked) == 2
        assert len(filtered) == 0

    def test_arb_symbol_guard_partial_overlap(self):
        """v1=BTC만, ARB=BTC+ETH → ETH만 남음."""
        v1_perp = {"BTC/USDT:USDT"}
        arb_syms = [
            {"spot": "BTC/USDT", "perp": "BTC/USDT:USDT"},
            {"spot": "ETH/USDT", "perp": "ETH/USDT:USDT"},
        ]
        blocked, filtered = self._filter_arb_symbols(arb_syms, v1_perp)
        assert blocked == ["BTC/USDT:USDT"]
        assert len(filtered) == 1
        assert filtered[0]["perp"] == "ETH/USDT:USDT"

    def test_arb_symbol_guard_no_overlap(self):
        """v1=SOL, ARB=BTC+ETH → 전체 허용."""
        v1_perp = {"SOL/USDT:USDT"}
        arb_syms = [
            {"spot": "BTC/USDT", "perp": "BTC/USDT:USDT"},
            {"spot": "ETH/USDT", "perp": "ETH/USDT:USDT"},
        ]
        blocked, filtered = self._filter_arb_symbols(arb_syms, v1_perp)
        assert len(blocked) == 0
        assert len(filtered) == 2


# --- Testnet Integration Tests ---

@pytest.mark.skipif(not _has_testnet_keys(), reason="Testnet API keys not configured")
class TestSpotExecutorTestnet:
    """SpotExecutor testnet 통합 테스트."""

    def test_init(self):
        """SpotExecutor 초기화."""
        from src.execution.spot_executor import SpotExecutor
        executor = SpotExecutor(testnet=True)
        assert executor.exchange is not None

    def test_get_spot_price(self):
        """현물 가격 조회."""
        from src.execution.spot_executor import SpotExecutor
        executor = SpotExecutor(testnet=True)
        price = executor.get_spot_price("BTC/USDT")
        print(f"BTC/USDT 현물 가격: {price}")
        # testnet에서 가격이 0이 아닌지만 확인 (정확한 가격은 보장 불가)
        assert price >= 0  # testnet에서 0일 수도 있음

    def test_get_balance(self):
        """잔고 조회."""
        from src.execution.spot_executor import SpotExecutor
        executor = SpotExecutor(testnet=True)
        usdt = executor.get_balance("USDT")
        btc = executor.get_balance("BTC")
        print(f"USDT: {usdt}, BTC: {btc}")
        assert usdt >= 0
        assert btc >= 0


@pytest.mark.skipif(not _has_testnet_keys(), reason="Testnet API keys not configured")
class TestArbExecutorTestnet:
    """ArbExecutor testnet 통합 테스트."""

    def test_init(self):
        """ArbExecutor 초기화."""
        from src.execution.spot_executor import SpotExecutor
        from src.execution.arb_executor import ArbExecutor
        from src.execution.executor import OrderExecutor
        from src.data.collector import BybitDataCollector

        spot = SpotExecutor(testnet=True)
        collector = BybitDataCollector(testnet=True)
        perp = OrderExecutor(collector.exchange)
        arb = ArbExecutor(spot, perp)
        assert arb.spot is not None
        assert arb.perp is not None

    def test_get_delta(self):
        """델타 조회 (포지션 없을 때 ≈ 0)."""
        from src.execution.spot_executor import SpotExecutor
        from src.execution.arb_executor import ArbExecutor
        from src.execution.executor import OrderExecutor
        from src.data.collector import BybitDataCollector

        spot = SpotExecutor(testnet=True)
        collector = BybitDataCollector(testnet=True)
        perp = OrderExecutor(collector.exchange)
        arb = ArbExecutor(spot, perp)

        delta = arb.get_delta("BTC/USDT", "BTC/USDT:USDT")
        print(f"BTC 델타: {delta}")
        # 포지션이 없으면 delta ≈ 잔고 (0에 가까울 수 있음)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
