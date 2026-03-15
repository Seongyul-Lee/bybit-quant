"""merge_strategy_params 헬퍼 함수 유닛 테스트."""

from src.utils.config import merge_strategy_params


class TestMergeStrategyParams:
    """merge_strategy_params 테스트."""

    def test_symbol_injected_from_strategy(self):
        """strategy.symbol이 params에 주입되는지 확인."""
        config = {
            "strategy": {"symbol": "ETHUSDT", "timeframe": "1h", "name": "eth_1h_momentum"},
            "params": {"threshold": 0.42},
        }
        result = merge_strategy_params(config)
        assert result["symbol"] == "ETHUSDT"
        assert result["timeframe"] == "1h"
        assert result["strategy_name"] == "eth_1h_momentum"
        assert result["threshold"] == 0.42

    def test_params_takes_priority(self):
        """params에 이미 같은 키가 있으면 params 값 우선."""
        config = {
            "strategy": {"symbol": "ETHUSDT", "timeframe": "1h"},
            "params": {"symbol": "BTCUSDT", "threshold": 0.5},
        }
        result = merge_strategy_params(config)
        assert result["symbol"] == "BTCUSDT"  # params 우선

    def test_empty_config(self):
        """빈 config에서도 에러 없이 빈 dict 반환."""
        assert merge_strategy_params({}) == {}

    def test_no_params_section(self):
        """params 섹션 없이 strategy만 있을 때."""
        config = {
            "strategy": {"symbol": "ETHUSDT", "timeframe": "1h"},
        }
        result = merge_strategy_params(config)
        assert result["symbol"] == "ETHUSDT"
        assert result["timeframe"] == "1h"

    def test_no_strategy_section(self):
        """strategy 섹션 없이 params만 있을 때."""
        config = {
            "params": {"threshold": 0.42},
        }
        result = merge_strategy_params(config)
        assert result == {"threshold": 0.42}

    def test_original_config_not_mutated(self):
        """원본 config의 params가 변경되지 않는지 확인."""
        original_params = {"threshold": 0.42}
        config = {
            "strategy": {"symbol": "ETHUSDT"},
            "params": original_params,
        }
        result = merge_strategy_params(config)
        assert "symbol" not in original_params  # 원본 미변경
        assert result["symbol"] == "ETHUSDT"

    def test_name_mapped_to_strategy_name(self):
        """strategy.name이 strategy_name으로 매핑되는지 확인."""
        config = {
            "strategy": {"name": "btc_1h_momentum"},
            "params": {},
        }
        result = merge_strategy_params(config)
        assert result["strategy_name"] == "btc_1h_momentum"
        assert "name" not in result
