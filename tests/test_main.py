"""main.py 헬퍼 함수 테스트."""

import pytest

from main import _convert_symbol


class TestConvertSymbol:
    """_convert_symbol 심볼 변환 테스트."""

    @pytest.mark.parametrize(
        "symbol_raw, expected",
        [
            ("BTCUSDT", "BTC/USDT:USDT"),
            ("ETHUSDT", "ETH/USDT:USDT"),
            ("DOGEUSDT", "DOGE/USDT:USDT"),
            ("1000PEPEUSDT", "1000PEPE/USDT:USDT"),
            ("BTCUSDC", "BTC/USDC:USDC"),
        ],
    )
    def test_valid_symbols(self, symbol_raw: str, expected: str) -> None:
        assert _convert_symbol(symbol_raw) == expected

    def test_invalid_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="알 수 없는 심볼 형식"):
            _convert_symbol("INVALID")
