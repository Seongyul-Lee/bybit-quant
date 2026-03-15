"""전략 config 유틸리티."""


def merge_strategy_params(config: dict) -> dict:
    """config['strategy'] 섹션의 핵심 키를 params에 병합.

    strategy 섹션의 symbol, timeframe, name을 params에 주입한다.
    params에 이미 같은 키가 있으면 params 값을 우선한다.

    Args:
        config: 전략 config.yaml 전체 딕셔너리.

    Returns:
        병합된 params 딕셔너리.
    """
    params = dict(config.get("params", {}) or {})
    strategy_section = config.get("strategy", {}) or {}

    # strategy 섹션에서 주입할 키 매핑 (source_key -> target_key)
    key_map = {
        "symbol": "symbol",
        "timeframe": "timeframe",
        "name": "strategy_name",
    }

    for src_key, tgt_key in key_map.items():
        if src_key in strategy_section and tgt_key not in params:
            params[tgt_key] = strategy_section[src_key]

    return params
