"""포트폴리오 합산 백테스트.

여러 전략의 시그널을 조합한 합산 백테스트를 실행한다.
단일 전략만 있을 경우 기존 backtest.py와 동일한 결과를 보여야 한다.

사용법:
    python portfolio_backtest.py
    python portfolio_backtest.py --strategies btc_1h_momentum,eth_1h_momentum
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.utils.logger import setup_logger

logger = setup_logger("portfolio_backtest")


def load_portfolio_config() -> dict:
    """config/portfolio.yaml 로드.

    Returns:
        portfolio 섹션 딕셔너리.
    """
    with open("config/portfolio.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("portfolio", raw)


def load_strategy_instance(strategy_name: str):
    """전략 인스턴스 및 설정 로드.

    Args:
        strategy_name: 전략 폴더명.

    Returns:
        (strategy_instance, config_dict) 튜플.
    """
    from src.portfolio.manager import PortfolioManager
    import importlib
    import inspect
    from src.strategies.base import BaseStrategy

    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    module = importlib.import_module(f"strategies.{strategy_name}.strategy")
    strategy_class = None
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            strategy_class = obj
            break

    if strategy_class is None:
        raise ValueError(f"BaseStrategy 서브클래스를 찾을 수 없음: {strategy_name}")

    strategy = strategy_class(config=config.get("params", {}))
    return strategy, config


def run_portfolio_backtest(strategy_names: list[str]) -> None:
    """포트폴리오 합산 백테스트 실행.

    Args:
        strategy_names: 전략 이름 리스트.
    """
    import vectorbt as vbt
    from src.analytics.reporter import Reporter

    portfolio_config = load_portfolio_config()
    position_pct = portfolio_config.get("allocation", {}).get(
        "position_pct_per_strategy", 0.20
    )
    max_symbol_exposure = portfolio_config.get("limits", {}).get(
        "max_symbol_exposure", 0.30
    )

    # 전략별 데이터 로드 및 시그널 생성
    strategy_results: list[dict] = []

    for name in strategy_names:
        logger.info(f"전략 로드: {name}")
        strategy, config = load_strategy_instance(name)

        symbol = config["strategy"]["symbol"]
        timeframe = config["strategy"]["timeframe"]

        data_path = f"data/processed/{symbol}_{timeframe}_features.parquet"
        if not os.path.exists(data_path):
            logger.error(f"데이터 파일 없음: {data_path} — 전략 {name} 스킵")
            continue

        df = pd.read_parquet(data_path)
        logger.info(f"  {name}: {symbol} {timeframe} — {len(df)}행")

        # 벡터화 시그널 생성
        signal_series, prob_series = strategy.generate_signals_vectorized(df)

        sl_pct = config.get("risk", {}).get("stop_loss_pct")
        tp_pct = config.get("risk", {}).get("take_profit_pct")
        fee_rate = config.get("execution", {}).get("fee_rate", 0.00055)
        max_position_pct = config.get("risk", {}).get("max_position_pct", position_pct)

        strategy_results.append({
            "name": name,
            "symbol": symbol,
            "timeframe": timeframe,
            "df": df,
            "signals": signal_series,
            "probabilities": prob_series,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "fee_rate": fee_rate,
            "max_position_pct": max_position_pct,
            "config": config,
        })

    if not strategy_results:
        logger.error("활성 전략 없음 — 종료")
        sys.exit(1)

    # === 전략별 개별 백테스트 ===
    print("\n" + "=" * 70)
    print("포트폴리오 백테스트 결과")
    print("=" * 70)

    strategy_portfolios = []
    strategy_metrics_list = []

    for result in strategy_results:
        df = result["df"]
        signal_series = result["signals"]

        portfolio_kwargs = dict(
            close=df["close"],
            entries=(signal_series == 1),
            exits=pd.Series(False, index=signal_series.index),
            fees=result["fee_rate"],
            slippage=0.002,
            init_cash=1_000_000,
            size=result["max_position_pct"],
            size_type="percent",
            sl_stop=result["sl_pct"],
            tp_stop=result["tp_pct"],
        )

        pf = vbt.Portfolio.from_signals(**portfolio_kwargs)
        strategy_portfolios.append(pf)

        # 거래 기반 지표
        trades = pf.trades.records_readable
        total_trades = len(trades)
        if total_trades > 0:
            winning = (trades["PnL"] > 0).sum()
            win_rate = float(winning / total_trades)
            gross_profit = float(trades.loc[trades["PnL"] > 0, "PnL"].sum())
            gross_loss = float(abs(trades.loc[trades["PnL"] < 0, "PnL"].sum()))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            win_rate = 0.0
            profit_factor = 0.0

        returns = pf.returns()
        total_return = float((1 + returns).prod() - 1)
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        dd = (cum_returns - peak) / peak
        max_dd = float(dd.min())

        metrics = {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }
        strategy_metrics_list.append(metrics)

        print(f"\n--- {result['name']} ({result['symbol']} {result['timeframe']}) ---")
        print(f"  총 수익률: {total_return:+.4%}")
        print(f"  MDD: {max_dd:.4%}")
        print(f"  거래 수: {total_trades}")
        print(f"  승률: {win_rate:.1%}")
        print(f"  Profit Factor: {profit_factor:.2f}")

    # === 포트폴리오 합산 (수익률 가중 평균) ===
    if len(strategy_results) > 1:
        print("\n" + "=" * 70)
        print("포트폴리오 합산 결과")
        print("=" * 70)

        # 전략별 수익률 시리즈를 시간축으로 정렬하여 합산
        # 각 전략의 자본 비중 = position_pct
        all_returns = []
        for i, result in enumerate(strategy_results):
            pf = strategy_portfolios[i]
            ret = pf.returns()
            # 전략 비중 적용
            weight = result["max_position_pct"]
            weighted_ret = ret * weight
            all_returns.append(weighted_ret)

        # 공통 인덱스에서 합산
        combined = pd.concat(all_returns, axis=1).fillna(0)
        portfolio_returns = combined.sum(axis=1)

        total_return = float((1 + portfolio_returns).prod() - 1)
        cum_returns = (1 + portfolio_returns).cumprod()
        peak = cum_returns.cummax()
        dd = (cum_returns - peak) / peak
        max_dd = float(dd.min())

        print(f"  합산 수익률: {total_return:+.4%}")
        print(f"  합산 MDD: {max_dd:.4%}")

        # 전략 간 상관관계
        if len(all_returns) >= 2:
            ret_df = pd.concat(
                [pf.returns() for pf in strategy_portfolios],
                axis=1,
                keys=[r["name"] for r in strategy_results],
            ).dropna()
            if len(ret_df) > 1:
                corr_matrix = ret_df.corr()
                print("\n  전략 간 상관관계:")
                print(corr_matrix.to_string(float_format=lambda x: f"{x:.3f}"))

        # 분산 효과 확인
        individual_max_dd = max(
            abs(m["max_drawdown"]) for m in strategy_metrics_list
        )
        print(f"\n  개별 최대 MDD: {-individual_max_dd:.4%}")
        print(f"  합산 MDD: {max_dd:.4%}")
        if abs(max_dd) < individual_max_dd:
            print("  → 분산 효과 확인")
        else:
            print("  → 분산 효과 없음 (상관관계 높음)")
    else:
        # 단일 전략 — 개별 백테스트와 동일
        print("\n단일 전략 포트폴리오 — 개별 백테스트와 동일한 결과")

    # === 보수적 비용 시나리오 ===
    print("\n" + "=" * 70)
    print("보수적 비용 시나리오 (슬리피지 0.1% 추가)")
    print("=" * 70)

    for result in strategy_results:
        df = result["df"]
        signal_series = result["signals"]

        conservative_kwargs = dict(
            close=df["close"],
            entries=(signal_series == 1),
            exits=pd.Series(False, index=signal_series.index),
            fees=result["fee_rate"],
            slippage=0.003,  # 보수적 슬리피지
            init_cash=1_000_000,
            size=result["max_position_pct"],
            size_type="percent",
            sl_stop=result["sl_pct"],
            tp_stop=result["tp_pct"],
        )

        pf_conservative = vbt.Portfolio.from_signals(**conservative_kwargs)
        trades = pf_conservative.trades.records_readable
        total_trades = len(trades)
        if total_trades > 0:
            winning = (trades["PnL"] > 0).sum()
            win_rate = float(winning / total_trades)
            gross_profit = float(trades.loc[trades["PnL"] > 0, "PnL"].sum())
            gross_loss = float(abs(trades.loc[trades["PnL"] < 0, "PnL"].sum()))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            win_rate = 0.0
            profit_factor = 0.0

        returns = pf_conservative.returns()
        total_return = float((1 + returns).prod() - 1)

        print(f"\n  {result['name']}:")
        print(f"    수익률: {total_return:+.4%}")
        print(f"    거래 수: {total_trades}")
        print(f"    PF: {profit_factor:.2f}")


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="bybit-quant 포트폴리오 백테스트")
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="전략 이름 (쉼표 구분, 미지정 시 portfolio.yaml)",
    )
    args = parser.parse_args()
    load_dotenv("config/.env")

    if args.strategies:
        strategy_names = [s.strip() for s in args.strategies.split(",")]
    else:
        portfolio_config = load_portfolio_config()
        strategy_names = portfolio_config.get("active_strategies", [])

    if not strategy_names:
        logger.error("활성 전략 없음 — portfolio.yaml 또는 --strategies를 확인하세요")
        sys.exit(1)

    logger.info(f"포트폴리오 백테스트 시작: {', '.join(strategy_names)}")
    run_portfolio_backtest(strategy_names)


if __name__ == "__main__":
    main()
