"""백테스트 단독 실행 스크립트.

사용법:
    python backtest.py --strategy ma_crossover
    python backtest.py --strategy ma_crossover --symbol BTCUSDT --timeframe 1h
"""

import argparse
import os
import sys

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.utils.logger import setup_logger
from src.utils.notify import TelegramNotifier

logger = setup_logger("backtest")


def run(strategy_name: str, symbol: str | None = None, timeframe: str | None = None) -> None:
    """백테스트 실행.

    Args:
        strategy_name: 전략 폴더명 (예: "ma_crossover").
        symbol: 심볼 오버라이드 (None이면 config.yaml 기본값).
        timeframe: 타임프레임 오버라이드 (None이면 config.yaml 기본값).
    """
    import vectorbt as vbt
    from src.analytics.reporter import Reporter

    # 전략 로드
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    symbol = symbol or config["strategy"]["symbol"]
    timeframe = timeframe or config["strategy"]["timeframe"]

    # 전략 인스턴스 생성
    if strategy_name == "ma_crossover":
        from strategies.ma_crossover.strategy import MACrossoverStrategy
        strategy = MACrossoverStrategy(config=config.get("params", {}))
    elif strategy_name == "lgbm_classifier":
        from strategies.lgbm_classifier.strategy import LGBMClassifierStrategy
        strategy = LGBMClassifierStrategy(config=config.get("params", {}))
    else:
        raise ValueError(f"알 수 없는 전략: {strategy_name}")

    # 데이터 로드
    data_path = f"data/processed/{symbol}_{timeframe}_features.parquet"
    if not os.path.exists(data_path):
        logger.error(f"데이터 파일 없음: {data_path}")
        logger.info("데이터 수집 후 processor를 먼저 실행하세요.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info(f"데이터 로드: {symbol} {timeframe} — {len(df)}행")

    # 신호 생성 (벡터화 우선, fallback으로 루프)
    if hasattr(strategy, "generate_signals_vectorized"):
        signal_series = strategy.generate_signals_vectorized(df)
        logger.info("벡터화 신호 생성 사용")
    else:
        signals = []
        for i in range(len(df)):
            sig = strategy.generate_signal(df.iloc[: i + 1])
            signals.append(sig)
        signal_series = pd.Series(signals, index=df.index)
        logger.info("루프 신호 생성 사용 (fallback)")

    # SL/TP 파라미터
    sl_pct = config.get("risk", {}).get("stop_loss_pct")
    tp_pct = config.get("risk", {}).get("take_profit_pct")

    sl_stop = sl_pct if sl_pct else None
    tp_stop = tp_pct if tp_pct else None

    logger.info(f"SL: {sl_stop}, TP: {tp_stop}")

    # vectorbt 포트폴리오
    portfolio = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=(signal_series == 1),
        exits=(signal_series == -1),
        fees=0.0004,
        slippage=0.001,
        init_cash=1_000_000,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print(f"백테스트 결과: {strategy_name} | {symbol} | {timeframe}")
    print("=" * 60)
    print(portfolio.stats())

    # 결과 저장
    reporter = Reporter()
    returns = portfolio.returns()
    metrics = reporter.calculate_metrics(returns)

    print("\n--- 커스텀 메트릭 ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    path = reporter.save_backtest_result(
        strategy_name=type(strategy).__name__,
        params=strategy.get_params(),
        symbol=symbol,
        timeframe=timeframe,
        period_start=str(df["timestamp"].iloc[0]),
        period_end=str(df["timestamp"].iloc[-1]),
        metrics=metrics,
    )
    print(f"\n결과 저장: {path}")

    # 텔레그램 알림
    total_trades = metrics.get("total_trades", "N/A")
    total_return = metrics.get("total_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    mdd = metrics.get("max_drawdown", 0)
    win_rate = metrics.get("win_rate", 0)

    tg_msg = (
        f"*백테스트 완료*\n"
        f"전략: `{strategy_name}` | {symbol} | {timeframe}\n\n"
        f"총 거래: `{total_trades}`\n"
        f"총 수익률: `{total_return:+.2%}`\n"
        f"승률: `{win_rate:.1%}`\n"
        f"샤프 비율: `{sharpe:.2f}`\n"
        f"MDD: `{mdd:.2%}`\n\n"
        f"SL: `{sl_stop}` | TP: `{tp_stop}`\n"
        f"결과 파일: `{path}`"
    )
    notifier = TelegramNotifier()
    notifier.send_sync(tg_msg)


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="bybit-quant 백테스트")
    parser.add_argument("--strategy", type=str, required=True, help="전략 이름")
    parser.add_argument("--symbol", type=str, default=None, help="심볼 (기본: config.yaml)")
    parser.add_argument("--timeframe", type=str, default=None, help="타임프레임 (기본: config.yaml)")

    args = parser.parse_args()
    load_dotenv("config/.env")
    run(args.strategy, args.symbol, args.timeframe)


if __name__ == "__main__":
    main()
