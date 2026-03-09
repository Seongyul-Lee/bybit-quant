"""bybit-quant 실거래/백테스트 진입점.

사용법:
    python main.py --strategy ma_crossover --mode live
    python main.py --strategy ma_crossover --mode backtest
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils.logger import setup_logger

logger = setup_logger("main")


def load_strategy(strategy_name: str):
    """전략 이름으로 전략 인스턴스를 동적 로드.

    Args:
        strategy_name: 전략 폴더명 (예: "ma_crossover").

    Returns:
        초기화된 전략 인스턴스.
    """
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if strategy_name == "ma_crossover":
        from strategies.ma_crossover.strategy import MACrossoverStrategy
        return MACrossoverStrategy(config=config.get("params", {}))
    elif strategy_name == "lgbm_classifier":
        from strategies.lgbm_classifier.strategy import LGBMClassifierStrategy
        return LGBMClassifierStrategy(config=config.get("params", {}))
    else:
        raise ValueError(f"알 수 없는 전략: {strategy_name}")


def run_live(strategy_name: str) -> None:
    """실거래 모드 실행.

    데이터 수집 → 신호 생성 → 리스크 체크 → 주문 실행 루프.

    Args:
        strategy_name: 전략 폴더명.
    """
    import ccxt
    import pandas as pd
    from src.data.collector import BybitDataCollector
    from src.data.processor import DataProcessor
    from src.risk.manager import RiskManager
    from src.execution.executor import OrderExecutor
    from src.utils.notify import TelegramNotifier

    strategy = load_strategy(strategy_name)
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        strategy_config = yaml.safe_load(f)

    symbol_raw = strategy_config["strategy"]["symbol"]  # e.g. "BTCUSDT"
    symbol = symbol_raw[:3] + "/" + symbol_raw[3:] + ":" + symbol_raw[3:]  # "BTC/USDT:USDT"
    timeframe = strategy_config["strategy"]["timeframe"]

    collector = BybitDataCollector()
    processor = DataProcessor()
    risk_manager = RiskManager()
    exchange = collector.exchange
    executor = OrderExecutor(exchange)
    notifier = TelegramNotifier()

    logger.info(f"실거래 시작: {strategy_name} | {symbol} | {timeframe}")

    while True:
        try:
            # 1. 데이터 수집
            df = collector.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=200)
            df = processor.add_features(df)

            # 2. 신호 생성
            signal = strategy.generate_signal(df)

            if signal == 0:
                logger.info("신호: 중립 — 대기")
                time.sleep(60)
                continue

            # 3. 리스크 체크
            positions = executor.sync_positions()
            balance = exchange.fetch_balance()
            portfolio_value = float(balance.get("total", {}).get("USDT", 0))

            ok, reason = risk_manager.check_all(
                daily_pnl=0,  # TODO: 일일 PnL 추적
                portfolio_value=portfolio_value,
                current_positions=len(positions),
            )

            if not ok:
                logger.warning(f"리스크 체크 실패: {reason}")
                notifier.send_sync(f"[경고] 리스크 체크 실패: {reason}")
                time.sleep(60)
                continue

            # 4. 포지션 사이징
            atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else 100
            position_size = risk_manager.calculate_atr_position_size(
                portfolio_value=portfolio_value,
                atr=atr,
            )

            # 5. 주문 실행
            side = "buy" if signal == 1 else "sell"
            entry_price = float(df["close"].iloc[-1])
            sl, tp = risk_manager.get_stop_take_profit(entry_price, "long" if signal == 1 else "short")

            order = executor.execute(
                symbol=symbol,
                side=side,
                amount=position_size,
                order_type=strategy_config.get("execution", {}).get("order_type", "limit"),
                price=entry_price,
                strategy_name=strategy_name,
                signal_score=signal,
                stop_loss=sl,
                take_profit=tp,
            )

            if order:
                msg = f"주문 실행: {side.upper()} {position_size:.4f} {symbol} @ {entry_price:.2f}"
                logger.info(msg)
                notifier.send_sync(msg)

        except KeyboardInterrupt:
            logger.info("사용자에 의해 종료")
            break
        except Exception as e:
            logger.error(f"실거래 루프 오류: {e}", exc_info=True)

        time.sleep(60)


def run_backtest(strategy_name: str) -> None:
    """백테스트 모드 실행.

    Args:
        strategy_name: 전략 폴더명.
    """
    import pandas as pd
    import vectorbt as vbt
    from src.analytics.reporter import Reporter

    strategy = load_strategy(strategy_name)
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        strategy_config = yaml.safe_load(f)

    symbol = strategy_config["strategy"]["symbol"]
    timeframe = strategy_config["strategy"]["timeframe"]

    data_path = f"data/processed/{symbol}_{timeframe}_features.parquet"
    if not os.path.exists(data_path):
        logger.error(f"데이터 파일 없음: {data_path}")
        logger.info("먼저 데이터를 수집하세요: python -c \"from src.data.collector import ...\"")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info(f"백테스트 데이터 로드: {len(df)}행")

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
    sl_pct = strategy_config.get("risk", {}).get("stop_loss_pct")
    tp_pct = strategy_config.get("risk", {}).get("take_profit_pct")

    sl_stop = sl_pct if sl_pct else None
    tp_stop = tp_pct if tp_pct else None

    # vectorbt 백테스트
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

    print("\n" + "=" * 60)
    print(f"백테스트 결과: {strategy_name}")
    print("=" * 60)
    print(portfolio.stats())

    # 결과 저장
    reporter = Reporter()
    returns = portfolio.returns()
    metrics = reporter.calculate_metrics(returns)
    path = reporter.save_backtest_result(
        strategy_name=type(strategy).__name__,
        params=strategy.get_params(),
        symbol=symbol,
        timeframe=timeframe,
        period_start=str(df["timestamp"].iloc[0]),
        period_end=str(df["timestamp"].iloc[-1]),
        metrics=metrics,
    )
    logger.info(f"결과 저장: {path}")

    # 텔레그램 알림
    from src.utils.notify import TelegramNotifier

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
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(description="bybit-quant 트레이딩 시스템")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="전략 이름 (예: ma_crossover)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "live"],
        required=True,
        help="실행 모드: backtest 또는 live",
    )

    args = parser.parse_args()

    if args.mode == "live":
        run_live(args.strategy)
    elif args.mode == "backtest":
        run_backtest(args.strategy)


if __name__ == "__main__":
    main()
