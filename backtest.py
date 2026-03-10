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


def run(
    strategy_name: str,
    symbol: str | None = None,
    timeframe: str | None = None,
    allow_short: bool = False,
) -> None:
    """백테스트 실행.

    Args:
        strategy_name: 전략 폴더명 (예: "ma_crossover").
        symbol: 심볼 오버라이드 (None이면 config.yaml 기본값).
        timeframe: 타임프레임 오버라이드 (None이면 config.yaml 기본값).
        allow_short: 매도 신호를 숏 포지션 entry로 활용할지 여부.
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
    max_position_pct = config.get("risk", {}).get("max_position_pct", 0.05)
    fee_rate = config.get("execution", {}).get("fee_rate", 0.00055)

    logger.info(f"SL: {sl_stop}, TP: {tp_stop}, 포지션 비중: {max_position_pct:.0%}, 수수료: {fee_rate:.4%}")

    # vectorbt 포트폴리오
    portfolio_kwargs = dict(
        close=df["close"],
        entries=(signal_series == 1),
        exits=(signal_series == -1),
        fees=fee_rate,
        slippage=0.001,
        init_cash=1_000_000,
        size=max_position_pct,
        size_type="percent",
        sl_stop=sl_stop,
        tp_stop=tp_stop,
    )
    if allow_short:
        portfolio_kwargs["short_entries"] = (signal_series == -1)
        portfolio_kwargs["short_exits"] = (signal_series == 1)
        portfolio_kwargs["upon_opposite_entry"] = "close"
        logger.info("숏 포지션 활성화: 매도 신호를 숏 entry로 사용")

    portfolio = vbt.Portfolio.from_signals(**portfolio_kwargs)

    # 결과 출력
    print("\n" + "=" * 60)
    print(f"백테스트 결과: {strategy_name} | {symbol} | {timeframe}")
    print("=" * 60)
    print(portfolio.stats())

    # 거래 기반 지표 추출
    trades = portfolio.trades.records_readable
    total_trades = len(trades)
    if total_trades > 0:
        winning_trades = (trades["PnL"] > 0).sum()
        trade_win_rate = float(winning_trades / total_trades)
        trade_gross_profit = float(trades.loc[trades["PnL"] > 0, "PnL"].sum())
        trade_gross_loss = float(abs(trades.loc[trades["PnL"] < 0, "PnL"].sum()))
        trade_profit_factor = (
            trade_gross_profit / trade_gross_loss if trade_gross_loss > 0 else float("inf")
        )
    else:
        trade_win_rate = 0.0
        trade_profit_factor = 0.0

    trade_stats = {
        "total_trades": total_trades,
        "win_rate": trade_win_rate,
        "profit_factor": trade_profit_factor,
    }

    # 결과 저장
    reporter = Reporter()
    returns = portfolio.returns()
    metrics = reporter.calculate_metrics(returns, timeframe=timeframe, trade_stats=trade_stats)

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

    # 텔레그램 알림 — 상세 지표 포함
    m = metrics
    benchmark_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    calmar = abs(m["total_return"] / m["max_drawdown"]) if m["max_drawdown"] != 0 else 0.0

    # 거래 상세 추출
    if total_trades > 0:
        best_trade_pct = float(trades["Return"].max() * 100)
        worst_trade_pct = float(trades["Return"].min() * 100)
        avg_win_pct = float(trades.loc[trades["PnL"] > 0, "Return"].mean() * 100)
        avg_loss_pct = float(trades.loc[trades["PnL"] < 0, "Return"].mean() * 100)
        # Duration: Exit Timestamp - Entry Timestamp (봉 단위)
        duration = trades["Exit Timestamp"] - trades["Entry Timestamp"]
        win_mask = trades["PnL"] > 0
        loss_mask = trades["PnL"] < 0
        avg_win_dur = float(duration[win_mask].mean()) if win_mask.any() else 0.0
        avg_loss_dur = float(duration[loss_mask].mean()) if loss_mask.any() else 0.0
        fees_paid = float(portfolio.stats()["Total Fees Paid"])
    else:
        best_trade_pct = worst_trade_pct = 0.0
        avg_win_pct = avg_loss_pct = 0.0
        avg_win_dur = avg_loss_dur = 0.0
        fees_paid = 0.0

    # 모델 품질 지표 (LightGBM 전략 전용)
    model_quality_section = ""
    if strategy_name == "lgbm_classifier":
        import json as _json
        meta_path = "strategies/lgbm_classifier/models/training_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            folds = meta.get("folds", [])
            if folds:
                val_f1s = [fd["val_f1"] for fd in folds]
                gaps = [fd["train_f1"] - fd["val_f1"] for fd in folds]
                overfit_count = sum(1 for g in gaps if g > 0.3)
                avg_val_f1 = sum(val_f1s) / len(val_f1s)
                avg_gap = sum(gaps) / len(gaps)
                conf_threshold = config.get("params", {}).get("confidence_threshold", 0.5)
                model_quality_section = (
                    f"\n*모델 품질 지표*\n"
                    f"Val F1 (macro): `{avg_val_f1:.4f}`\n"
                    f"Train-Val Gap: `{avg_gap:.4f}`\n"
                    f"과적합 fold: `{overfit_count}/{len(folds)}`\n"
                    f"Confidence Threshold: `{conf_threshold}`\n"
                )

    tg_msg = (
        f"*백테스트 성과 지표*\n"
        f"전략: `{strategy_name}` | {symbol} | {timeframe}\n\n"
        f"총 수익률: `{m['total_return']:+.2%}`\n"
        f"벤치마크(B&H): `{benchmark_return:+.2%}`\n"
        f"샤프 비율: `{m['sharpe_ratio']:.2f}`\n"
        f"MDD: `{m['max_drawdown']:.2%}`\n"
        f"Calmar Ratio: `{calmar:.2f}`\n"
        f"승률: `{m['win_rate']:.1%}`\n"
        f"Profit Factor: `{m['profit_factor']:.2f}`\n"
        f"총 거래: `{m['total_trades']}`\n"
        f"{model_quality_section}\n"
        f"*백테스트 상세*\n"
        f"초기 자본: `$1,000,000`\n"
        f"최종 자본: `${1_000_000 * (1 + m['total_return']):,.0f}`\n"
        f"SL: `{sl_stop}` | TP: `{tp_stop}`\n"
        f"수수료: `${fees_paid:,.0f}`\n"
        f"최고 거래: `{best_trade_pct:+.2f}%`\n"
        f"최악 거래: `{worst_trade_pct:+.2f}%`\n"
        f"평균 수익: `{avg_win_pct:+.2f}%`\n"
        f"평균 손실: `{avg_loss_pct:+.2f}%`\n"
        f"평균 수익 기간: `{avg_win_dur:.0f}봉`\n"
        f"평균 손실 기간: `{avg_loss_dur:.0f}봉`\n\n"
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
    parser.add_argument("--allow-short", action="store_true", help="매도 신호를 숏 포지션 entry로 활용")

    args = parser.parse_args()
    load_dotenv("config/.env")
    run(args.strategy, args.symbol, args.timeframe, allow_short=args.allow_short)


if __name__ == "__main__":
    main()
