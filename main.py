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

import pandas as pd
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

    if strategy_name == "btc_1h_momentum":
        from strategies.btc_1h_momentum.strategy import LGBMClassifierStrategy
        return LGBMClassifierStrategy(config=config.get("params", {}))
    else:
        raise ValueError(f"알 수 없는 전략: {strategy_name}")


def _timeframe_to_seconds(timeframe: str) -> int:
    """타임프레임 문자열을 초 단위로 변환.

    Args:
        timeframe: "1m", "5m", "15m", "1h", "4h", "1d" 등.

    Returns:
        타임프레임에 해당하는 초.
    """
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    return value * multipliers.get(unit, 60)


def _load_saved_state() -> dict:
    """current_state.json에서 저장된 상태를 로드.

    Returns:
        저장된 상태 딕셔너리. 파일 없으면 빈 딕셔너리.
    """
    state_path = "config/current_state.json"
    if os.path.exists(state_path):
        try:
            import json
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"상태 파일 로드 실패: {e}")
    return {}


def _convert_symbol(symbol_raw: str) -> str:
    """파일명용 심볼(BTCUSDT)을 ccxt 심볼(BTC/USDT:USDT)로 변환."""
    for quote in ("USDT", "USDC", "BTC"):
        if symbol_raw.endswith(quote):
            base = symbol_raw[:-len(quote)]
            return f"{base}/{quote}:{quote}"
    raise ValueError(f"알 수 없는 심볼 형식: {symbol_raw}")


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """DataFrame에서 ATR을 직접 계산.

    Args:
        df: OHLCV DataFrame (high, low, close 컬럼 필요).
        period: ATR 기간 (기본 14).

    Returns:
        최신 ATR 값. 데이터 부족 시 0.0.
    """
    if len(df) < period + 1:
        return 0.0
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean()
    val = atr_series.iloc[-1]
    if pd.isna(val):
        return 0.0
    return float(val)


def _collect_closed_pnl(trades: list, last_trade_ids: set) -> float:
    """이미 처리한 거래를 제외하고 closedPnl을 합산.

    Args:
        trades: exchange.fetch_my_trades() 결과 리스트.
        last_trade_ids: 이미 처리한 거래 ID 집합 (in-place 업데이트).

    Returns:
        새로 감지된 거래의 closedPnl 합계.
    """
    pnl = 0.0
    for t in trades:
        tid = t["id"]
        if tid in last_trade_ids:
            continue
        cpnl = float(t.get("info", {}).get("closedPnl", 0))
        if cpnl != 0:
            pnl += cpnl
        last_trade_ids.add(tid)
    return pnl


def run_live(strategy_name: str) -> None:
    """실거래 모드 실행.

    데이터 수집 → 신호 생성 → 리스크 체크 → 주문 실행 루프.

    Args:
        strategy_name: 전략 폴더명.
    """
    from src.data.collector import BybitDataCollector
    from src.risk.manager import RiskManager, PnLTracker
    from src.execution.executor import OrderExecutor
    from src.utils.notify import TelegramNotifier

    strategy = load_strategy(strategy_name)
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        strategy_config = yaml.safe_load(f)

    symbol_raw = strategy_config["strategy"]["symbol"]  # e.g. "BTCUSDT"
    symbol = _convert_symbol(symbol_raw)
    timeframe = strategy_config["strategy"]["timeframe"]

    collector = BybitDataCollector()
    risk_manager = RiskManager()
    exchange = collector.exchange
    executor = OrderExecutor(exchange)
    notifier = TelegramNotifier()
    pnl_tracker = PnLTracker()

    # Step 1: 레버리지 설정
    leverage = risk_manager.params["position"]["max_leverage"]
    try:
        exchange.set_leverage(leverage, symbol)
        logger.info(f"레버리지 설정: {leverage}x")
    except Exception as e:
        logger.warning(f"레버리지 설정 실패: {e}")

    # 저장된 상태 복원
    saved_state = _load_saved_state()
    if "circuit_breaker" in saved_state:
        risk_manager.circuit_breaker.from_dict(saved_state["circuit_breaker"])
        logger.info(f"CircuitBreaker 상태 복원: {saved_state['circuit_breaker']}")
    if "pnl_tracker" in saved_state:
        pnl_tracker.from_dict(saved_state["pnl_tracker"])
        logger.info(f"PnLTracker 상태 복원: {saved_state['pnl_tracker']}")
    last_processed_bar = saved_state.get("last_processed_bar")
    last_trade_ids: set = set(saved_state.get("last_trade_ids", []))

    # 시작 시 거래소 포지션 동기화
    logger.info("시작 시 거래소 포지션 동기화...")
    prev_positions = executor.sync_positions()
    saved_positions = saved_state.get("positions", {})

    # 불일치 감지 및 경고
    for sym in set(list(saved_positions.keys()) + list(prev_positions.keys())):
        saved = saved_positions.get(sym)
        actual = prev_positions.get(sym)
        if saved and not actual:
            logger.warning(f"포지션 불일치: {sym} 저장={saved['side']} → 실제=없음")
        elif actual and not saved:
            logger.warning(f"포지션 불일치: {sym} 저장=없음 → 실제={actual['side']}")
        elif saved and actual and saved["side"] != actual["side"]:
            logger.warning(f"포지션 불일치: {sym} 저장={saved['side']} → 실제={actual['side']}")

    logger.info(f"초기 포지션: {len(prev_positions)}개 (거래소 기준)")

    # 폴링 주기 계산
    tf_seconds = _timeframe_to_seconds(timeframe)
    poll_interval = max(tf_seconds // 6, 30)

    logger.info(f"실거래 시작: {strategy_name} | {symbol} | {timeframe} | 폴링 {poll_interval}초")

    while True:
        try:
            # 1. 데이터 수집 (processor 없이 직접 — 전략이 내부에서 피처 계산)
            df = collector.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=1000)

            # 봉 중복 체크: 같은 봉이면 스킵
            current_bar = str(df["timestamp"].iloc[-1])
            if current_bar == last_processed_bar:
                time.sleep(poll_interval)
                continue

            # 2. 포지션 동기화 및 청산 감지
            positions = executor.sync_positions()

            # SL/TP 청산 감지: 이전에 있었는데 현재 없는 포지션
            for sym, prev_pos in prev_positions.items():
                if sym not in positions:
                    logger.info(f"포지션 청산 감지: {sym}")
                    try:
                        trades = exchange.fetch_my_trades(sym, limit=5)
                        closed_pnl = _collect_closed_pnl(trades, last_trade_ids)
                        if closed_pnl != 0:
                            risk_manager.circuit_breaker.record_trade(closed_pnl)
                            pnl_tracker.record_pnl(closed_pnl)
                            executor.record_closed_pnl(
                                symbol=sym,
                                pnl=closed_pnl,
                                strategy_name=strategy_name,
                                reason="SL/TP 자동 청산",
                            )
                            logger.info(f"청산 PnL 기록: {closed_pnl:+.2f}")
                    except Exception as e:
                        logger.warning(f"청산 PnL 조회 실패: {e}")

            # 3. 신호 생성
            signal, prob = strategy.generate_signal(df)

            if signal == 0:
                logger.info("신호: 중립 — 대기")
                last_processed_bar = current_bar
                prev_positions = positions
                # 상태 저장
                executor._save_state(positions, extra_state={
                    "circuit_breaker": risk_manager.circuit_breaker.to_dict(),
                    "pnl_tracker": pnl_tracker.to_dict(),
                    "last_processed_bar": last_processed_bar,
                    "last_trade_ids": list(last_trade_ids)[-100:],
                })
                time.sleep(poll_interval)
                continue

            # 4. 기존 포지션과 신호 방향 비교
            signal_side = "long"  # 2클래스 롱 전용 모델
            existing_pos = positions.get(symbol)

            if existing_pos:
                if existing_pos["side"] == signal_side:
                    logger.info(f"이미 {signal_side} 포지션 보유 — 스킵")
                    last_processed_bar = current_bar
                    prev_positions = positions
                    time.sleep(poll_interval)
                    continue
                else:
                    # 반대 방향 → 기존 청산 먼저
                    logger.info(f"포지션 반전: {existing_pos['side']} → {signal_side}")
                    close_order = executor.close_position(symbol, existing_pos, strategy_name)
                    if close_order:
                        try:
                            trades = exchange.fetch_my_trades(symbol, limit=5)
                            closed_pnl = _collect_closed_pnl(trades, last_trade_ids)
                            if closed_pnl != 0:
                                risk_manager.circuit_breaker.record_trade(closed_pnl)
                                pnl_tracker.record_pnl(closed_pnl)
                                executor.record_closed_pnl(
                                    symbol=symbol,
                                    pnl=closed_pnl,
                                    strategy_name=strategy_name,
                                    reason="반전 청산",
                                )
                        except Exception as e:
                            logger.warning(f"반전 청산 PnL 조회 실패: {e}")
                    # 포지션 수 재조회
                    positions = executor.sync_positions()

            # 5. 리스크 체크
            balance = exchange.fetch_balance()
            portfolio_value = float(balance.get("total", {}).get("USDT", 0))

            # 변동성 계산
            atr_value = _compute_atr(df)
            current_vol = float(atr_value / df["close"].iloc[-1]) if atr_value > 0 else 0.0

            ok, reason = risk_manager.check_all(
                daily_pnl=pnl_tracker.daily_pnl,
                portfolio_value=portfolio_value,
                current_positions=len(positions),
                current_volatility=current_vol,
                monthly_pnl=pnl_tracker.monthly_pnl,
            )

            if not ok:
                logger.warning(f"리스크 체크 실패: {reason}")
                notifier.send_sync(f"[경고] 리스크 체크 실패: {reason}")
                if "Circuit Breaker" in reason:
                    notifier.send_sync(
                        f"[긴급] Circuit Breaker 발동 — 수동 리셋 필요: {reason}"
                    )
                last_processed_bar = current_bar
                prev_positions = positions
                executor._save_state(positions, extra_state={
                    "circuit_breaker": risk_manager.circuit_breaker.to_dict(),
                    "pnl_tracker": pnl_tracker.to_dict(),
                    "last_processed_bar": last_processed_bar,
                    "last_trade_ids": list(last_trade_ids)[-100:],
                })
                time.sleep(poll_interval)
                continue

            # 6. 포지션 사이징
            atr = _compute_atr(df)
            if atr <= 0:
                logger.warning("ATR 계산 불가 — 주문 스킵")
                last_processed_bar = current_bar
                prev_positions = positions
                time.sleep(poll_interval)
                continue
            entry_price = float(df["close"].iloc[-1])
            position_size = risk_manager.calculate_atr_position_size(
                portfolio_value=portfolio_value,
                atr=atr,
                entry_price=entry_price,
            )

            # 7. 주문 실행
            side = "buy" if signal == 1 else "sell"
            sl, tp = risk_manager.get_stop_take_profit(entry_price, signal_side)

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

            last_processed_bar = current_bar
            prev_positions = executor.sync_positions()

            # 상태 저장
            executor._save_state(prev_positions, extra_state={
                "circuit_breaker": risk_manager.circuit_breaker.to_dict(),
                "pnl_tracker": pnl_tracker.to_dict(),
                "last_processed_bar": last_processed_bar,
                "last_trade_ids": list(last_trade_ids)[-100:],
            })

        except KeyboardInterrupt:
            logger.info("사용자에 의해 종료")
            try:
                executor._save_state(prev_positions, extra_state={
                    "circuit_breaker": risk_manager.circuit_breaker.to_dict(),
                    "pnl_tracker": pnl_tracker.to_dict(),
                    "last_processed_bar": last_processed_bar,
                    "last_trade_ids": list(last_trade_ids)[-100:],
                })
                logger.info("종료 전 상태 저장 완료")
            except Exception as e:
                logger.error(f"종료 시 상태 저장 실패: {e}")
            break
        except Exception as e:
            logger.error(f"실거래 루프 오류: {e}", exc_info=True)

        time.sleep(poll_interval)


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
        signal_series, prob_series = strategy.generate_signals_vectorized(df)
        logger.info("벡터화 신호 생성 사용")
    else:
        signals = []
        for i in range(len(df)):
            sig, _ = strategy.generate_signal(df.iloc[: i + 1])
            signals.append(sig)
        signal_series = pd.Series(signals, index=df.index)
        logger.info("루프 신호 생성 사용 (fallback)")

    # SL/TP 파라미터
    sl_pct = strategy_config.get("risk", {}).get("stop_loss_pct")
    tp_pct = strategy_config.get("risk", {}).get("take_profit_pct")

    sl_stop = sl_pct if sl_pct else None
    tp_stop = tp_pct if tp_pct else None
    max_position_pct = strategy_config.get("risk", {}).get("max_position_pct", 0.05)

    # vectorbt 백테스트
    portfolio = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=(signal_series == 1),
        exits=pd.Series(False, index=signal_series.index),  # SL/TP에만 의존
        fees=0.0004,
        slippage=0.002,
        init_cash=1_000_000,
        size=max_position_pct,
        size_type="targetpercent",
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
    metrics = reporter.calculate_metrics(returns, timeframe=timeframe)
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
