"""bybit-quant 실거래/백테스트 진입점.

사용법:
    python main.py --mode live                                # 포트폴리오 모드 (portfolio.yaml 전체)
    python main.py --strategy btc_1h_momentum --mode live     # 단일 전략 모드 (하위 호환)
    python main.py --strategy btc_1h_momentum --mode backtest # 백테스트
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
        strategy_name: 전략 폴더명 (예: "btc_1h_momentum").

    Returns:
        초기화된 전략 인스턴스.
    """
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if strategy_name == "btc_1h_momentum":
        from strategies.btc_1h_momentum.strategy import LGBMClassifierStrategy
        return LGBMClassifierStrategy(config=config.get("params", {}))
    elif strategy_name == "eth_1h_momentum":
        from strategies.eth_1h_momentum.strategy import LGBMClassifierStrategy as ETHStrategy
        return ETHStrategy(config=config.get("params", {}))
    elif strategy_name == "btc_1h_mean_reversion":
        from strategies.btc_1h_mean_reversion.strategy import MeanReversionStrategy
        return MeanReversionStrategy(config=config.get("params", {}))
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


def _load_portfolio_config() -> dict:
    """config/portfolio.yaml 로드.

    Returns:
        portfolio 섹션 딕셔너리.
    """
    config_path = "config/portfolio.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("portfolio", raw)


def run_live(strategy_name: str | None = None) -> None:
    """실거래 모드 실행.

    포트폴리오 레이어를 통해 전략을 관리한다.
    --strategy 인자가 있으면 해당 전략만, 없으면 portfolio.yaml 전체.

    Args:
        strategy_name: 전략 폴더명. None이면 portfolio.yaml 전체.
    """
    from src.data.collector import BybitDataCollector
    from src.risk.manager import RiskManager, PnLTracker
    from src.execution.executor import OrderExecutor
    from src.utils.notify import TelegramNotifier
    from src.portfolio.manager import PortfolioManager
    from src.portfolio.risk import PortfolioRiskManager
    from src.portfolio.virtual_position import VirtualPositionTracker

    # 1. 포트폴리오 설정 로드
    portfolio_config = _load_portfolio_config()

    # --strategy 지정 시 해당 전략만 활성화
    if strategy_name:
        portfolio_config["active_strategies"] = [strategy_name]

    # 2. 포트폴리오 매니저 초기화
    portfolio_manager = PortfolioManager(portfolio_config)
    portfolio_manager.load_strategies_from_config(portfolio_config)

    portfolio_risk = PortfolioRiskManager(portfolio_config.get("risk", {}))
    virtual_tracker = VirtualPositionTracker()

    # 3. 인프라 초기화
    collector = BybitDataCollector()
    risk_manager = RiskManager()
    exchange = collector.exchange
    executor = OrderExecutor(exchange)
    notifier = TelegramNotifier()
    pnl_tracker = PnLTracker()

    # 레버리지 설정
    leverage = risk_manager.params["position"]["max_leverage"]
    # 활성 전략의 심볼 집합
    active_symbols: set[str] = set()
    for name in portfolio_manager.get_active_strategies():
        cfg = portfolio_manager.get_strategy_config(name)
        sym_raw = cfg.get("strategy", {}).get("symbol", "BTCUSDT")
        sym = _convert_symbol(sym_raw)
        active_symbols.add(sym)
        try:
            exchange.set_leverage(leverage, sym)
            logger.info(f"레버리지 설정: {leverage}x ({sym})")
        except Exception as e:
            logger.warning(f"레버리지 설정 실패 ({sym}): {e}")

    # 4. 저장된 상태 복원
    saved_state = _load_saved_state()
    if "circuit_breaker" in saved_state:
        risk_manager.circuit_breaker.from_dict(saved_state["circuit_breaker"])
        logger.info(f"CircuitBreaker 상태 복원: {saved_state['circuit_breaker']}")
    if "pnl_tracker" in saved_state:
        pnl_tracker.from_dict(saved_state["pnl_tracker"])
        logger.info(f"PnLTracker 상태 복원: {saved_state['pnl_tracker']}")
    if "virtual_positions" in saved_state:
        virtual_tracker.from_dict(saved_state["virtual_positions"])
        logger.info(f"가상 포지션 복원: {len(virtual_tracker.virtual_positions)}개 전략")
    if "portfolio_risk" in saved_state:
        portfolio_risk.from_dict(saved_state["portfolio_risk"])
        logger.info("포트폴리오 리스크 상태 복원")

    last_processed_bars: dict[str, str] = saved_state.get("last_processed_bars", {})
    last_trade_ids: set = set(saved_state.get("last_trade_ids", []))

    # 5. 거래소 포지션 동기화
    logger.info("시작 시 거래소 포지션 동기화...")
    prev_positions = executor.sync_positions()
    saved_positions = saved_state.get("positions", {})

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

    # 6. 폴링 주기 — 모든 전략 중 가장 짧은 타임프레임 기준
    min_tf_seconds = float("inf")
    for name in portfolio_manager.get_active_strategies():
        cfg = portfolio_manager.get_strategy_config(name)
        tf = cfg.get("strategy", {}).get("timeframe", "1h")
        min_tf_seconds = min(min_tf_seconds, _timeframe_to_seconds(tf))
    poll_interval = max(int(min_tf_seconds) // 6, 30)

    active_names = ", ".join(portfolio_manager.get_active_strategies())
    logger.info(f"실거래 시작: [{active_names}] | 폴링 {poll_interval}초")

    # 7. 포트폴리오 가치 피크 추적
    peak_value = saved_state.get("peak_value", 0.0)

    def _save_current_state() -> None:
        """현재 상태를 atomic write로 저장."""
        executor._save_state(prev_positions, extra_state={
            "circuit_breaker": risk_manager.circuit_breaker.to_dict(),
            "pnl_tracker": pnl_tracker.to_dict(),
            "virtual_positions": virtual_tracker.to_dict(),
            "portfolio_risk": portfolio_risk.to_dict(),
            "peak_value": peak_value,
            "last_processed_bars": last_processed_bars,
            "last_trade_ids": list(last_trade_ids)[-100:],
        })

    while True:
        try:
            # 8. 각 전략별 데이터 수집 + 시그널 수집
            data_dict: dict[str, pd.DataFrame] = {}

            for name in portfolio_manager.get_active_strategies():
                strategy = portfolio_manager.get_strategy(name)
                cfg = portfolio_manager.get_strategy_config(name)
                sym_raw = cfg.get("strategy", {}).get("symbol", "BTCUSDT")
                sym = _convert_symbol(sym_raw)
                tf = cfg.get("strategy", {}).get("timeframe", "1h")

                df = collector.fetch_ohlcv(symbol=sym, timeframe=tf, limit=1000)
                current_bar = str(df["timestamp"].iloc[-1])

                # 봉 중복 체크
                if current_bar == last_processed_bars.get(name):
                    continue

                data_dict[name] = df
                last_processed_bars[name] = current_bar

            if not data_dict:
                time.sleep(poll_interval)
                continue

            # 9. 포지션 동기화 및 청산 감지
            positions = executor.sync_positions()

            for sym, prev_pos in prev_positions.items():
                if sym not in positions:
                    logger.info(f"포지션 청산 감지: {sym}")
                    # 해당 심볼의 가상 포지션도 청산
                    strategies_with_pos = virtual_tracker.get_strategies_for_symbol(sym)
                    try:
                        trades = exchange.fetch_my_trades(sym, limit=5)
                        closed_pnl = _collect_closed_pnl(trades, last_trade_ids)
                        if closed_pnl != 0:
                            risk_manager.circuit_breaker.record_trade(closed_pnl)
                            pnl_tracker.record_pnl(closed_pnl)
                            # PnL을 가상 포지션 보유 전략에 분배 (FIFO: 첫 전략에 할당)
                            if strategies_with_pos:
                                pnl_strategy = strategies_with_pos[0]
                                portfolio_risk.record_trade(pnl_strategy, closed_pnl)
                            executor.record_closed_pnl(
                                symbol=sym,
                                pnl=closed_pnl,
                                strategy_name=strategies_with_pos[0] if strategies_with_pos else "",
                                reason="SL/TP 자동 청산",
                            )
                            logger.info(f"청산 PnL 기록: {closed_pnl:+.2f}")
                    except Exception as e:
                        logger.warning(f"청산 PnL 조회 실패: {e}")
                    # 모든 관련 가상 포지션 청산
                    for strat_name in strategies_with_pos:
                        virtual_tracker.close(strat_name, sym)

            # 10. 시그널 수집
            signals = portfolio_manager.collect_signals(data_dict)

            # 시그널 없는 전략은 로깅
            for name in data_dict:
                sig = signals.get(name, (0, 0.0))
                if sig[0] == 0:
                    logger.info(f"{name}: 신호 중립 — 대기")

            # 매수 시그널이 없으면 상태 저장 후 다음 루프
            if not any(sig == 1 for sig, _ in signals.values()):
                prev_positions = positions
                _save_current_state()
                time.sleep(poll_interval)
                continue

            # 11. 잔고 조회 및 피크 업데이트
            balance = exchange.fetch_balance()
            portfolio_value = float(balance.get("total", {}).get("USDT", 0))
            peak_value = max(peak_value, portfolio_value)

            # 12. 일일 손실 체크
            ok, reason = portfolio_risk.check_daily_loss(
                pnl_tracker.daily_pnl, portfolio_value
            )
            if not ok:
                logger.warning(f"일일 손실 한도: {reason}")
                notifier.send_sync(f"[경고] {reason} — 당일 추가 진입 차단")
                prev_positions = positions
                _save_current_state()
                time.sleep(poll_interval)
                continue

            # 13. 포트폴리오 MDD 스케일링 계수
            portfolio_scale = portfolio_risk.get_position_scale(
                portfolio_value, peak_value
            )
            if portfolio_scale <= 0:
                logger.warning("포트폴리오 MDD 한도 — 전체 차단")
                notifier.send_sync(
                    "[긴급] 포트폴리오 MDD 한도 도달 — 전체 진입 차단"
                )
                prev_positions = positions
                _save_current_state()
                time.sleep(poll_interval)
                continue

            # 14. 전략별 스케일링 계수 (Rolling PF + 누적 PF)
            strategy_scales: dict[str, float] = {}
            for name in signals:
                if not portfolio_risk.check_strategy_health(name):
                    strategy_scales[name] = 0.0
                    logger.warning(f"전략 비활성화 (누적 PF 미달): {name}")
                    notifier.send_sync(f"[경고] {name} 전략 비활성화 (PF 미달)")
                else:
                    strategy_scales[name] = portfolio_risk.get_strategy_scale(name)

            # 활성 전략만 필터
            healthy_signals = {
                name: sig
                for name, sig in signals.items()
                if strategy_scales.get(name, 0) > 0
            }

            # 15. 자본 배분 → 주문 목록 (스케일링 적용)
            orders = portfolio_manager.allocate(
                healthy_signals,
                portfolio_value,
                virtual_tracker,
                portfolio_scale=portfolio_scale,
                strategy_scales=strategy_scales,
            )

            # 16. 각 주문 실행
            for order in orders:
                strat_name = order["strategy"]
                sym = order["symbol"]
                cfg = portfolio_manager.get_strategy_config(strat_name)

                # 데이터에서 진입 가격 가져오기
                df = data_dict.get(strat_name)
                if df is None:
                    continue
                entry_price = float(df["close"].iloc[-1])

                # 기존 포지션 체크
                existing_pos = positions.get(sym)
                if existing_pos:
                    if existing_pos["side"] == "long":
                        logger.info(f"이미 {sym} long 포지션 보유 — 스킵")
                        continue

                # 전략별 리스크 체크
                atr_value = _compute_atr(df)
                current_vol = float(atr_value / entry_price) if atr_value > 0 else 0.0

                ok, reason = risk_manager.check_all(
                    daily_pnl=pnl_tracker.daily_pnl,
                    portfolio_value=portfolio_value,
                    current_positions=len(positions),
                    current_volatility=current_vol,
                    monthly_pnl=pnl_tracker.monthly_pnl,
                )

                if not ok:
                    logger.warning(f"리스크 체크 실패 ({strat_name}): {reason}")
                    notifier.send_sync(f"[경고] 리스크 체크 실패: {reason}")
                    if "Circuit Breaker" in reason:
                        notifier.send_sync(
                            f"[긴급] Circuit Breaker 발동 — 수동 리셋 필요: {reason}"
                        )
                    continue

                # 포지션 사이징
                atr = _compute_atr(df)
                if atr <= 0:
                    logger.warning(f"ATR 계산 불가 ({strat_name}) — 주문 스킵")
                    continue

                position_size = risk_manager.calculate_atr_position_size(
                    portfolio_value=portfolio_value,
                    atr=atr,
                    entry_price=entry_price,
                )

                # SL/TP 계산
                sl_pct = cfg.get("risk", {}).get("stop_loss_pct")
                tp_pct = cfg.get("risk", {}).get("take_profit_pct")
                sl, tp = risk_manager.get_stop_take_profit(
                    entry_price, "long",
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                )

                # 가상 포지션 생성
                virtual_tracker.open(strat_name, sym, "long", position_size, entry_price)

                # 실제 주문: 가상 합산과 실제의 차이만큼
                current_real = positions.get(sym, {})
                deltas = virtual_tracker.get_delta_orders(sym, current_real)

                for delta in deltas:
                    exec_order = executor.execute(
                        symbol=delta["symbol"],
                        side=delta["side"],
                        amount=delta["amount"],
                        order_type=cfg.get("execution", {}).get("order_type", "limit"),
                        price=entry_price,
                        strategy_name=strat_name,
                        signal_score=1,
                        stop_loss=sl,
                        take_profit=tp,
                    )

                    if exec_order:
                        msg = (
                            f"주문 실행: {delta['side'].upper()} "
                            f"{delta['amount']:.4f} {sym} @ {entry_price:.2f} "
                            f"({strat_name})"
                        )
                        logger.info(msg)
                        notifier.send_sync(msg)

            # 17. 상태 업데이트 및 저장
            prev_positions = executor.sync_positions()
            _save_current_state()

        except KeyboardInterrupt:
            logger.info("사용자에 의해 종료")
            try:
                _save_current_state()
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
        default=None,
        help="전략 이름 (미지정 시 portfolio.yaml 전체 로드)",
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
        if not args.strategy:
            logger.error("백테스트 모드에서는 --strategy가 필수입니다.")
            sys.exit(1)
        run_backtest(args.strategy)


if __name__ == "__main__":
    main()
