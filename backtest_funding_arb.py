"""펀딩비 차익거래 백테스트.

현물 매수 + 선물 숏의 델타 중립 포지션에서 펀딩비를 수취하는
정적 전략을 시뮬레이션한다.

사용법:
    python backtest_funding_arb.py
    python backtest_funding_arb.py --symbol ETH
    python backtest_funding_arb.py --strategy buy_and_hold
"""

import argparse
from typing import Callable

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 구간 정의
# ─────────────────────────────────────────────
PERIODS = {
    "In-Sample":       ("2024-01-01", "2025-06-30"),
    "Post-Validation": ("2025-07-01", "2026-01-18"),
    "Strict OOS":      ("2026-01-19", None),
    "Full":            ("2024-01-01", None),
}

LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0]

# v1 포트폴리오 구간별 수익률 (기존 백테스트 결과)
V1_RETURNS = {
    "In-Sample":       15.10,   # %
    "Post-Validation": -0.09,
    "Strict OOS":      -2.71,
    "Full":            12.30,   # 근사치
}


# ─────────────────────────────────────────────
# 전략 함수
# ─────────────────────────────────────────────
def strategy_buy_and_hold(row: pd.Series, state: dict) -> dict:
    """시작 시 진입, 끝까지 보유. 가장 단순한 기준선.

    Phase 0 발견 '지속 보유가 정답'의 직접적 구현.
    비용 1회만 발생. 음수 펀딩비 구간도 견딤.
    """
    if state["position"] == "flat":
        return {"action": "open", "position_pct": 1.0}
    return {"action": "hold"}


def strategy_threshold(
    row: pd.Series,
    state: dict,
    entry_threshold: float = 0.00005,
    exit_threshold: float = -0.00005,
) -> dict:
    """펀딩비가 양수일 때 진입, 음수 전환 시 청산.

    음수 구간에서 불필요한 펀딩비 지불을 회피하지만,
    진입/청산 비용이 추가로 발생.
    """
    if state["position"] == "flat":
        if row["funding_rate"] > entry_threshold:
            return {"action": "open", "position_pct": 1.0}
        return {"action": "wait"}
    else:
        if row["funding_rate"] < exit_threshold:
            return {"action": "close"}
        return {"action": "hold"}


def strategy_ma_filter(
    row: pd.Series,
    state: dict,
    entry_ma_field: str = "fr_ma_3",
    entry_threshold: float = 0.00003,
    exit_ma_field: str = "fr_ma_7",
    exit_threshold: float = 0.0,
) -> dict:
    """펀딩비 이동평균으로 추세 판단.

    MA3(24h) > threshold 시 진입, MA7(56h) < 0 시 청산.
    단기 양수 스파이크에 속지 않고, 추세적으로 양수인 구간만 참여.
    """
    if state["position"] == "flat":
        ma_val = row.get(entry_ma_field)
        if ma_val is not None and not np.isnan(ma_val) and ma_val > entry_threshold:
            return {"action": "open", "position_pct": 1.0}
        return {"action": "wait"}
    else:
        ma_val = row.get(exit_ma_field)
        if ma_val is not None and not np.isnan(ma_val) and ma_val < exit_threshold:
            return {"action": "close"}
        return {"action": "hold"}


STRATEGIES = {
    "buy_and_hold": strategy_buy_and_hold,
    "threshold": strategy_threshold,
    "ma_filter": strategy_ma_filter,
}


# ─────────────────────────────────────────────
# 시뮬레이션 엔진
# ─────────────────────────────────────────────
def simulate_funding_arb(
    df: pd.DataFrame,
    strategy_fn: Callable,
    initial_capital: float = 10000.0,
    leverage: float = 1.0,
    spot_fee: float = 0.001,
    perp_fee: float = 0.00055,
    slippage: float = 0.0005,
) -> dict:
    """펀딩비 차익거래 시뮬레이션.

    상태 머신:
      FLAT → OPEN: 현물 매수 + 선물 숏 동시 진입
      OPEN → OPEN: 펀딩비 수취/지불, 포지션 유지
      OPEN → FLAT: 현물 매도 + 선물 숏 청산

    수익 계산 (각 8시간 구간):
      포지션 OPEN 상태에서:
        funding_pnl = position_size * funding_rate * leverage
        (양수 FR → 숏이 수취, 음수 FR → 숏이 지불)

    비용 계산:
      진입 비용 = position_size * (spot_fee + perp_fee + slippage * 2)
      청산 비용 = position_size * (spot_fee + perp_fee + slippage * 2)

    Args:
        df: 8h 피처 데이터셋 (funding_rate 컬럼 필수).
        strategy_fn: callable(row, state) → action dict.
        initial_capital: 초기 자본 (USDT).
        leverage: 선물 레버리지.
        spot_fee: 현물 편도 수수료.
        perp_fee: 선물 편도 수수료.
        slippage: 편도 슬리피지.

    Returns:
        결과 딕셔너리 (메트릭 + 거래 로그 + 자본 곡선).
    """
    df = df.copy().reset_index(drop=True)

    capital = initial_capital
    position_size = 0.0
    entry_cost_rate = spot_fee + perp_fee + slippage * 2
    exit_cost_rate = spot_fee + perp_fee + slippage * 2

    state = {"position": "flat"}

    # 기록용
    equity_list = []
    timestamps = []
    trade_log = []
    total_funding_collected = 0.0
    total_funding_paid = 0.0
    total_costs = 0.0
    entry_idx = None

    for i, row in df.iterrows():
        ts = row["timestamp"]
        fr = row["funding_rate"]

        # 펀딩비 수취/지불 (포지션 보유 중일 때)
        if state["position"] == "open" and not np.isnan(fr):
            funding_pnl = position_size * fr * leverage
            if funding_pnl >= 0:
                total_funding_collected += funding_pnl
            else:
                total_funding_paid += abs(funding_pnl)
            capital += funding_pnl

        # 전략 결정
        action = strategy_fn(row, state)

        if action["action"] == "open" and state["position"] == "flat":
            pct = action.get("position_pct", 1.0)
            position_size = capital * pct
            cost = position_size * entry_cost_rate
            capital -= cost
            total_costs += cost
            state["position"] = "open"
            entry_idx = i
            trade_log.append({
                "timestamp": ts,
                "action": "open",
                "position_size": position_size,
                "cost": cost,
                "capital": capital,
            })

        elif action["action"] == "close" and state["position"] == "open":
            cost = position_size * exit_cost_rate
            capital -= cost
            total_costs += cost
            hold_periods = i - entry_idx if entry_idx is not None else 0
            trade_log.append({
                "timestamp": ts,
                "action": "close",
                "position_size": position_size,
                "cost": cost,
                "capital": capital,
                "hold_periods": hold_periods,
            })
            position_size = 0.0
            state["position"] = "flat"
            entry_idx = None

        equity_list.append(capital)
        timestamps.append(ts)

    # 마지막에 포지션이 열려 있으면 청산 비용 고려한 시가평가
    if state["position"] == "open":
        implied_exit_cost = position_size * exit_cost_rate
        equity_list[-1] = capital - implied_exit_cost

    equity = pd.Series(equity_list, index=timestamps)
    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    return _compute_metrics(
        equity=equity,
        trade_df=trade_df,
        initial_capital=initial_capital,
        total_funding_collected=total_funding_collected,
        total_funding_paid=total_funding_paid,
        total_costs=total_costs,
        total_periods=len(df),
        position_periods=sum(1 for e in equity_list if e != initial_capital or True),
        df=df,
        state=state,
    )


def _compute_metrics(
    equity: pd.Series,
    trade_df: pd.DataFrame,
    initial_capital: float,
    total_funding_collected: float,
    total_funding_paid: float,
    total_costs: float,
    total_periods: int,
    position_periods: int,
    df: pd.DataFrame,
    state: dict,
) -> dict:
    """시뮬레이션 결과에서 메트릭을 계산한다."""
    final_capital = equity.iloc[-1]
    total_return_pct = (final_capital / initial_capital - 1) * 100

    # 기간 계산 (일 단위)
    ts_index = equity.index
    if hasattr(ts_index[0], "timestamp"):
        start_ts = ts_index[0]
        end_ts = ts_index[-1]
    else:
        start_ts = pd.Timestamp(ts_index[0])
        end_ts = pd.Timestamp(ts_index[-1])
    total_days = max((end_ts - start_ts).total_seconds() / 86400, 1)
    total_years = total_days / 365.0

    annualized_return_pct = ((final_capital / initial_capital) ** (1 / total_years) - 1) * 100 if total_years > 0 else 0

    # 일별 수익률로 리샘플링 (8h → daily, 하루 3회 합산)
    equity_ts = equity.copy()
    equity_ts.index = pd.to_datetime(equity_ts.index)
    daily_equity = equity_ts.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    sharpe_ratio = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(365)

    # MDD 계산
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown_pct = drawdown.min() * 100

    calmar_ratio = 0.0
    if max_drawdown_pct < 0:
        calmar_ratio = annualized_return_pct / abs(max_drawdown_pct)

    # 거래 통계
    entries = trade_df[trade_df["action"] == "open"] if len(trade_df) > 0 else pd.DataFrame()
    exits = trade_df[trade_df["action"] == "close"] if len(trade_df) > 0 else pd.DataFrame()
    total_entries = len(entries)
    total_exits = len(exits)

    avg_hold_days = 0.0
    if len(exits) > 0 and "hold_periods" in exits.columns:
        avg_hold_days = exits["hold_periods"].mean() * 8 / 24  # 8h per period

    # 시장 참여 시간 계산
    time_in_market_pct = 0.0
    if total_periods > 0 and len(trade_df) > 0:
        open_periods = 0
        in_pos = False
        open_idx = 0
        for _, t in trade_df.iterrows():
            if t["action"] == "open":
                in_pos = True
                open_idx = df[df["timestamp"] == t["timestamp"]].index[0] if len(df[df["timestamp"] == t["timestamp"]]) > 0 else 0
            elif t["action"] == "close":
                close_idx = df[df["timestamp"] == t["timestamp"]].index[0] if len(df[df["timestamp"] == t["timestamp"]]) > 0 else 0
                open_periods += close_idx - open_idx
                in_pos = False
        if in_pos:
            open_periods += total_periods - open_idx
        time_in_market_pct = open_periods / total_periods * 100

    net_funding = total_funding_collected - total_funding_paid
    net_profit = net_funding - total_costs

    return {
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized_return_pct, 2),
        "total_funding_collected": round(total_funding_collected, 2),
        "total_funding_paid": round(total_funding_paid, 2),
        "net_funding": round(net_funding, 2),
        "total_costs": round(total_costs, 2),
        "net_profit": round(net_profit, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "calmar_ratio": round(calmar_ratio, 2),
        "total_entries": total_entries,
        "total_exits": total_exits,
        "avg_hold_days": round(avg_hold_days, 1),
        "time_in_market_pct": round(time_in_market_pct, 1),
        "equity_curve": equity,
        "trade_log": trade_df,
    }


# ─────────────────────────────────────────────
# 데이터 로드 + 구간 필터
# ─────────────────────────────────────────────
def load_data(symbol: str = "BTC") -> pd.DataFrame:
    """8h 펀딩비 피처 데이터를 로드한다."""
    path = f"data/processed/{symbol}USDT_8h_funding_features.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def filter_period(df: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    """구간별 데이터 필터링."""
    start_ts = pd.Timestamp(start, tz="UTC")
    mask = df["timestamp"] >= start_ts
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        mask &= df["timestamp"] < end_ts
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────
# 결과 출력 헬퍼
# ─────────────────────────────────────────────
def _fmt(val, suffix: str = "", width: int = 10) -> str:
    """숫자를 포맷팅한다."""
    if isinstance(val, float):
        s = f"{val:+.2f}{suffix}" if suffix == "%" else f"{val:.2f}{suffix}"
        if suffix == "%" and val >= 0:
            s = f"+{val:.2f}{suffix}"
    elif isinstance(val, int):
        s = f"{val}{suffix}"
    else:
        s = str(val)
    return s.rjust(width)


def print_table(headers: list[str], rows: list[list], col_width: int = 12) -> None:
    """간단한 테이블을 출력한다."""
    sep = "+" + "+".join("-" * (col_width + 2) for _ in headers) + "+"
    header_line = "|" + "|".join(h.center(col_width + 2) for h in headers) + "|"
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        line = "|" + "|".join(str(v).center(col_width + 2) for v in row) + "|"
        print(line)
    print(sep)


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def run_all(symbol: str = "BTC", single_strategy: str | None = None) -> None:
    """전체 분석을 실행한다."""
    df = load_data(symbol)
    print(f"\n{'=' * 60}")
    print(f"펀딩비 차익거래 Phase 1: 정적 전략 백테스트 결과")
    print(f"심볼: {symbol}USDT | 데이터: {len(df)}건")
    print(f"기간: {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"{'=' * 60}")

    strategies_to_run = (
        {single_strategy: STRATEGIES[single_strategy]}
        if single_strategy
        else STRATEGIES
    )

    # ── 1. 전략 비교 (전체 기간) ──
    print(f"\n1. {symbol} 전략 비교 (전체 기간, 레버리지 1x)")
    full_df = filter_period(df, *PERIODS["Full"])

    results = {}
    for name, fn in strategies_to_run.items():
        results[name] = simulate_funding_arb(full_df, fn)

    metrics_keys = [
        ("총 수익률", "total_return_pct", "%"),
        ("연환산", "annualized_return_pct", "%"),
        ("MDD", "max_drawdown_pct", "%"),
        ("샤프", "sharpe_ratio", ""),
        ("칼마", "calmar_ratio", ""),
        ("진입 횟수", "total_entries", ""),
        ("참여 시간", "time_in_market_pct", "%"),
        ("순 펀딩비", "net_funding", "$"),
        ("총 비용", "total_costs", "$"),
    ]

    strat_names = list(results.keys())
    short_names = {"buy_and_hold": "A:B&H", "threshold": "B:Thres", "ma_filter": "C:MA"}
    headers = ["메트릭"] + [short_names.get(n, n) for n in strat_names]

    rows = []
    for label, key, unit in metrics_keys:
        row = [label]
        for name in strat_names:
            val = results[name][key]
            if unit == "$":
                row.append(f"${val:,.0f}")
            elif unit == "%":
                row.append(f"{val:+.2f}%")
            else:
                row.append(f"{val:.2f}")
        rows.append(row)

    print_table(headers, rows)

    # ── 2. 최적 전략 구간별 성과 ──
    # 최적 전략 = 샤프 비율이 가장 높은 전략
    best_name = max(results, key=lambda n: results[n]["sharpe_ratio"])
    best_fn = strategies_to_run[best_name]
    print(f"\n2. 최적 전략 구간별 성과 (전략: {short_names.get(best_name, best_name)})")

    period_results = {}
    for pname, (start, end) in PERIODS.items():
        if pname == "Full":
            continue
        pdf = filter_period(df, start, end)
        if len(pdf) < 3:
            continue
        period_results[pname] = simulate_funding_arb(pdf, best_fn)

    headers = ["구간", "수익률", "MDD", "샤프"]
    rows = []
    for pname, res in period_results.items():
        rows.append([
            pname,
            f"{res['total_return_pct']:+.2f}%",
            f"{res['max_drawdown_pct']:+.2f}%",
            f"{res['sharpe_ratio']:.2f}",
        ])
    print_table(headers, rows)

    # ── 3. 레버리지 감도 (Buy & Hold) ──
    print(f"\n3. 레버리지 감도 (전략 A: Buy & Hold)")

    lev_results = {}
    bnh_fn = strategy_buy_and_hold
    for lev in LEVERAGE_LEVELS:
        lev_results[lev] = simulate_funding_arb(full_df, bnh_fn, leverage=lev)

    headers = ["레버리지", "연환산", "MDD", "샤프"]
    rows = []
    for lev in LEVERAGE_LEVELS:
        res = lev_results[lev]
        rows.append([
            f"{lev:.1f}x",
            f"{res['annualized_return_pct']:+.2f}%",
            f"{res['max_drawdown_pct']:+.2f}%",
            f"{res['sharpe_ratio']:.2f}",
        ])
    print_table(headers, rows)

    # 최적 레버리지 (MDD < -3% 이내에서 최대 샤프)
    valid_levs = {lev: r for lev, r in lev_results.items() if r["max_drawdown_pct"] >= -3.0}
    if valid_levs:
        optimal_lev = max(valid_levs, key=lambda l: valid_levs[l]["sharpe_ratio"])
        print(f"  -> 최적 레버리지 (MDD >= -3%): {optimal_lev:.1f}x")
    else:
        # MDD -3% 넘어도 가장 MDD가 작은 것
        optimal_lev = min(lev_results, key=lambda l: abs(lev_results[l]["max_drawdown_pct"]))
        print(f"  -> 모든 레버리지 MDD > -3%. 최소 MDD: {optimal_lev:.1f}x "
              f"(MDD {lev_results[optimal_lev]['max_drawdown_pct']:+.2f}%)")

    # ── 4. v1 + 펀딩비 합산 (Strict OOS) ──
    print(f"\n4. v1 + 펀딩비 합산 (Strict OOS)")

    strict_df = filter_period(df, *PERIODS["Strict OOS"])
    if len(strict_df) >= 3:
        arb_strict = simulate_funding_arb(strict_df, best_fn)
        arb_ret = arb_strict["total_return_pct"]
        v1_ret = V1_RETURNS.get("Strict OOS", -2.71)

        # 합산 (50/50 배분)
        combined_ret = v1_ret * 0.5 + arb_ret * 0.5

        headers = ["전략", "수익률", "MDD"]
        rows = [
            ["v1 단독", f"{v1_ret:+.2f}%", "N/A"],
            ["펀딩비 단독", f"{arb_ret:+.2f}%", f"{arb_strict['max_drawdown_pct']:+.2f}%"],
            ["v1 50%+펀딩50%", f"{combined_ret:+.2f}%", "N/A"],
        ]
        print_table(headers, rows)

        cover_ok = combined_ret > v1_ret
        print(f"  -> 하락장 커버: {'PASS' if cover_ok else 'FAIL'}")
    else:
        print("  Strict OOS 구간 데이터 부족")
        arb_strict = None
        arb_ret = 0.0

    # ── 5. ETH 비교 (symbol이 BTC일 때만) ──
    if symbol == "BTC":
        print(f"\n5. ETH 결과 (전략 A 기준)")
        try:
            eth_df = load_data("ETH")
            eth_full = filter_period(eth_df, *PERIODS["Full"])
            eth_res = simulate_funding_arb(eth_full, bnh_fn)
            btc_res = results.get("buy_and_hold", results[list(results.keys())[0]])

            headers = ["메트릭", "BTC", "ETH"]
            rows = [
                ["연환산", f"{btc_res['annualized_return_pct']:+.2f}%",
                 f"{eth_res['annualized_return_pct']:+.2f}%"],
                ["MDD", f"{btc_res['max_drawdown_pct']:+.2f}%",
                 f"{eth_res['max_drawdown_pct']:+.2f}%"],
                ["샤프", f"{btc_res['sharpe_ratio']:.2f}",
                 f"{eth_res['sharpe_ratio']:.2f}"],
            ]
            print_table(headers, rows)
        except FileNotFoundError:
            print("  ETH 데이터 없음")

    # ── 6. Phase 2 진입 판단 ──
    print(f"\n6. Phase 2 진입 판단")

    best_full = results[best_name]
    ann_ret = best_full["annualized_return_pct"]
    sharpe = best_full["sharpe_ratio"]
    strict_ret = arb_ret if arb_strict else 0.0
    v1_ret_strict = V1_RETURNS.get("Strict OOS", -2.71)
    combined = v1_ret_strict * 0.5 + strict_ret * 0.5

    checks = [
        (ann_ret > 5, f"연환산 > 5%: {ann_ret:+.2f}%"),
        (strict_ret > 0, f"Strict OOS > 0%: {strict_ret:+.2f}%"),
        (sharpe > 0.8, f"샤프 > 0.8: {sharpe:.2f}"),
        (combined > v1_ret_strict, f"합산 Strict OOS > v1 단독: {combined:+.2f}% > {v1_ret_strict:+.2f}%"),
    ]

    passed = 0
    for ok, desc in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {'[v]' if ok else '[x]'} {status}: {desc}")
        if ok:
            passed += 1

    print(f"\n  결과: {passed}/4 통과", end="")
    if passed >= 3:
        print(" -> Phase 2(ML 강화) 진행 권장")
    else:
        print(" -> Phase 2 진행 보류, 추가 분석 필요")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="펀딩비 차익거래 백테스트")
    parser.add_argument("--symbol", default="BTC", choices=["BTC", "ETH"],
                        help="심볼 (기본: BTC)")
    parser.add_argument("--strategy", default=None,
                        choices=list(STRATEGIES.keys()),
                        help="단일 전략만 실행")
    args = parser.parse_args()

    run_all(symbol=args.symbol, single_strategy=args.strategy)


if __name__ == "__main__":
    main()
