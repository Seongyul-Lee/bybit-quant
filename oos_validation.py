"""OOS(Out-of-Sample) 검증 스크립트.

config.yaml과 학습된 모델을 기반으로 구간별 성과를 측정한다.
Post-Validation 구간에서 성공 기준 충족 여부를 판단한다.

사용법:
    python oos_validation.py --strategy btc_1h_momentum
    python oos_validation.py --strategy eth_1h_momentum
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml

from strategies._common.features import FeatureEngine


def load_config(strategy_name: str = "btc_1h_momentum"):
    """config.yaml에서 파라미터를 로드.

    Args:
        strategy_name: 전략 폴더명.
    """
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def simulate_period(df_period, signals_period, sl_pct, tp_pct, max_hold,
                    position_pct=0.05, fee_per_side=0.00055, slippage_per_side=0.0):
    """포지션 없을 때만 진입, SL/TP로 청산하는 순차 시뮬레이션."""
    close = df_period["close"].values
    high = df_period["high"].values
    low = df_period["low"].values
    sigs = signals_period.values
    n = len(close)

    trades = []
    i = 0

    while i < n:
        if sigs[i] != 1:
            i += 1
            continue

        entry_price = close[i]
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)

        exit_bar = None
        exit_return = 0
        exit_type = "timeout"

        for j in range(i + 1, min(i + 1 + max_hold, n)):
            if high[j] >= tp_price and low[j] <= sl_price:
                exit_bar = j
                exit_return = -sl_pct
                exit_type = "sl"
                break
            elif high[j] >= tp_price:
                exit_bar = j
                exit_return = tp_pct
                exit_type = "tp"
                break
            elif low[j] <= sl_price:
                exit_bar = j
                exit_return = -sl_pct
                exit_type = "sl"
                break

        if exit_bar is None:
            exit_bar = min(i + max_hold, n - 1)
            exit_return = (close[exit_bar] - entry_price) / entry_price
            exit_type = "timeout"

        trades.append({
            "entry_bar": i,
            "exit_bar": exit_bar,
            "return": exit_return,
            "type": exit_type,
            "holding": exit_bar - i,
        })

        i = exit_bar + 1

    if not trades:
        return {"trades": 0, "pf": 0, "total_return": 0, "win_rate": 0,
                "mdd": 0, "tp_count": 0, "sl_count": 0, "timeout_count": 0}

    returns = np.array([t["return"] for t in trades])
    wins = returns > 0

    cumulative = 1.0
    equity_curve = [1.0]
    for r in returns:
        total_cost = 2 * (fee_per_side + slippage_per_side)
        pnl = position_pct * (r - total_cost)
        cumulative *= (1 + pnl)
        equity_curve.append(cumulative)

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = dd.min() * 100

    gross_profit = returns[wins].sum() if wins.any() else 0
    gross_loss = abs(returns[~wins].sum()) if (~wins).any() else 0.001
    pf = gross_profit / gross_loss

    types = [t["type"] for t in trades]

    return {
        "trades": len(trades),
        "win_rate": wins.mean() * 100,
        "pf": pf,
        "total_return": (cumulative - 1) * 100,
        "mdd": mdd,
        "tp_count": types.count("tp"),
        "sl_count": types.count("sl"),
        "timeout_count": types.count("timeout"),
    }


def run_oos_validation(strategy_name: str = "btc_1h_momentum"):
    """OOS 검증을 실행하고 결과를 출력.

    Args:
        strategy_name: 전략 폴더명.
    """
    config = load_config(strategy_name)
    params = config.get("params", {})
    risk = config.get("risk", {})
    strat_cfg = config.get("strategy", {})

    symbol = strat_cfg.get("symbol", "BTCUSDT")
    timeframe = strat_cfg.get("timeframe", "1h")

    CONFIDENCE_THRESHOLD = params.get("confidence_threshold", 0.46)
    SL_PCT = risk.get("stop_loss_pct", 0.015)
    TP_PCT = risk.get("take_profit_pct", 0.015)
    MAX_HOLD = params.get("max_holding_period", 16)
    POSITION_PCT = risk.get("max_position_pct", 0.05)
    FEE_PER_SIDE = config.get("execution", {}).get("fee_rate", 0.00055)
    ENSEMBLE_FOLDS = params.get("ensemble_folds", None)
    MODELS_DIR = params.get("models_dir", f"strategies/{strategy_name}/models")

    # 모델 로드 (앙상블 또는 단일)
    if ENSEMBLE_FOLDS:
        models = []
        for fold_idx in ENSEMBLE_FOLDS:
            path = os.path.join(MODELS_DIR, f"fold_{fold_idx:02d}.txt")
            models.append(lgb.Booster(model_file=path))
        print(f"앙상블 모델: {len(models)}개 fold ({ENSEMBLE_FOLDS})")
    else:
        model_path = params.get("model_path", f"strategies/{strategy_name}/models/latest.txt")
        model = lgb.Booster(model_file=model_path)
        models = None
        print("단일 모델 (latest.txt)")

    feature_names_path = params.get(
        "feature_names_path", f"strategies/{strategy_name}/models/feature_names.json"
    )
    with open(feature_names_path) as f:
        feature_names = json.load(f)

    # 학습 메타 로드
    meta_path = os.path.join(MODELS_DIR, "training_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # 앙상블일 때 PV 시작점 = 가장 최신 fold의 val_end
    if ENSEMBLE_FOLDS:
        latest_fold = max(ENSEMBLE_FOLDS)
        fm = meta["folds_metrics"][latest_fold]
        val_end_str = fm["val_period"].split(" ~ ")[-1].strip()
    else:
        best_fold_idx = meta.get("best_fold_idx", -1)
        fm = meta["folds_metrics"][best_fold_idx]
        val_end_str = fm["val_period"].split(" ~ ")[-1].strip()

    # 데이터 로드 & 시그널 생성
    data_path = f"data/processed/{symbol}_{timeframe}_features.parquet"
    print(f"전략: {strategy_name} | 심볼: {symbol} | 타임프레임: {timeframe}")
    df = pd.read_parquet(data_path)
    engine = FeatureEngine(config={"symbol": symbol})
    df_feat = engine.compute_all_features(df)

    X = df_feat[feature_names]
    valid_mask = ~X.isna().any(axis=1)

    signals = pd.Series(0, index=df.index, dtype=int)
    proba_series = pd.Series(np.nan, index=df.index)

    if models:
        preds = [m.predict(X[valid_mask]) for m in models]
        proba = np.mean(preds, axis=0)
    else:
        proba = model.predict(X[valid_mask])

    proba_series.loc[valid_mask] = proba

    # 펀딩비 적응형 threshold
    funding_filter = params.get("funding_filter", {})
    if funding_filter.get("enabled", False) and "funding_rate_zscore" in df_feat.columns:
        fr_zscore = df_feat["funding_rate_zscore"].values
        zscore_thresholds = funding_filter.get("zscore_thresholds", [])
        # 기본: 최고 threshold 초과 시 차단 (999)
        adaptive_thr = np.full(len(df), 999.0)
        # zscore_thresholds를 역순으로 적용 (가장 넓은 범위부터)
        for rule in sorted(zscore_thresholds, key=lambda x: x["zscore_below"], reverse=True):
            mask = fr_zscore < rule["zscore_below"]
            adaptive_thr[mask] = rule["confidence"]
        # NaN인 구간은 기본 threshold
        adaptive_thr[np.isnan(fr_zscore)] = CONFIDENCE_THRESHOLD
        signals.loc[valid_mask] = np.where(proba >= adaptive_thr[valid_mask], 1, 0)
        print(f"펀딩비 적응형 threshold 적용: {zscore_thresholds}")
    else:
        signals.loc[valid_mask] = np.where(proba >= CONFIDENCE_THRESHOLD, 1, 0)

    ts = pd.to_datetime(df["timestamp"])

    # 구간 정의
    val_end_ts = pd.Timestamp(val_end_str)
    if val_end_ts.tzinfo is None:
        val_end_ts = val_end_ts.tz_localize("UTC")

    oos_boundary = pd.Timestamp("2026-01-19", tz="UTC")

    periods = {
        "In-Sample": (ts.iloc[0], val_end_ts),
        "Post-Validation": (val_end_ts, oos_boundary),
        "Strict OOS (2026-01-19~)": (oos_boundary, ts.iloc[-1]),
    }

    # 결과 출력
    print("=" * 80)
    print("OOS 검증")
    print("=" * 80)
    print(f"confidence_threshold: {CONFIDENCE_THRESHOLD}")
    print(f"SL: {SL_PCT*100}% / TP: {TP_PCT*100}%")
    print(f"Max Hold: {MAX_HOLD}")
    print(f"Post-Val: {val_end_ts} ~ {oos_boundary}")
    print()

    # 시그널 분포
    print(f"전체 시그널: 매수={int((signals==1).sum())}, 비매수={int((signals==0).sum())}")
    print()

    # 시나리오 정의: (이름, slippage_per_side)
    scenarios = [
        ("낙관적 (limit)", 0.0005),
        ("보수적 (taker)", 0.001),
    ]

    all_scenario_results = {}

    for scenario_name, slippage in scenarios:
        total_cost_round = 2 * (FEE_PER_SIDE + slippage) * 100
        print(f"\n{'='*80}")
        print(f"시나리오: {scenario_name} - 슬리피지 편도 {slippage*100:.2f}%, 왕복 총비용 {total_cost_round:.2f}%")
        print(f"{'='*80}")
        print(f"{'구간':<30} {'거래':>5} {'승률':>6} {'PF':>6} {'수익률':>8} {'MDD':>7}")
        print("-" * 70)

        results = {}
        for name, (start, end) in periods.items():
            mask = (ts >= start) & (ts < end)
            idx = df.index[mask]
            if len(idx) == 0:
                print(f"{name:<30} 데이터 없음")
                continue
            result = simulate_period(
                df.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                signals.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                SL_PCT, TP_PCT, MAX_HOLD, POSITION_PCT, FEE_PER_SIDE,
                slippage_per_side=slippage,
            )
            results[name] = result
            if result["trades"] == 0:
                print(f"{name:<30} 거래 없음")
                continue
            r = result
            print(f"{name:<30} {r['trades']:>5} {r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
                  f"{r['total_return']:>+7.2f}% {r['mdd']:>6.2f}%")

        all_scenario_results[scenario_name] = results

    print()

    # 성공 기준 판단 (보수적 시나리오 기준)
    conservative = all_scenario_results.get("보수적 (taker)", {})
    optimistic = all_scenario_results.get("낙관적 (limit)", {})

    is_result = conservative.get("In-Sample", {})
    pv_result = conservative.get("Post-Validation", {})

    if pv_result.get("trades", 0) == 0:
        print("FAIL: Post-Validation 구간에 거래가 없습니다 (보수적).")
        return all_scenario_results

    pv_pf = pv_result["pf"]
    pv_return = pv_result["total_return"]
    pv_trades = pv_result["trades"]
    is_pf = is_result.get("pf", 1.0)

    pf_drop = (is_pf - pv_pf) / is_pf * 100 if is_pf > 0 else 100

    # 거래 수 기준 완화: 25건
    checks = {
        "보수적 PV PF >= 1.20": pv_pf >= 1.2,
        "보수적 PV 수익률 > 0%": pv_return > 0,
        "PV 거래 수 >= 25": pv_trades >= 25,
        "PF 하락률 <= 50%": pf_drop <= 50,
    }

    opt_pv = optimistic.get("Post-Validation", {})
    opt_pv_pf = opt_pv.get("pf", 0) if opt_pv else 0

    print("--- 성공 기준 (보수적 시나리오 기준) ---")
    for criterion, passed in checks.items():
        status = "O" if passed else "X"
        print(f"  [{status}] {criterion}")

    print(f"\n  IS PF: {is_pf:.2f} → 보수적 PV PF: {pv_pf:.2f} (하락률: {pf_drop:.1f}%)")
    print(f"  낙관적 PV PF: {opt_pv_pf:.2f}")

    all_passed = all(checks.values())
    print(f"\n최종 결과: {'SUCCESS' if all_passed else 'FAIL'}")
    print("=" * 80)

    return all_scenario_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOS 검증")
    parser.add_argument("--strategy", type=str, default="btc_1h_momentum",
                        help="전략 이름 (기본: btc_1h_momentum)")
    args = parser.parse_args()
    run_oos_validation(strategy_name=args.strategy)
