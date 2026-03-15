"""Out-of-sample validation for ML strategies."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.validation.gate import (
    GATE_VERSION,
    evaluate_deployment_gate,
    save_strategy_validation_artifact,
    save_validation_run_summary,
)
from strategies._common.features import FeatureEngine

JSON_RESULT_MARKER = "__JSON_RESULT__"
DEFAULT_OUTPUT_DIR = "reports/validation"
STRICT_OOS_BOUNDARY = pd.Timestamp("2026-01-19", tz="UTC")


def load_config(strategy_name: str) -> dict[str, Any]:
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_active_strategies() -> list[str]:
    with open("config/portfolio.yaml", "r", encoding="utf-8") as handle:
        portfolio = yaml.safe_load(handle)
    return portfolio["portfolio"]["active_strategies"]


def to_utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def to_utc_series(values: Any) -> pd.Series:
    timestamps = pd.to_datetime(values)
    if timestamps.dt.tz is None:
        return timestamps.dt.tz_localize("UTC")
    return timestamps.dt.tz_convert("UTC")


def simulate_period(
    df_period: pd.DataFrame,
    signals_period: pd.Series,
    sl_pct: float,
    tp_pct: float,
    max_hold: int,
    position_pct: float = 0.05,
    fee_per_side: float = 0.00055,
    slippage_per_side: float = 0.0,
) -> dict[str, Any]:
    """Simulate long-only one-position-at-a-time execution."""

    close = df_period["close"].to_numpy()
    high = df_period["high"].to_numpy()
    low = df_period["low"].to_numpy()
    sigs = signals_period.to_numpy()
    n_bars = len(close)

    trades: list[dict[str, Any]] = []
    index = 0

    while index < n_bars:
        if sigs[index] != 1:
            index += 1
            continue

        entry_price = close[index]
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)

        exit_bar: int | None = None
        exit_return = 0.0
        exit_type = "timeout"

        for probe in range(index + 1, min(index + 1 + max_hold, n_bars)):
            if high[probe] >= tp_price and low[probe] <= sl_price:
                exit_bar = probe
                exit_return = -sl_pct
                exit_type = "sl"
                break
            if high[probe] >= tp_price:
                exit_bar = probe
                exit_return = tp_pct
                exit_type = "tp"
                break
            if low[probe] <= sl_price:
                exit_bar = probe
                exit_return = -sl_pct
                exit_type = "sl"
                break

        if exit_bar is None:
            exit_bar = min(index + max_hold, n_bars - 1)
            exit_return = (close[exit_bar] - entry_price) / entry_price

        trades.append(
            {
                "entry_bar": index,
                "exit_bar": exit_bar,
                "return": exit_return,
                "type": exit_type,
                "holding": exit_bar - index,
            }
        )
        index = exit_bar + 1

    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "pf": 0.0,
            "total_return": 0.0,
            "mdd": 0.0,
            "tp_count": 0,
            "sl_count": 0,
            "timeout_count": 0,
        }

    returns = np.array([trade["return"] for trade in trades], dtype=float)
    wins = returns > 0

    cumulative = 1.0
    equity_curve = [1.0]
    for trade_return in returns:
        total_cost = 2 * (fee_per_side + slippage_per_side)
        pnl = position_pct * (trade_return - total_cost)
        cumulative *= 1 + pnl
        equity_curve.append(cumulative)

    equity = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    gross_profit = returns[wins].sum() if wins.any() else 0.0
    gross_loss = abs(returns[~wins].sum()) if (~wins).any() else 0.001
    trade_types = [trade["type"] for trade in trades]

    return {
        "trades": len(trades),
        "win_rate": float(wins.mean() * 100),
        "pf": float(gross_profit / gross_loss),
        "total_return": float((cumulative - 1) * 100),
        "mdd": float(drawdown.min() * 100),
        "tp_count": trade_types.count("tp"),
        "sl_count": trade_types.count("sl"),
        "timeout_count": trade_types.count("timeout"),
    }


def _load_models(
    strategy_name: str,
    params: dict[str, Any],
) -> tuple[list[lgb.Booster], list[int], str, dict[str, Any]]:
    models_dir = params.get("models_dir", f"strategies/{strategy_name}/models")
    ensemble_folds = params.get("ensemble_folds")
    meta_path = os.path.join(models_dir, "training_meta.json")

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    if ensemble_folds:
        models = []
        for fold_idx in ensemble_folds:
            model_path = os.path.join(models_dir, f"fold_{fold_idx:02d}.txt")
            models.append(lgb.Booster(model_file=model_path))
        return models, list(ensemble_folds), "ensemble", meta

    model_path = params.get("model_path", f"{models_dir}/latest.txt")
    model = lgb.Booster(model_file=model_path)
    best_fold_idx = int(meta.get("best_fold_idx", -1))
    model_folds = [best_fold_idx] if best_fold_idx >= 0 else []
    return [model], model_folds, "single", meta


def _resolve_validation_end(
    meta: dict[str, Any],
    model_folds: list[int],
    model_mode: str,
) -> pd.Timestamp:
    folds_metrics = meta["folds_metrics"]
    if model_mode == "ensemble" and model_folds:
        latest_fold = max(model_folds)
        fold_meta = folds_metrics[latest_fold]
    else:
        best_fold_idx = int(meta.get("best_fold_idx", -1))
        fold_meta = folds_metrics[best_fold_idx]
    val_end_str = fold_meta["val_period"].split(" ~ ")[-1].strip()
    return to_utc_timestamp(val_end_str)


def _build_signals(
    df: pd.DataFrame,
    df_feat: pd.DataFrame,
    feature_names: list[str],
    models: list[lgb.Booster],
    config: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    params = config.get("params", {})
    risk = config.get("risk", {})

    confidence_threshold = params.get("confidence_threshold", 0.46)
    X = df_feat[feature_names]
    valid_mask = ~X.isna().any(axis=1)

    raw_predictions = [model.predict(X[valid_mask]) for model in models]
    if len(raw_predictions) == 1:
        probabilities = raw_predictions[0]
    else:
        probabilities = np.mean(raw_predictions, axis=0)

    probability_series = pd.Series(np.nan, index=df.index, dtype=float)
    probability_series.loc[valid_mask] = probabilities

    adaptive_threshold = np.full(len(df), confidence_threshold, dtype=float)

    funding_filter = params.get("funding_filter", {})
    if funding_filter.get("enabled", False) and "funding_rate_zscore" in df_feat.columns:
        funding_z = df_feat["funding_rate_zscore"].to_numpy()
        adaptive_threshold = np.full(len(df), 999.0, dtype=float)
        rules = funding_filter.get("zscore_thresholds", [])
        for rule in sorted(rules, key=lambda item: item["zscore_below"], reverse=True):
            mask = funding_z < rule["zscore_below"]
            adaptive_threshold[mask] = rule["confidence"]
        adaptive_threshold[np.isnan(funding_z)] = confidence_threshold

    oi_filter = params.get("oi_filter", {})
    if oi_filter.get("enabled", False) and "oi_zscore" in df_feat.columns:
        oi_block = oi_filter.get("block_zscore")
        if oi_block is not None:
            oi_z = df_feat["oi_zscore"].to_numpy()
            blocked = (oi_z >= oi_block) & ~np.isnan(oi_z)
            adaptive_threshold[blocked] = 999.0

    signals = pd.Series(0, index=df.index, dtype=int)
    signals.loc[valid_mask] = np.where(
        probability_series.loc[valid_mask] >= adaptive_threshold[valid_mask],
        1,
        0,
    )

    _ = risk  # keep config access explicit for parity with caller expectations
    return signals, probability_series


def _build_periods(start_ts: pd.Timestamp, val_end_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        "In-Sample": (start_ts, val_end_ts),
        "Post-Validation": (val_end_ts, STRICT_OOS_BOUNDARY),
        "Strict OOS (2026-01-19~)": (STRICT_OOS_BOUNDARY, end_ts),
    }


def _period_result(results: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return dict(results.get(name, {}))


def _evaluate_scenarios(
    df: pd.DataFrame,
    signals: pd.Series,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    sl_pct: float,
    tp_pct: float,
    max_hold: int,
    position_pct: float,
    fee_per_side: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    ts = to_utc_series(df["timestamp"])
    scenarios = [
        ("optimistic_limit", "Optimistic (limit)", 0.0005),
        ("conservative_taker", "Conservative (taker)", 0.0010),
    ]

    scenario_results: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario_key, scenario_label, slippage in scenarios:
        period_results: dict[str, dict[str, Any]] = {}
        for period_name, (start, end) in periods.items():
            mask = (ts >= start) & (ts < end)
            idx = df.index[mask]
            if len(idx) == 0:
                period_results[period_name] = {
                    "trades": 0,
                    "win_rate": 0.0,
                    "pf": 0.0,
                    "total_return": 0.0,
                    "mdd": 0.0,
                    "tp_count": 0,
                    "sl_count": 0,
                    "timeout_count": 0,
                }
                continue

            period_results[period_name] = simulate_period(
                df.iloc[idx[0] : idx[-1] + 1].reset_index(drop=True),
                signals.iloc[idx[0] : idx[-1] + 1].reset_index(drop=True),
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                max_hold=max_hold,
                position_pct=position_pct,
                fee_per_side=fee_per_side,
                slippage_per_side=slippage,
            )

        scenario_results[scenario_key] = {
            "label": scenario_label,
            "slippage_per_side": slippage,
            "periods": period_results,
        }

    return scenario_results


def _build_structured_result(
    strategy_name: str,
    scenario_results: dict[str, dict[str, Any]],
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    model_mode: str,
    model_folds: list[int],
    signal_count: int,
    generated_at: datetime,
) -> dict[str, Any]:
    conservative_periods = scenario_results["conservative_taker"]["periods"]
    optimistic_periods = scenario_results["optimistic_limit"]["periods"]

    is_result = _period_result(conservative_periods, "In-Sample")
    pv_result = _period_result(conservative_periods, "Post-Validation")
    strict_result = _period_result(conservative_periods, "Strict OOS (2026-01-19~)")

    is_pf = float(is_result.get("pf", 0) or 0)
    pv_pf = float(pv_result.get("pf", 0) or 0)
    pf_drop = ((is_pf - pv_pf) / is_pf * 100) if is_pf > 0 else 100.0

    conservative_summary = {
        "is_pf": is_pf,
        "pv_pf": pv_pf,
        "pv_return": float(pv_result.get("total_return", 0) or 0),
        "pv_trades": int(pv_result.get("trades", 0) or 0),
        "pf_drop": float(pf_drop),
        "optimistic_pv_pf": float(
            optimistic_periods.get("Post-Validation", {}).get("pf", 0) or 0
        ),
    }

    post_validation = {
        "pf": pv_pf,
        "trades": int(pv_result.get("trades", 0) or 0),
        "total_return": float(pv_result.get("total_return", 0) or 0),
        "mdd": float(pv_result.get("mdd", 0) or 0),
        "win_rate": float(pv_result.get("win_rate", 0) or 0),
        "pf_drop": float(pf_drop),
    }
    strict_oos = {
        "pf": float(strict_result.get("pf", 0) or 0),
        "trades": int(strict_result.get("trades", 0) or 0),
        "total_return": float(strict_result.get("total_return", 0) or 0),
        "mdd": float(strict_result.get("mdd", 0) or 0),
        "win_rate": float(strict_result.get("win_rate", 0) or 0),
    }

    gate_result = evaluate_deployment_gate(conservative_summary, strict_oos)
    period_payload = {
        "in_sample": {
            "start": periods["In-Sample"][0].isoformat(),
            "end": periods["In-Sample"][1].isoformat(),
        },
        "post_validation": {
            "start": periods["Post-Validation"][0].isoformat(),
            "end": periods["Post-Validation"][1].isoformat(),
        },
        "strict_oos": {
            "start": periods["Strict OOS (2026-01-19~)"][0].isoformat(),
            "end": periods["Strict OOS (2026-01-19~)"][1].isoformat(),
        },
    }

    return {
        "strategy": strategy_name,
        "generated_at": generated_at.isoformat(),
        "gate_version": gate_result["gate_version"],
        "gate_criteria": gate_result["criteria"],
        "model_selection": {
            "mode": model_mode,
            "folds": model_folds,
        },
        "signal_count": signal_count,
        "periods": period_payload,
        "scenarios": scenario_results,
        "conservative": conservative_summary,
        "post_validation": post_validation,
        "strict_oos": strict_oos,
        "deployment_decision": {
            "passed": gate_result["passed"],
            "failure_reasons": gate_result["failure_reasons"],
        },
        "passed": gate_result["passed"],
        "failure_reasons": gate_result["failure_reasons"],
    }


def _print_result_summary(strategy_name: str, structured: dict[str, Any]) -> None:
    pv = structured["post_validation"]
    strict = structured["strict_oos"]
    decision = structured["deployment_decision"]
    status = "PASS" if decision["passed"] else "FAIL"

    print("=" * 80)
    print(f"OOS validation: {strategy_name}")
    print("=" * 80)
    print(f"Gate version: {structured['gate_version']}")
    print(
        f"Post-Validation: PF {pv['pf']:.2f} | Return {pv['total_return']:+.2f}% "
        f"| Trades {pv['trades']} | PF drop {pv['pf_drop']:.2f}%"
    )
    print(
        f"Strict OOS:      PF {strict['pf']:.2f} | Return {strict['total_return']:+.2f}% "
        f"| Trades {strict['trades']}"
    )
    print(f"Deployment decision: {status}")
    if decision["failure_reasons"]:
        for reason in decision["failure_reasons"]:
            print(f"  - {reason}")
    artifact = structured.get("artifact", {})
    if artifact.get("path"):
        print(f"Artifact: {artifact['path']}")
    print()


def run_oos_validation(
    strategy_name: str,
    save_artifact: bool = True,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    artifact_time: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(strategy_name)
    params = config.get("params", {})
    risk = config.get("risk", {})
    strategy_cfg = config.get("strategy", {})

    symbol = strategy_cfg.get("symbol", "BTCUSDT")
    timeframe = strategy_cfg.get("timeframe", "1h")
    sl_pct = risk.get("stop_loss_pct", 0.015)
    tp_pct = risk.get("take_profit_pct", 0.015)
    max_hold = params.get("max_holding_period", 16)
    position_pct = risk.get("max_position_pct", 0.05)
    fee_per_side = config.get("execution", {}).get("fee_rate", 0.00055)

    models, model_folds, model_mode, meta = _load_models(strategy_name, params)
    val_end_ts = _resolve_validation_end(meta, model_folds, model_mode)

    feature_names_path = params.get(
        "feature_names_path",
        f"strategies/{strategy_name}/models/feature_names.json",
    )
    with open(feature_names_path, "r", encoding="utf-8") as handle:
        feature_names = json.load(handle)

    data_path = f"data/processed/{symbol}_{timeframe}_features.parquet"
    df = pd.read_parquet(data_path)
    engine = FeatureEngine(config={"symbol": symbol})
    df_feat = engine.compute_all_features(df)
    signals, _ = _build_signals(df, df_feat, feature_names, models, config)

    ts = to_utc_series(df["timestamp"])
    periods = _build_periods(ts.iloc[0], val_end_ts, ts.iloc[-1])
    scenario_results = _evaluate_scenarios(
        df=df,
        signals=signals,
        periods=periods,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        max_hold=max_hold,
        position_pct=position_pct,
        fee_per_side=fee_per_side,
    )

    generated_at = artifact_time or datetime.now(timezone.utc)
    structured = _build_structured_result(
        strategy_name=strategy_name,
        scenario_results=scenario_results,
        periods=periods,
        model_mode=model_mode,
        model_folds=model_folds,
        signal_count=int((signals == 1).sum()),
        generated_at=generated_at,
    )

    if save_artifact:
        structured = save_strategy_validation_artifact(
            structured,
            strategy_name=strategy_name,
            output_dir=output_dir,
            timestamp=generated_at,
        )

    return scenario_results, structured


def run_oos_validation_suite(
    strategy_names: list[str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc)
    strategy_results: dict[str, Any] = {}

    for strategy_name in strategy_names:
        _, structured = run_oos_validation(
            strategy_name,
            save_artifact=True,
            output_dir=output_dir,
            artifact_time=generated_at,
        )
        strategy_results[strategy_name] = structured
        _print_result_summary(strategy_name, structured)

    passed_count = sum(1 for result in strategy_results.values() if result["passed"])
    failed = [name for name, result in strategy_results.items() if not result["passed"]]

    summary = {
        "generated_at": generated_at.isoformat(),
        "gate_version": GATE_VERSION,
        "strategy_count": len(strategy_results),
        "passed_count": passed_count,
        "failed_count": len(strategy_results) - passed_count,
        "failed_strategies": failed,
        "strategies": strategy_results,
    }
    summary = save_validation_run_summary(
        summary,
        output_dir=output_dir,
        timestamp=generated_at,
    )

    print("=" * 80)
    print("OOS validation suite")
    print("=" * 80)
    print(f"Passed: {passed_count}/{len(strategy_results)}")
    if failed:
        print("Failed strategies: " + ", ".join(failed))
    print(f"Summary artifact: {summary['artifact']['path']}")
    print()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run out-of-sample validation.")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy name to validate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all active strategies from config/portfolio.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured JSON payload after the standard output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used for validation artifacts.",
    )
    args = parser.parse_args()

    if bool(args.strategy) == bool(args.all):
        parser.error("Specify exactly one of --strategy or --all.")

    if args.all:
        summary = run_oos_validation_suite(
            strategy_names=load_active_strategies(),
            output_dir=args.output_dir,
        )
        if args.json:
            print(JSON_RESULT_MARKER)
            print(json.dumps(summary, ensure_ascii=False))
        return

    _, structured = run_oos_validation(
        strategy_name=args.strategy,
        save_artifact=True,
        output_dir=args.output_dir,
    )
    _print_result_summary(args.strategy, structured)

    if args.json:
        print(JSON_RESULT_MARKER)
        print(json.dumps(structured, ensure_ascii=False))


if __name__ == "__main__":
    main()
