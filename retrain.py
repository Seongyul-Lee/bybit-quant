"""Retraining pipeline for active ML strategies."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.utils.logger import setup_logger
from src.validation.gate import decide_model_replacement

logger = setup_logger("retrain")

DEFAULT_MIN_PF_RATIO = 0.9
RETRAIN_LOG_PATH = "retrain_log.json"
JSON_RESULT_MARKER = "__JSON_RESULT__"


def load_portfolio_config() -> dict[str, Any]:
    with open("config/portfolio.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_strategy_config(strategy_name: str) -> dict[str, Any]:
    path = f"strategies/{strategy_name}/config.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_ccxt_symbol(file_symbol: str) -> str:
    base = file_symbol.replace("USDT", "")
    return f"{base}/USDT:USDT"


def get_last_timestamp(parquet_path: str) -> pd.Timestamp | None:
    if not os.path.exists(parquet_path):
        return None

    df = pd.read_parquet(parquet_path, columns=["timestamp"])
    if df.empty:
        return None

    timestamp = pd.to_datetime(df["timestamp"]).max()
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp


def collect_data(symbol: str, timeframe: str) -> bool:
    """Collect incremental OHLCV, funding rate, and OI, then rebuild processed data."""

    from src.data.collector import BybitDataCollector
    from src.data.processor import DataProcessor

    ccxt_symbol = get_ccxt_symbol(symbol)
    collector = BybitDataCollector()
    clean_symbol = ccxt_symbol.replace("/", "").replace(":", "")
    ohlcv_dir = f"data/raw/bybit/{clean_symbol}/{timeframe}"

    last_ts = None
    if os.path.exists(ohlcv_dir):
        parquet_files = sorted(
            filename for filename in os.listdir(ohlcv_dir) if filename.endswith(".parquet")
        )
        if parquet_files:
            last_file = os.path.join(ohlcv_dir, parquet_files[-1])
            last_ts = get_last_timestamp(last_file)

    if last_ts is not None:
        since_str = (last_ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("Collect OHLCV incrementally: %s %s since %s", symbol, timeframe, since_str)
    else:
        since_str = "2024-01-01T00:00:00Z"
        logger.info("Collect OHLCV from scratch: %s %s", symbol, timeframe)

    try:
        ohlcv_df = collector.fetch_ohlcv_bulk(ccxt_symbol, timeframe, since_str)
        if not ohlcv_df.empty:
            collector.save_ohlcv(ohlcv_df, ccxt_symbol, timeframe)
            logger.info("OHLCV rows saved: %s", len(ohlcv_df))
        else:
            logger.info("No new OHLCV rows")
    except Exception as exc:
        logger.error("OHLCV collection failed: %s", exc)
        return False

    funding_path = f"data/raw/bybit/{clean_symbol}/funding_rate.parquet"
    funding_last = get_last_timestamp(funding_path)
    funding_since = (
        (funding_last + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if funding_last is not None
        else "2024-01-01T00:00:00Z"
    )

    try:
        funding_df = collector.fetch_funding_rate_bulk(ccxt_symbol, funding_since)
        if not funding_df.empty:
            if os.path.exists(funding_path):
                existing = pd.read_parquet(funding_path)
                funding_df = pd.concat([existing, funding_df], ignore_index=True)
                funding_df = (
                    funding_df.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            os.makedirs(os.path.dirname(funding_path), exist_ok=True)
            funding_df.to_parquet(funding_path, index=False, compression="snappy")
            logger.info("Funding rows saved: %s", len(funding_df))
        else:
            logger.info("No new funding rows")
    except Exception as exc:
        logger.error("Funding collection failed: %s", exc)
        return False

    oi_path = f"data/raw/bybit/{clean_symbol}/open_interest_{timeframe}.parquet"
    oi_last = get_last_timestamp(oi_path)
    oi_since = (
        (oi_last + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if oi_last is not None
        else "2024-01-01T00:00:00Z"
    )

    try:
        oi_df = collector.fetch_open_interest_bulk(ccxt_symbol, timeframe, oi_since)
        if not oi_df.empty:
            if os.path.exists(oi_path):
                existing = pd.read_parquet(oi_path)
                oi_df = pd.concat([existing, oi_df], ignore_index=True)
                oi_df = (
                    oi_df.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            os.makedirs(os.path.dirname(oi_path), exist_ok=True)
            oi_df.to_parquet(oi_path, index=False, compression="snappy")
            logger.info("OI rows saved: %s", len(oi_df))
        else:
            logger.info("No new OI rows")
    except Exception as exc:
        logger.error("OI collection failed: %s", exc)

    try:
        all_ohlcv = []
        if os.path.exists(ohlcv_dir):
            for filename in sorted(os.listdir(ohlcv_dir)):
                if filename.endswith(".parquet"):
                    all_ohlcv.append(pd.read_parquet(os.path.join(ohlcv_dir, filename)))

        if not all_ohlcv:
            logger.error("No OHLCV source files found")
            return False

        combined = pd.concat(all_ohlcv, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        processor = DataProcessor()
        timeframe_minutes = {"1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
        processor.process_and_save(combined, symbol, timeframe, timeframe_minutes)
        logger.info("Processed parquet rebuilt: %s_%s", symbol, timeframe)
    except Exception as exc:
        logger.error("Processed parquet rebuild failed: %s", exc)
        return False

    return True


def run_oos_validation(strategy_name: str) -> dict[str, Any] | None:
    """Run OOS validation and parse the structured payload emitted by the script."""

    try:
        result = subprocess.run(
            [sys.executable, "oos_validation.py", "--strategy", strategy_name, "--json"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        logger.error("OOS validation timed out after 300 seconds")
        return None
    except Exception as exc:
        logger.error("OOS validation process failed: %s", exc)
        return None

    output = result.stdout or ""
    if result.returncode != 0:
        logger.error("OOS validation exited with code %s: %s", result.returncode, result.stderr)
        return None
    if JSON_RESULT_MARKER not in output:
        logger.error("OOS validation JSON marker missing")
        logger.error("Output tail: %s", output[-300:])
        return None

    try:
        payload = json.loads(output.split(JSON_RESULT_MARKER)[-1].strip())
    except json.JSONDecodeError as exc:
        logger.error("OOS validation JSON decode failed: %s", exc)
        return None

    conservative = payload.get("conservative", {})
    strict_oos = payload.get("strict_oos", {})
    decision = payload.get("deployment_decision", {})
    artifact = payload.get("artifact", {})

    return {
        "pv_pf": float(conservative.get("pv_pf", 0) or 0),
        "pv_return": float(conservative.get("pv_return", 0) or 0),
        "pv_trades": int(conservative.get("pv_trades", 0) or 0),
        "strict_oos_pf": float(strict_oos.get("pf", 0) or 0),
        "passed": bool(payload.get("passed", False)),
        "failure_reasons": decision.get(
            "failure_reasons",
            payload.get("failure_reasons", []),
        ),
        "gate_version": payload.get("gate_version"),
        "artifact_path": artifact.get("path"),
    }


def auto_select_ensemble_folds(
    training_meta: dict[str, Any],
    max_overfit_gap: float = 0.3,
    n_folds: int = 2,
    min_pv_months: int = 2,
) -> list[int]:
    """Select recent usable folds automatically from training metadata."""

    folds = training_meta["folds_metrics"]
    if not folds:
        return [0]

    last_val_end_str = folds[-1].get("val_period", "").split(" ~ ")[-1].strip()
    pv_cutoff = None
    if last_val_end_str:
        last_val_end = pd.Timestamp(last_val_end_str)
        pv_cutoff = last_val_end - pd.DateOffset(months=min_pv_months)

    candidates = []
    for fold_meta in folds:
        if fold_meta.get("val_f1_macro", 0) <= 0:
            continue
        if fold_meta.get("overfit_gap", 1.0) >= max_overfit_gap:
            continue
        if pv_cutoff is not None:
            val_end_str = fold_meta.get("val_period", "").split(" ~ ")[-1].strip()
            if val_end_str and pd.Timestamp(val_end_str) > pv_cutoff:
                continue
        candidates.append(fold_meta)

    if not candidates:
        candidates = [fold_meta for fold_meta in folds if fold_meta.get("val_f1_macro", 0) > 0]
        if pv_cutoff is not None:
            pv_safe = []
            for fold_meta in candidates:
                val_end_str = fold_meta.get("val_period", "").split(" ~ ")[-1].strip()
                if val_end_str and pd.Timestamp(val_end_str) <= pv_cutoff:
                    pv_safe.append(fold_meta)
            if pv_safe:
                candidates = pv_safe

    if not candidates:
        logger.warning("No eligible ensemble candidate found; falling back to latest fold")
        return [folds[-1]["fold"]]

    total_folds = len(folds)
    recent_cutoff = total_folds // 2
    recent_candidates = [candidate for candidate in candidates if candidate["fold"] >= recent_cutoff]
    if len(recent_candidates) >= n_folds:
        candidates = recent_candidates

    candidates.sort(key=lambda item: (-item["fold"], -item["val_f1_macro"]))
    selected = [candidate["fold"] for candidate in candidates[:n_folds]]
    return sorted(selected)


def update_ensemble_folds(strategy_name: str, new_folds: list[int]) -> None:
    """Update only the ensemble_folds line in a strategy config."""

    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    folds_value = str(new_folds) if len(new_folds) > 1 else "null"
    new_line = f"  ensemble_folds: {folds_value}"

    if "ensemble_folds:" in content:
        content = re.sub(r"  ensemble_folds:.*", new_line, content)
    else:
        content = content.replace("  models_dir:", f"{new_line}\n  models_dir:")

    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(content)

    logger.info("Updated ensemble_folds for %s: %s", strategy_name, new_folds)


def _safe_restore_models(backup_dir: str, models_dir: str) -> None:
    for attempt in range(3):
        try:
            if os.path.exists(models_dir):
                shutil.rmtree(models_dir)
            shutil.copytree(backup_dir, models_dir)
            return
        except (PermissionError, FileNotFoundError, OSError) as exc:
            if attempt >= 2:
                raise
            logger.warning("Model restore retry %s/3: %s", attempt + 1, exc)
            time.sleep(1)


def _restore_ensemble_config(strategy_name: str, config_backup: str) -> None:
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(config_backup)
    logger.info("Restored config.yaml for %s", strategy_name)


def retrain_strategy(
    strategy_name: str,
    dry_run: bool = False,
    skip_data: bool = False,
) -> dict[str, Any]:
    """Retrain one strategy and replace models only if the new gate allows it."""

    logger.info("%s", "=" * 60)
    logger.info("Retrain start: %s%s", strategy_name, " (dry-run)" if dry_run else "")
    logger.info("%s", "=" * 60)

    config = load_strategy_config(strategy_name)
    strategy_cfg = config.get("strategy", {})
    retrain_cfg = config.get("retrain", {})
    params_cfg = config.get("params", {})

    symbol = strategy_cfg.get("symbol", "BTCUSDT")
    timeframe = strategy_cfg.get("timeframe", "1h")
    window_type = retrain_cfg.get("window_type", "expanding")
    window_months = retrain_cfg.get("window_months", 15)
    min_pf_ratio = retrain_cfg.get("min_pf_ratio", DEFAULT_MIN_PF_RATIO)
    auto_ensemble = retrain_cfg.get("auto_ensemble", True)
    ensemble_n_folds = retrain_cfg.get("ensemble_n_folds", 2)
    use_optuna = retrain_cfg.get("use_optuna", False)

    result = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "strategy": strategy_name,
        "status": "failed",
        "old_pv_pf": None,
        "new_pv_pf": None,
        "strict_oos_pf": None,
        "old_ensemble": params_cfg.get("ensemble_folds"),
        "new_ensemble": None,
        "window_type": window_type,
        "window_months": window_months,
        "gate_version": None,
        "gate_passed": False,
        "gate_failure_reasons": [],
        "validation_artifact": None,
        "reason": "",
    }

    models_dir = f"strategies/{strategy_name}/models"
    backup_dir = (
        f"strategies/{strategy_name}/models_backup_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    )

    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config_backup = handle.read()

    if not skip_data:
        logger.info("Step 1: collect data")
        if not collect_data(symbol, timeframe):
            result["reason"] = "data collection failed"
            logger.error(result["reason"])
            return result
    else:
        logger.info("Step 1: skip data collection")

    logger.info("Step 2: validate current model")
    old_oos = run_oos_validation(strategy_name)
    if old_oos is not None:
        result["old_pv_pf"] = old_oos["pv_pf"]
        logger.info("Current model PV PF: %.2f", old_oos["pv_pf"])
    else:
        logger.warning("Current model validation failed; proceeding without baseline comparison")

    logger.info("Step 3: backup models -> %s", backup_dir)
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(models_dir, backup_dir)

    logger.info("Step 4: train model (%s, %s months)", window_type, window_months)
    train_cmd = [sys.executable, "train_lgbm.py", "--strategy", strategy_name]
    if window_type == "sliding":
        train_cmd.extend(["--sliding-window", "--sliding-window-months", str(window_months)])

    if use_optuna:
        optuna_trials = retrain_cfg.get("optuna_trials", 50)
        train_cmd.extend(["--optuna-trials", str(optuna_trials)])
    else:
        best_params_path = os.path.join(models_dir, "best_params.json")
        if os.path.exists(best_params_path):
            train_cmd.append("--load-params")
        else:
            train_cmd.append("--no-optuna")

    try:
        train_result = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if train_result.returncode != 0:
            raise RuntimeError(train_result.stderr or "training command failed")
        logger.info("Training completed")
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        logger.info("Restoring backup models")
        _safe_restore_models(backup_dir, models_dir)
        result["reason"] = f"training failed: {exc}"
        return result

    if auto_ensemble:
        logger.info("Step 5: auto-select ensemble folds")
        meta_path = os.path.join(models_dir, "training_meta.json")
        with open(meta_path, "r", encoding="utf-8") as handle:
            training_meta = json.load(handle)

        new_folds = auto_select_ensemble_folds(training_meta, n_folds=ensemble_n_folds)
        result["new_ensemble"] = new_folds
        logger.info("Selected ensemble folds: %s", new_folds)
        update_ensemble_folds(strategy_name, new_folds)
    else:
        logger.info("Step 5: auto-select ensemble disabled")

    logger.info("Step 6: validate new model")
    new_oos = run_oos_validation(strategy_name)
    if new_oos is None:
        logger.error("New model validation failed to produce a result")
        logger.info("Restoring backup models and config")
        _safe_restore_models(backup_dir, models_dir)
        _restore_ensemble_config(strategy_name, config_backup)
        result["reason"] = "validation failed"
        return result

    result["new_pv_pf"] = new_oos["pv_pf"]
    result["strict_oos_pf"] = new_oos["strict_oos_pf"]
    result["gate_version"] = new_oos.get("gate_version")
    result["gate_passed"] = new_oos.get("passed", False)
    result["gate_failure_reasons"] = new_oos.get("failure_reasons", [])
    result["validation_artifact"] = new_oos.get("artifact_path")

    logger.info("Step 7: replacement decision")
    should_replace, decision_reason = decide_model_replacement(
        old_oos=old_oos,
        new_oos=new_oos,
        min_pf_ratio=min_pf_ratio,
    )
    result["reason"] = decision_reason

    if dry_run:
        logger.info("[DRY-RUN] replacement %s", "approved" if should_replace else "rejected")
        logger.info("Restoring backup models and config for dry-run")
        _safe_restore_models(backup_dir, models_dir)
        _restore_ensemble_config(strategy_name, config_backup)
        result["status"] = "dry_run"
        result["reason"] = f"dry-run: {decision_reason}"
        return result

    if should_replace:
        result["status"] = "updated"
        logger.info("Model replacement approved: %s", decision_reason)
    else:
        result["status"] = "kept"
        logger.info("Keeping current model: %s", decision_reason)
        _safe_restore_models(backup_dir, models_dir)
        _restore_ensemble_config(strategy_name, config_backup)

    return result


def save_retrain_log(entry: dict[str, Any]) -> None:
    logs = []
    if os.path.exists(RETRAIN_LOG_PATH):
        with open(RETRAIN_LOG_PATH, "r", encoding="utf-8") as handle:
            logs = json.load(handle)

    logs.append(entry)

    with open(RETRAIN_LOG_PATH, "w", encoding="utf-8") as handle:
        json.dump(logs, handle, indent=2, ensure_ascii=False)

    logger.info("Retrain log written: %s", RETRAIN_LOG_PATH)


def send_notification(results: list[dict[str, Any]]) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    try:
        import urllib.parse
        import urllib.request

        lines = ["Retrain pipeline results", ""]
        for result in results:
            lines.append(
                f"{result['strategy']}: {result['status']}\n"
                f"  PV PF: {result.get('old_pv_pf', 'N/A')} -> {result.get('new_pv_pf', 'N/A')}\n"
                f"  Gate: {'PASS' if result.get('gate_passed') else 'FAIL'}\n"
                f"  {result.get('reason', '')}"
            )

        message = "\n".join(lines)
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
        urllib.request.urlopen(url, data, timeout=10)
        logger.info("Telegram notification sent")
    except Exception as exc:
        logger.warning("Telegram notification failed: %s", exc)


def main() -> None:
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(description="Retrain active strategies.")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="One strategy to retrain.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Retrain all active strategies from config/portfolio.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and validate, but restore the original models afterwards.",
    )
    parser.add_argument(
        "--skip-data-collection",
        action="store_true",
        help="Skip data collection if local data is already up to date.",
    )
    args = parser.parse_args()

    if not args.strategy and not args.all:
        parser.error("Specify --strategy or --all.")
    if args.strategy and args.all:
        parser.error("Use only one of --strategy or --all.")

    if args.all:
        portfolio = load_portfolio_config()
        strategies = portfolio["portfolio"]["active_strategies"]
    else:
        strategies = [args.strategy]

    if not args.skip_data_collection:
        collected = set()
        for strategy_name in strategies:
            strategy_config = load_strategy_config(strategy_name)
            symbol = strategy_config.get("strategy", {}).get("symbol", "BTCUSDT")
            timeframe = strategy_config.get("strategy", {}).get("timeframe", "1h")
            key = f"{symbol}_{timeframe}"
            if key in collected:
                continue
            logger.info("Collect shared data for %s %s", symbol, timeframe)
            collect_data(symbol, timeframe)
            collected.add(key)

    results = []
    for strategy_name in strategies:
        strategy_config = load_strategy_config(strategy_name)
        retrain_cfg = strategy_config.get("retrain", {})
        if not retrain_cfg.get("enabled", True):
            logger.info("%s: retrain disabled, skipping", strategy_name)
            continue

        result = retrain_strategy(
            strategy_name=strategy_name,
            dry_run=args.dry_run,
            skip_data=True,
        )
        results.append(result)
        save_retrain_log(result)

    print("\n" + "=" * 60)
    print("Retrain pipeline results")
    print("=" * 60)
    for result in results:
        old_pf = f"{result['old_pv_pf']:.2f}" if result.get("old_pv_pf") else "N/A"
        new_pf = f"{result['new_pv_pf']:.2f}" if result.get("new_pv_pf") else "N/A"
        print(
            f"  {result['strategy']}: [{result['status']}] "
            f"PV PF {old_pf} -> {new_pf} | {result.get('reason', '')}"
        )
    print("=" * 60)

    send_notification(results)


if __name__ == "__main__":
    main()
