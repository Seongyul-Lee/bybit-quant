"""주기적 재학습 파이프라인.

최신 데이터로 모델을 재학습하고 OOS 검증 통과 시 모델을 교체한다.
검증 실패 시 기존 모델을 자동 복원한다.

사용법:
    python retrain.py --strategy btc_1h_momentum
    python retrain.py --all
    python retrain.py --all --dry-run
    python retrain.py --strategy btc_1h_momentum --skip-data-collection

자동 실행 (Windows Task Scheduler):
    매월 1일 UTC 01:00에 실행
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.utils.logger import setup_logger

logger = setup_logger("retrain")

# 모델 교체 최소 PF 비율 (기본값, config에서 오버라이드 가능)
DEFAULT_MIN_PF_RATIO = 0.9
RETRAIN_LOG_PATH = "retrain_log.json"


def load_portfolio_config() -> dict:
    """portfolio.yaml에서 활성 전략 목록을 로드."""
    with open("config/portfolio.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_strategy_config(strategy_name: str) -> dict:
    """전략별 config.yaml 로드."""
    path = f"strategies/{strategy_name}/config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_ccxt_symbol(file_symbol: str) -> str:
    """파일명 심볼을 ccxt 심볼로 변환. BTCUSDT → BTC/USDT:USDT."""
    # BTCUSDT → BTC/USDT:USDT, ETHUSDT → ETH/USDT:USDT
    base = file_symbol.replace("USDT", "")
    return f"{base}/USDT:USDT"


def get_last_timestamp(parquet_path: str) -> pd.Timestamp | None:
    """Parquet 파일의 마지막 타임스탬프를 반환."""
    if not os.path.exists(parquet_path):
        return None
    df = pd.read_parquet(parquet_path, columns=["timestamp"])
    if df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"]).max()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def collect_data(symbol: str, timeframe: str) -> bool:
    """심볼의 OHLCV + 펀딩비 + OI 데이터를 증분 수집.

    Args:
        symbol: 파일명 심볼 (예: BTCUSDT).
        timeframe: 타임프레임 (예: 1h).

    Returns:
        성공 여부.
    """
    from src.data.collector import BybitDataCollector
    from src.data.processor import DataProcessor

    ccxt_symbol = get_ccxt_symbol(symbol)
    collector = BybitDataCollector()

    # 1. OHLCV 증분 수집
    clean_symbol = ccxt_symbol.replace("/", "").replace(":", "")
    ohlcv_dir = f"data/raw/bybit/{clean_symbol}/{timeframe}"
    last_ts = None
    if os.path.exists(ohlcv_dir):
        # 모든 월별 파일에서 마지막 타임스탬프 찾기
        parquet_files = sorted(
            [f for f in os.listdir(ohlcv_dir) if f.endswith(".parquet")]
        )
        if parquet_files:
            last_file = os.path.join(ohlcv_dir, parquet_files[-1])
            last_ts = get_last_timestamp(last_file)

    if last_ts:
        since_str = (last_ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(f"OHLCV 증분 수집: {symbol} {timeframe} since {since_str}")
    else:
        since_str = "2024-01-01T00:00:00Z"
        logger.info(f"OHLCV 전체 수집: {symbol} {timeframe}")

    try:
        ohlcv_df = collector.fetch_ohlcv_bulk(ccxt_symbol, timeframe, since_str)
        if not ohlcv_df.empty:
            collector.save_ohlcv(ohlcv_df, ccxt_symbol, timeframe)
            logger.info(f"OHLCV 수집 완료: {len(ohlcv_df)}건")
        else:
            logger.info("OHLCV: 새 데이터 없음")
    except Exception as e:
        logger.error(f"OHLCV 수집 실패: {e}")
        return False

    # 2. 펀딩비 증분 수집
    fr_path = f"data/raw/bybit/{clean_symbol}/funding_rate.parquet"
    fr_last = get_last_timestamp(fr_path)
    fr_since = (
        (fr_last + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if fr_last
        else "2024-01-01T00:00:00Z"
    )

    try:
        fr_df = collector.fetch_funding_rate_bulk(ccxt_symbol, fr_since)
        if not fr_df.empty:
            if os.path.exists(fr_path):
                existing = pd.read_parquet(fr_path)
                fr_df = pd.concat([existing, fr_df], ignore_index=True)
                fr_df = fr_df.drop_duplicates(subset=["timestamp"]).sort_values(
                    "timestamp"
                ).reset_index(drop=True)
            os.makedirs(os.path.dirname(fr_path), exist_ok=True)
            fr_df.to_parquet(fr_path, index=False, compression="snappy")
            logger.info(f"펀딩비 수집 완료: {len(fr_df)}건 (누적)")
        else:
            logger.info("펀딩비: 새 데이터 없음")
    except Exception as e:
        logger.error(f"펀딩비 수집 실패: {e}")
        return False

    # 3. OI 증분 수집
    oi_path = f"data/raw/bybit/{clean_symbol}/open_interest_{timeframe}.parquet"
    oi_last = get_last_timestamp(oi_path)
    oi_since = (
        (oi_last + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if oi_last
        else "2024-01-01T00:00:00Z"
    )

    try:
        oi_df = collector.fetch_open_interest_bulk(ccxt_symbol, timeframe, oi_since)
        if not oi_df.empty:
            if os.path.exists(oi_path):
                existing = pd.read_parquet(oi_path)
                oi_df = pd.concat([existing, oi_df], ignore_index=True)
                oi_df = oi_df.drop_duplicates(subset=["timestamp"]).sort_values(
                    "timestamp"
                ).reset_index(drop=True)
            os.makedirs(os.path.dirname(oi_path), exist_ok=True)
            oi_df.to_parquet(oi_path, index=False, compression="snappy")
            logger.info(f"OI 수집 완료: {len(oi_df)}건 (누적)")
        else:
            logger.info("OI: 새 데이터 없음")
    except Exception as e:
        logger.error(f"OI 수집 실패: {e}")
        # OI는 필수가 아니므로 실패해도 계속 진행

    # 4. processed parquet 재생성
    try:
        all_ohlcv = []
        if os.path.exists(ohlcv_dir):
            for f in sorted(os.listdir(ohlcv_dir)):
                if f.endswith(".parquet"):
                    all_ohlcv.append(pd.read_parquet(os.path.join(ohlcv_dir, f)))

        if not all_ohlcv:
            logger.error("OHLCV 데이터 없음")
            return False

        combined = pd.concat(all_ohlcv, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values(
            "timestamp"
        ).reset_index(drop=True)

        processor = DataProcessor()
        timeframe_minutes = {"1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 60)
        processor.process_and_save(combined, symbol, timeframe, timeframe_minutes)
        logger.info(f"processed parquet 재생성 완료: {symbol}_{timeframe}")
    except Exception as e:
        logger.error(f"데이터 처리 실패: {e}")
        return False

    return True


def get_current_pv_pf(strategy_name: str) -> float | None:
    """현재 모델의 보수적 PV PF를 가져온다.

    retrain_log.json에서 최근 기록을 찾거나, 없으면 None 반환.
    """
    if os.path.exists(RETRAIN_LOG_PATH):
        with open(RETRAIN_LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
        for entry in reversed(logs):
            if entry["strategy"] == strategy_name and entry.get("new_pv_pf"):
                return entry["new_pv_pf"]
    return None


def run_oos_validation(strategy_name: str) -> dict | None:
    """OOS 검증을 실행하고 결과를 파싱.

    Returns:
        {"pv_pf": float, "pv_return": float, "pv_trades": int,
         "strict_oos_pf": float, "passed": bool} 또는 실패 시 None.
    """
    try:
        result = subprocess.run(
            [sys.executable, "oos_validation.py", "--strategy", strategy_name, "--json"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout
        if not output:
            logger.error(f"OOS 검증 출력 없음: {result.stderr}")
            return None

        # __JSON_RESULT__ 마커 이후의 JSON 파싱
        json_marker = "__JSON_RESULT__"
        if json_marker not in output:
            logger.error("OOS 검증: JSON 결과 마커 없음")
            logger.error(f"Output tail: {output[-300:]}")
            return None

        json_str = output.split(json_marker)[-1].strip()
        structured = json.loads(json_str)

        cons = structured.get("conservative", {})
        strict = structured.get("strict_oos", {})

        return {
            "pv_pf": cons.get("pv_pf", 0),
            "pv_return": cons.get("pv_return", 0),
            "pv_trades": cons.get("pv_trades", 0),
            "strict_oos_pf": strict.get("pf", 0),
            "passed": structured.get("passed", False),
        }
    except subprocess.TimeoutExpired:
        logger.error("OOS 검증 타임아웃 (300초)")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"OOS 검증 JSON 파싱 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"OOS 검증 실패: {e}")
        return None


def auto_select_ensemble_folds(
    training_meta: dict,
    max_overfit_gap: float = 0.3,
    n_folds: int = 2,
    min_pv_months: int = 2,
) -> list[int]:
    """학습 메타데이터만으로 앙상블 fold를 자동 선택.

    기준:
    1. val_f1_macro > 0 (검증에서 양의 F1)
    2. overfit_gap < max_overfit_gap (과적합 제한)
    3. 최신 fold의 val_end가 데이터 끝 - min_pv_months 이전 (PV 구간 확보)
    4. val_f1 상위 n_folds개 선택

    앙상블 PV 앵커 = max(selected_folds)의 val_end.
    따라서 가장 최신 fold가 너무 최근이면 PV 구간이 부족해진다.
    """
    folds = training_meta["folds_metrics"]
    if not folds:
        return [0]

    # 데이터 끝 시점 추정 (마지막 fold의 val_end)
    last_val_end_str = folds[-1].get("val_period", "").split(" ~ ")[-1].strip()
    if last_val_end_str:
        last_val_end = pd.Timestamp(last_val_end_str)
        pv_cutoff = last_val_end - pd.DateOffset(months=min_pv_months)
    else:
        pv_cutoff = None

    candidates = []
    for fm in folds:
        if fm.get("val_f1_macro", 0) <= 0:
            continue
        if fm.get("overfit_gap", 1.0) >= max_overfit_gap:
            continue

        # PV 구간 확보: 이 fold의 val_end가 너무 최근이면 제외
        if pv_cutoff:
            val_end_str = fm.get("val_period", "").split(" ~ ")[-1].strip()
            if val_end_str:
                val_end = pd.Timestamp(val_end_str)
                if val_end > pv_cutoff:
                    continue

        candidates.append(fm)

    if not candidates:
        # Fallback: gap 조건만 완화
        candidates = [
            fm for fm in folds
            if fm.get("val_f1_macro", 0) > 0
        ]
        # PV cutoff 적용
        if pv_cutoff:
            pv_ok = [
                fm for fm in candidates
                if pd.Timestamp(fm.get("val_period", "").split(" ~ ")[-1].strip()) <= pv_cutoff
            ]
            if pv_ok:
                candidates = pv_ok

    if not candidates:
        logger.warning("앙상블 후보 없음 — 마지막 PV-safe fold 사용")
        return [folds[-1]["fold"]]

    # 최신 fold 우선 (더 최근 데이터로 학습), 동점이면 val_f1 높은 순
    # 최근 절반의 fold에서만 선택 (오래된 fold는 현재 시장과 무관)
    total_folds = len(folds)
    recent_cutoff = total_folds // 2
    recent_candidates = [c for c in candidates if c["fold"] >= recent_cutoff]
    if len(recent_candidates) >= n_folds:
        candidates = recent_candidates

    candidates.sort(key=lambda x: (-x["fold"], -x["val_f1_macro"]))
    selected = [c["fold"] for c in candidates[:n_folds]]
    return sorted(selected)


def update_ensemble_folds(strategy_name: str, new_folds: list[int]) -> None:
    """config.yaml의 ensemble_folds를 업데이트 (원본 포맷 보존).

    정규식으로 ensemble_folds 행만 교체하여 주석/포맷을 유지한다.
    """
    import re

    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    folds_str = str(new_folds) if new_folds and len(new_folds) > 1 else "null"
    new_line = f"  ensemble_folds: {folds_str}"

    if "ensemble_folds:" in content:
        content = re.sub(
            r"  ensemble_folds:.*",
            new_line,
            content,
        )
    else:
        # ensemble_folds가 없으면 models_dir 앞에 추가
        content = content.replace(
            "  models_dir:",
            f"{new_line}\n  models_dir:",
        )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"ensemble_folds 업데이트: {new_folds}")


def _safe_restore_models(backup_dir: str, models_dir: str) -> None:
    """Windows 파일 핸들 문제를 고려한 안전한 모델 복원."""
    import time

    for attempt in range(3):
        try:
            if os.path.exists(models_dir):
                shutil.rmtree(models_dir)
            shutil.copytree(backup_dir, models_dir)
            return
        except (PermissionError, FileNotFoundError, OSError) as e:
            if attempt < 2:
                logger.warning(f"모델 복원 재시도 ({attempt+1}/3): {e}")
                time.sleep(1)
            else:
                raise


def _restore_ensemble_config(strategy_name: str, config_backup: str) -> None:
    """config.yaml을 백업된 내용으로 복원."""
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_backup)
    logger.info("config.yaml 복원 완료")


def retrain_strategy(
    strategy_name: str,
    dry_run: bool = False,
    skip_data: bool = False,
) -> dict[str, Any]:
    """단일 전략 재학습 파이프라인.

    1. 최신 데이터 수집 (OHLCV + 펀딩비 + OI)
    2. 기존 모델 백업
    3. 재학습 (config의 window_type/window_months 사용)
    4. 앙상블 fold 자동 선택
    5. OOS 검증 (보수적 비용)
    6. 검증 통과 시 → 모델 교체
    7. 검증 실패 시 → 기존 모델 복원

    Args:
        strategy_name: 전략 이름.
        dry_run: True면 검증만 하고 교체하지 않음.
        skip_data: True면 데이터 수집 건너뜀.

    Returns:
        결과 딕셔너리.
    """
    logger.info(f"{'='*60}")
    logger.info(f"재학습 시작: {strategy_name}" + (" (dry-run)" if dry_run else ""))
    logger.info(f"{'='*60}")

    config = load_strategy_config(strategy_name)
    strat_cfg = config.get("strategy", {})
    retrain_cfg = config.get("retrain", {})
    params_cfg = config.get("params", {})

    symbol = strat_cfg.get("symbol", "BTCUSDT")
    timeframe = strat_cfg.get("timeframe", "1h")
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
        "old_ensemble": params_cfg.get("ensemble_folds"),
        "new_ensemble": None,
        "window_type": window_type,
        "window_months": window_months,
        "reason": "",
    }

    models_dir = f"strategies/{strategy_name}/models"
    backup_dir = f"strategies/{strategy_name}/models_backup_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    # config.yaml 백업 (dry-run 복원용)
    config_path = f"strategies/{strategy_name}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_backup = f.read()

    # Step 1: 데이터 수집
    if not skip_data:
        logger.info("Step 1: 데이터 수집...")
        if not collect_data(symbol, timeframe):
            result["reason"] = "데이터 수집 실패"
            logger.error(result["reason"])
            return result
    else:
        logger.info("Step 1: 데이터 수집 건너뜀 (--skip-data-collection)")

    # Step 2: 기존 OOS 검증 (비교 기준)
    logger.info("Step 2: 기존 모델 OOS 검증...")
    old_oos = run_oos_validation(strategy_name)
    if old_oos:
        result["old_pv_pf"] = old_oos["pv_pf"]
        logger.info(f"기존 모델 보수적 PV PF: {old_oos['pv_pf']:.2f}")
    else:
        logger.warning("기존 모델 OOS 검증 실패 — 비교 없이 진행")

    # Step 3: 모델 백업
    logger.info(f"Step 3: 모델 백업 → {backup_dir}")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(models_dir, backup_dir)

    # Step 4: 재학습
    logger.info(f"Step 4: 재학습 ({window_type}, {window_months}개월)...")
    train_cmd = [
        sys.executable, "train_lgbm.py",
        "--strategy", strategy_name,
    ]

    if window_type == "sliding":
        train_cmd.extend(["--sliding-window", "--sliding-window-months", str(window_months)])

    # 파라미터 결정: Optuna vs 기존 파라미터 로드 vs 기본값
    if use_optuna:
        optuna_trials = retrain_cfg.get("optuna_trials", 50)
        train_cmd.extend(["--optuna-trials", str(optuna_trials)])
    else:
        # 기존 best_params.json 로드
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
            raise RuntimeError(f"학습 실패: {train_result.stderr}")
        logger.info("재학습 완료")
    except Exception as e:
        logger.error(f"재학습 실패: {e}")
        # 백업 복원
        logger.info("백업에서 모델 복원...")
        _safe_restore_models(backup_dir, models_dir)
        result["reason"] = f"재학습 실패: {e}"
        return result

    # Step 5: 앙상블 fold 자동 선택
    if auto_ensemble:
        logger.info("Step 5: 앙상블 fold 자동 선택...")
        meta_path = os.path.join(models_dir, "training_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        new_folds = auto_select_ensemble_folds(meta, n_folds=ensemble_n_folds)
        result["new_ensemble"] = new_folds
        logger.info(f"선택된 앙상블 folds: {new_folds}")
        update_ensemble_folds(strategy_name, new_folds)
    else:
        logger.info("Step 5: 앙상블 자동 선택 비활성화")

    # Step 6: OOS 검증
    logger.info("Step 6: 새 모델 OOS 검증...")
    new_oos = run_oos_validation(strategy_name)
    if new_oos is None:
        logger.error("새 모델 OOS 검증 실패")
        # 백업 복원
        logger.info("백업에서 모델 복원...")
        _safe_restore_models(backup_dir, models_dir)
        _restore_ensemble_config(strategy_name, config_backup)
        result["reason"] = "OOS 검증 실패"
        return result

    result["new_pv_pf"] = new_oos["pv_pf"]
    result["strict_oos_pf"] = new_oos["strict_oos_pf"]

    # Step 7: 교체 판단
    should_replace = True
    if old_oos and old_oos["pv_pf"] > 0:
        if new_oos["pv_pf"] < old_oos["pv_pf"] * min_pf_ratio:
            should_replace = False
            result["reason"] = (
                f"PV PF 하락 과도: {old_oos['pv_pf']:.2f} → {new_oos['pv_pf']:.2f} "
                f"(기준: ×{min_pf_ratio}={old_oos['pv_pf']*min_pf_ratio:.2f})"
            )
    elif not new_oos["passed"]:
        should_replace = False
        result["reason"] = "OOS 성공 기준 미충족"

    if dry_run:
        # dry-run: 결과만 출력하고 모델 + config 복원
        logger.info(f"[DRY-RUN] 교체 {'예정' if should_replace else '불가'}")
        logger.info("백업에서 모델 복원 (dry-run)...")
        _safe_restore_models(backup_dir, models_dir)
        # config 복원: 원래 ensemble_folds로 되돌림
        _restore_ensemble_config(strategy_name, config_backup)
        result["status"] = "dry_run"
        result["reason"] = f"dry-run: {'교체 예정' if should_replace else '교체 불가'}"
        return result

    if should_replace:
        result["status"] = "updated"
        result["reason"] = (
            f"모델 교체 완료: PV PF {result['old_pv_pf'] or 'N/A'} → {new_oos['pv_pf']:.2f}"
        )
        logger.info(f"모델 교체 완료! PV PF: {new_oos['pv_pf']:.2f}")
        # 백업 유지 (롤백용)
    else:
        result["status"] = "kept"
        logger.info(f"기존 모델 유지: {result['reason']}")
        # 백업 복원
        _safe_restore_models(backup_dir, models_dir)
        _restore_ensemble_config(strategy_name, config_backup)

    return result


def save_retrain_log(entry: dict) -> None:
    """재학습 결과를 retrain_log.json에 추가."""
    logs = []
    if os.path.exists(RETRAIN_LOG_PATH):
        with open(RETRAIN_LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)

    logs.append(entry)

    with open(RETRAIN_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    logger.info(f"재학습 로그 저장: {RETRAIN_LOG_PATH}")


def send_notification(results: list[dict]) -> None:
    """텔레그램 알림 전송 (설정된 경우)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    try:
        import urllib.request
        import urllib.parse

        lines = ["📊 재학습 파이프라인 결과\n"]
        for r in results:
            status_emoji = {"updated": "✅", "kept": "⏸️", "failed": "❌", "dry_run": "🔍"}
            emoji = status_emoji.get(r["status"], "❓")
            lines.append(
                f"{emoji} {r['strategy']}: {r['status']}\n"
                f"  PV PF: {r.get('old_pv_pf', 'N/A')} → {r.get('new_pv_pf', 'N/A')}\n"
                f"  {r.get('reason', '')}"
            )

        message = "\n".join(lines)
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
        urllib.request.urlopen(url, data, timeout=10)
        logger.info("텔레그램 알림 전송 완료")
    except Exception as e:
        logger.warning(f"텔레그램 알림 실패: {e}")


def main() -> None:
    """재학습 CLI 진입점."""
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(description="주기적 재학습 파이프라인")
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="재학습할 전략 이름 (미지정 시 --all 필요)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="모든 활성 전략 재학습"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="검증만 실행하고 모델 교체하지 않음"
    )
    parser.add_argument(
        "--skip-data-collection", action="store_true",
        help="데이터 수집 건너뜀 (이미 최신 데이터가 있을 때)"
    )

    args = parser.parse_args()

    if not args.strategy and not args.all:
        parser.error("--strategy 또는 --all을 지정하세요.")

    # 전략 목록 결정
    if args.all:
        portfolio = load_portfolio_config()
        strategies = portfolio["portfolio"]["active_strategies"]
    else:
        strategies = [args.strategy]

    # 데이터 수집: 심볼 중복 방지 (BTC 전략이 2개여도 수집은 1번)
    if not args.skip_data_collection:
        collected_symbols = set()
        for strat_name in strategies:
            config = load_strategy_config(strat_name)
            symbol = config.get("strategy", {}).get("symbol", "BTCUSDT")
            timeframe = config.get("strategy", {}).get("timeframe", "1h")
            key = f"{symbol}_{timeframe}"
            if key not in collected_symbols:
                logger.info(f"데이터 수집: {symbol} {timeframe}")
                collect_data(symbol, timeframe)
                collected_symbols.add(key)

    # 전략별 재학습
    results = []
    for strat_name in strategies:
        config = load_strategy_config(strat_name)
        retrain_cfg = config.get("retrain", {})
        if not retrain_cfg.get("enabled", True):
            logger.info(f"{strat_name}: 재학습 비활성화 — 건너뜀")
            continue

        result = retrain_strategy(
            strat_name,
            dry_run=args.dry_run,
            skip_data=True,  # 위에서 이미 수집함
        )
        results.append(result)
        save_retrain_log(result)

    # 결과 요약
    print("\n" + "=" * 60)
    print("재학습 파이프라인 결과")
    print("=" * 60)
    for r in results:
        status_map = {"updated": "교체", "kept": "유지", "failed": "실패", "dry_run": "검증만"}
        old_pf = f"{r['old_pv_pf']:.2f}" if r.get("old_pv_pf") else "N/A"
        new_pf = f"{r['new_pv_pf']:.2f}" if r.get("new_pv_pf") else "N/A"
        print(
            f"  {r['strategy']}: [{status_map.get(r['status'], r['status'])}] "
            f"PV PF {old_pf} → {new_pf} | {r.get('reason', '')}"
        )
    print("=" * 60)

    # 알림
    send_notification(results)


if __name__ == "__main__":
    main()
