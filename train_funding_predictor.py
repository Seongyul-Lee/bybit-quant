"""펀딩비 예측 모델 학습.

Phase 0에서 생성한 8h 피처 데이터셋으로 LightGBM 회귀 모델을 학습.
8h 데이터 특성(~540건/fold)에 맞는 경량 파라미터 사용.

사용법:
    python train_funding_predictor.py
    python train_funding_predictor.py --symbol ETH
    python train_funding_predictor.py --optuna-trials 50
"""

import argparse
import json
import os
import shutil
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.logger import setup_logger

logger = setup_logger("funding_predictor")

# ─────────────────────────────────────────────
# 피처 / 타겟 정의
# ─────────────────────────────────────────────
FEATURES = [
    # 펀딩비 패턴 (핵심 — 상관관계 0.70~0.78)
    "fr_ma_3",           # r=0.777
    "fr_ma_7",           # r=0.742
    "fr_ma_21",          # r=0.701
    "fr_std_7",          # r=0.402
    "fr_zscore",         # r=0.261
    # 시장 변수 (보조)
    "return_8h",         # r=0.117
    "return_24h",        # r=0.205
    "volatility_24h",    # r=0.034
    "oi_change_8h",      # r=0.063
    "oi_change_24h",     # r=0.069
    "buy_ratio",         # r=-0.116
    # 시간 피처
    "hour_utc",
    "day_of_week",
]

TARGET = "next_funding_rate"

# 타겟 스케일링 상수 (펀딩비 ~0.0001 → ~1.0 스케일)
# Huber/regression 목적함수가 미세한 값에서 기울기를 잃지 않도록 함
TARGET_SCALE = 10000

# 8h 데이터 전용 기본 파라미터 (fold당 ~540건에 맞춤)
DEFAULT_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "huber",
    "metric": "mae",
    "n_estimators": 300,
    "num_leaves": 8,
    "min_child_samples": 10,
    "learning_rate": 0.03,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_depth": 4,
    "huber_delta": 0.5,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "num_threads": 1,
}


def load_data(symbol: str = "BTC") -> pd.DataFrame:
    """8h 펀딩비 피처 데이터를 로드하고 NaN 행을 제거한다."""
    path = f"data/processed/{symbol}USDT_8h_funding_features.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    required_cols = FEATURES + [TARGET]
    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"NaN 행 {dropped}건 제거 ({before} → {len(df)})")

    return df


def generate_folds(
    df: pd.DataFrame,
    train_months: int = 6,
    val_months: int = 1,
    embargo_bars: int = 1,
) -> list[dict]:
    """시간 기반 Walk-Forward fold 생성 (슬라이딩 윈도우).

    Args:
        df: timestamp 컬럼이 있는 데이터프레임.
        train_months: 학습 기간 (월).
        val_months: 검증 기간 (월).
        embargo_bars: 학습/검증 사이 제거할 봉 수.

    Returns:
        fold 리스트.
    """
    timestamps = pd.to_datetime(df["timestamp"])
    end_date = timestamps.iloc[-1]

    folds = []
    first_val_start = timestamps.iloc[0] + pd.DateOffset(months=train_months)
    val_start = first_val_start

    while True:
        val_end = val_start + pd.DateOffset(months=val_months)
        if val_end > end_date:
            break

        train_start = val_start - pd.DateOffset(months=train_months)
        train_mask = (timestamps >= train_start) & (timestamps < val_start)
        train_idx = df.index[train_mask].tolist()

        # embargo
        if len(train_idx) > embargo_bars:
            train_idx = train_idx[:-embargo_bars]

        val_mask = (timestamps >= val_start) & (timestamps < val_end)
        val_idx = df.index[val_mask].tolist()

        if len(train_idx) < 50 or len(val_idx) < 10:
            val_start = val_end
            continue

        folds.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_start": str(timestamps.iloc[train_idx[0]]),
            "train_end": str(timestamps.iloc[train_idx[-1]]),
            "val_start": str(val_start),
            "val_end": str(val_end),
        })
        val_start = val_end

    logger.info(f"Walk-Forward fold 생성: {len(folds)}개 (슬라이딩 {train_months}개월)")
    return folds


def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
) -> dict:
    """Optuna로 하이퍼파라미터 최적화 (8h 데이터 전용 탐색 범위)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **DEFAULT_PARAMS,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "num_leaves": trial.suggest_int("num_leaves", 4, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
            "huber_delta": trial.suggest_float("huber_delta", 0.1, 2.0),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        pred_val = model.predict(X_val)
        val_ic, _ = spearmanr(pred_val, y_val)
        if np.isnan(val_ic):
            return -1.0

        pred_train = model.predict(X_train)
        train_ic, _ = spearmanr(pred_train, y_train)
        if np.isnan(train_ic):
            train_ic = 0.0
        gap = train_ic - val_ic

        return val_ic - 0.5 * max(gap - 0.2, 0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = {**DEFAULT_PARAMS, **study.best_params}
    logger.info(f"Optuna 최적 IC: {study.best_value:.4f}")
    logger.info(f"최적 파라미터: {study.best_params}")

    return best_params


def compute_fold_extras(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """negative_recall, positive_recall 등 추가 메트릭 계산."""
    actual_neg = y_true < 0
    actual_pos = y_true >= 0
    pred_neg = y_pred < 0
    pred_pos = y_pred >= 0

    negative_recall = float(np.mean(pred_neg[actual_neg])) if actual_neg.sum() > 0 else 0.0
    positive_recall = float(np.mean(pred_pos[actual_pos])) if actual_pos.sum() > 0 else 0.0
    direction_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    return {
        "negative_recall": round(negative_recall, 4),
        "positive_recall": round(positive_recall, 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "n_actual_neg": int(actual_neg.sum()),
        "n_actual_pos": int(actual_pos.sum()),
    }


def _atomic_write_text(path: str, content: str) -> None:
    """텍스트 파일 atomic write."""
    dir_path = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        shutil.move(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _atomic_write_json(path: str, data) -> None:
    """JSON 파일 atomic write."""
    dir_path = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        shutil.move(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def train(
    symbol: str = "BTC",
    optuna_trials: int = 50,
    no_optuna: bool = False,
) -> dict:
    """펀딩비 예측 모델을 학습한다."""
    df = load_data(symbol)
    logger.info(f"데이터 로드: {symbol}USDT, {len(df)}건, "
                f"기간: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")

    folds = generate_folds(df)
    if not folds:
        raise ValueError("fold를 생성할 수 없습니다.")

    # 첫 fold에서 Optuna 또는 기본 파라미터 결정
    first = folds[0]
    X_train_0 = df.loc[first["train_idx"], FEATURES]
    y_train_0 = df.loc[first["train_idx"], TARGET]
    X_val_0 = df.loc[first["val_idx"], FEATURES]
    y_val_0 = df.loc[first["val_idx"], TARGET]

    if no_optuna:
        params = DEFAULT_PARAMS.copy()
        logger.info("기본 파라미터 사용")
    else:
        logger.info(f"Optuna 튜닝 ({optuna_trials} trials)...")
        params = optimize_hyperparams(
            X_train_0, y_train_0 * TARGET_SCALE,
            X_val_0, y_val_0 * TARGET_SCALE,
            optuna_trials,
        )

    # 모든 fold에서 학습
    fold_models = []
    folds_metrics = []

    for i, fold in enumerate(folds):
        X_train = df.loc[fold["train_idx"], FEATURES]
        y_train = df.loc[fold["train_idx"], TARGET] * TARGET_SCALE
        X_val = df.loc[fold["val_idx"], FEATURES]
        y_val = df.loc[fold["val_idx"], TARGET] * TARGET_SCALE

        logger.info(f"Fold {i}/{len(folds)-1}: "
                     f"train={fold['train_start'][:10]}~{fold['train_end'][:10]} "
                     f"({len(X_train)}건) | "
                     f"val={fold['val_start'][:10]}~{fold['val_end'][:10]} "
                     f"({len(X_val)}건)")

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)

        train_mae = float(np.mean(np.abs(pred_train - y_train)))
        val_mae = float(np.mean(np.abs(pred_val - y_val)))

        train_ic, _ = spearmanr(pred_train, y_train)
        val_ic, _ = spearmanr(pred_val, y_val)
        train_ic = float(train_ic) if not np.isnan(train_ic) else 0.0
        val_ic = float(val_ic) if not np.isnan(val_ic) else 0.0

        extras = compute_fold_extras(y_val.values, pred_val)
        gap = train_ic - val_ic

        metrics = {
            "fold": i,
            "train_mae": round(train_mae, 8),
            "val_mae": round(val_mae, 8),
            "train_ic": round(train_ic, 4),
            "val_ic": round(val_ic, 4),
            "overfit_gap": round(gap, 4),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": model.best_iteration_ if hasattr(model, "best_iteration_") else -1,
            "train_period": f"{fold['train_start']} ~ {fold['train_end']}",
            "val_period": f"{fold['val_start']} ~ {fold['val_end']}",
            # 호환 키
            "train_f1_macro": train_ic,
            "val_f1_macro": val_ic,
            "val_logloss": val_mae,
        }
        metrics.update(extras)

        logger.info(f"  Train IC: {train_ic:.4f} | Val IC: {val_ic:.4f} | "
                     f"Val MAE: {val_mae:.6f} | DirAcc: {extras['direction_accuracy']:.1%} | "
                     f"NegRec: {extras['negative_recall']:.1%}")

        fold_models.append(model)
        folds_metrics.append(metrics)

    # 모델 선택: Val IC > 0 인 fold 중 최신 우선
    eligible = [
        (i, m["val_ic"], m["overfit_gap"])
        for i, m in enumerate(folds_metrics)
        if m["val_ic"] > 0.01 and m["overfit_gap"] < 0.5
    ]

    if eligible:
        best = max(eligible, key=lambda x: (x[0], x[1]))
    else:
        # fallback: 최신 fold
        best = (len(folds) - 1, folds_metrics[-1]["val_ic"], folds_metrics[-1]["overfit_gap"])
        logger.warning("적합 fold 없음 — 최신 fold 사용")

    best_fold_idx = best[0]
    best_val_ic = best[1]
    selected_model = fold_models[best_fold_idx]

    logger.info(f"모델 선택: Fold {best_fold_idx} (Val IC: {best_val_ic:.4f})")

    # 피처 중요도
    importance = dict(zip(FEATURES, [int(x) for x in selected_model.feature_importances_]))

    # 저장
    save_dir = "strategies/funding_arb/models"
    os.makedirs(save_dir, exist_ok=True)

    _atomic_write_text(
        os.path.join(save_dir, "latest.txt"),
        selected_model.booster_.model_to_string(),
    )
    _atomic_write_json(os.path.join(save_dir, "feature_names.json"), FEATURES)

    serializable_params = {
        k: v for k, v in params.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }
    _atomic_write_json(os.path.join(save_dir, "best_params.json"), serializable_params)

    meta = {
        "mode": "regressor",
        "n_folds": len(folds),
        "best_fold_idx": best_fold_idx,
        "best_val_f1": best_val_ic,
        "folds_metrics": folds_metrics,
        "feature_importance": importance,
    }
    _atomic_write_json(os.path.join(save_dir, "training_meta.json"), meta)

    for i, fm in enumerate(fold_models):
        fold_path = os.path.join(save_dir, f"fold_{i:02d}.txt")
        _atomic_write_text(fold_path, fm.booster_.model_to_string())

    logger.info(f"모델 저장 완료: {save_dir} ({len(fold_models)}개 fold)")

    result = {
        "model": selected_model,
        "params": params,
        "feature_names": FEATURES,
        "folds_metrics": folds_metrics,
        "feature_importance": importance,
        "best_fold_idx": best_fold_idx,
        "best_val_f1": best_val_ic,
        "fold_models": fold_models,
    }

    _print_results(result, symbol)
    return result


def _print_results(result: dict, symbol: str) -> None:
    """학습 결과를 출력한다."""
    metrics_list = result["folds_metrics"]

    print(f"\n{'=' * 70}")
    print(f"펀딩비 예측 모델 학습 결과 ({symbol}USDT)")
    print(f"{'=' * 70}")

    print(f"\n{'Fold':>4} | {'Val IC':>8} | {'Val MAE':>10} | {'Dir Acc':>8} | "
          f"{'Neg Rec':>8} | {'Gap':>8}")
    print("-" * 65)

    for m in metrics_list:
        print(f"  {m['fold']:2d} | {m['val_ic']:8.4f} | {m['val_mae']:10.6f} | "
              f"{m['direction_accuracy']:7.1%} | {m['negative_recall']:7.1%} | "
              f"{m['overfit_gap']:8.4f}")

    avg_ic = np.mean([m["val_ic"] for m in metrics_list])
    avg_mae = np.mean([m["val_mae"] for m in metrics_list])
    avg_dir = np.mean([m["direction_accuracy"] for m in metrics_list])
    avg_neg = np.mean([m["negative_recall"] for m in metrics_list])
    print("-" * 65)
    print(f" AVG | {avg_ic:8.4f} | {avg_mae:10.6f} | {avg_dir:7.1%} | {avg_neg:7.1%}")

    best_idx = result["best_fold_idx"]
    print(f"\n선택된 모델: Fold {best_idx} (Val IC: {result['best_val_f1']:.4f})")

    importance = result["feature_importance"]
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n피처 중요도 (Top 5):")
    for name, imp in sorted_imp[:5]:
        print(f"  {name:20s}: {imp}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="펀딩비 예측 모델 학습")
    parser.add_argument("--symbol", default="BTC", choices=["BTC", "ETH"],
                        help="심볼 (기본: BTC)")
    parser.add_argument("--optuna-trials", type=int, default=50,
                        help="Optuna 시행 횟수 (기본: 50)")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Optuna 튜닝 비활성화 (기본 파라미터 사용)")
    args = parser.parse_args()

    train(
        symbol=args.symbol,
        optuna_trials=args.optuna_trials,
        no_optuna=args.no_optuna,
    )


if __name__ == "__main__":
    main()
