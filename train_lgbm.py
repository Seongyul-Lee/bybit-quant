"""LightGBM 모델 학습 CLI 스크립트.

사용법:
    python train_lgbm.py --strategy btc_1h_momentum --no-optuna
    python train_lgbm.py --strategy eth_1h_momentum --no-optuna
    python train_lgbm.py --symbol BTCUSDT --timeframe 1h --no-optuna  # 하위 호환
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from src.utils.logger import setup_logger
from strategies._common.features import FeatureEngine
from strategies._common.labeler import TripleBarrierLabeler
from strategies._common.trainer import WalkForwardTrainer
from strategies._common.evaluator import ModelEvaluator

logger = setup_logger("train_lgbm")


def main() -> None:
    """학습 CLI 진입점."""
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(description="LightGBM 모델 학습")
    parser.add_argument("--strategy", type=str, default="btc_1h_momentum",
                        help="전략 이름 (기본: btc_1h_momentum)")
    parser.add_argument("--symbol", type=str, default=None, help="심볼 (미지정 시 전략 config에서 로드)")
    parser.add_argument("--timeframe", type=str, default=None, help="타임프레임 (미지정 시 전략 config에서 로드)")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Optuna 시행 수 (기본: 50)")
    parser.add_argument("--no-optuna", action="store_true", help="Optuna 튜닝 비활성화")
    parser.add_argument("--min-train-months", type=int, default=6, help="최소 학습 기간 (월)")
    parser.add_argument("--val-months", type=int, default=1, help="검증 기간 (월)")
    parser.add_argument("--embargo-bars", type=int, default=24, help="Embargo 봉 수")
    parser.add_argument(
        "--upper-barrier", type=float, default=1.5, help="상단 배리어 ATR 배수"
    )
    parser.add_argument(
        "--lower-barrier", type=float, default=1.5, help="하단 배리어 ATR 배수"
    )
    parser.add_argument(
        "--max-holding", type=int, default=24, help="최대 보유 기간 (봉 수)"
    )
    parser.add_argument(
        "--use-all-features", action="store_true",
        help="피처 선별 없이 전체 피처 사용 (기본: 선별된 ~18개 사용)"
    )
    parser.add_argument(
        "--corr-threshold", type=float, default=0.9,
        help="상관관계 제거 기준 (기본: 0.9, 0이면 비활성화)"
    )
    parser.add_argument(
        "--sliding-window", action="store_true",
        help="슬라이딩 윈도우 사용 (기본: 확장 윈도우)"
    )
    parser.add_argument(
        "--sliding-window-months", type=int, default=12,
        help="슬라이딩 윈도우 학습 데이터 최대 기간 (월, 기본: 12)"
    )
    parser.add_argument(
        "--load-params", action="store_true",
        help="기존 best_params.json에서 하이퍼파라미터 로드 (Optuna 재실행 없이 동일 파라미터 사용)"
    )
    parser.add_argument(
        "--exclude-gaps", action="store_true", default=True,
        help="gap fill 구간을 학습에서 제외 (기본: True)"
    )
    parser.add_argument(
        "--no-exclude-gaps", action="store_true",
        help="gap fill 구간 제외 비활성화"
    )
    parser.add_argument(
        "--gap-buffer-bars", type=int, default=24,
        help="gap 구간 후 추가 제외할 봉 수 (기본: 24, 라벨 오염 방지)"
    )

    args = parser.parse_args()

    # 전략 config 로드
    strategy_config_path = f"strategies/{args.strategy}/config.yaml"
    if os.path.exists(strategy_config_path):
        with open(strategy_config_path, "r", encoding="utf-8") as f:
            strategy_config = yaml.safe_load(f)
        strat_cfg = strategy_config.get("strategy", {})
        params_cfg = strategy_config.get("params", {})
        symbol = args.symbol or strat_cfg.get("symbol", "BTCUSDT")
        timeframe = args.timeframe or strat_cfg.get("timeframe", "1h")
        # CLI 미지정 시 config에서 배리어/max_holding 로드
        if args.upper_barrier == 1.5 and "upper_barrier_multiplier" in params_cfg:
            args.upper_barrier = params_cfg["upper_barrier_multiplier"]
        if args.lower_barrier == 1.5 and "lower_barrier_multiplier" in params_cfg:
            args.lower_barrier = params_cfg["lower_barrier_multiplier"]
        if args.max_holding == 24 and "max_holding_period" in params_cfg:
            args.max_holding = params_cfg["max_holding_period"]
        logger.info(f"전략 config 로드: {strategy_config_path}")
    else:
        symbol = args.symbol or "BTCUSDT"
        timeframe = args.timeframe or "1h"
        params_cfg = {}
        logger.warning(f"전략 config 없음: {strategy_config_path} — CLI 인자 사용")

    save_dir = f"strategies/{args.strategy}/models"

    # 데이터 로드
    data_path = f"data/processed/{symbol}_{timeframe}_features.parquet"
    if not os.path.exists(data_path):
        logger.error(f"데이터 파일 없음: {data_path}")
        logger.info("먼저 데이터를 수집하고 processor를 실행하세요.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info(f"데이터 로드: {symbol} {timeframe} — {len(df)}행")

    # 1. 피처 계산
    logger.info("피처 계산 시작...")
    feature_engine = FeatureEngine(config={"symbol": symbol})
    df = feature_engine.compute_all_features(df)

    if args.use_all_features:
        feature_names = feature_engine.get_feature_names()
        logger.info(f"전체 피처 사용: {len(feature_names)}개")
    else:
        feature_names = feature_engine.get_selected_features()
        logger.info(f"선별 피처 사용: {len(feature_names)}개")

    # 2. 라벨 생성 (전략별 라벨러 선택)
    labeler_type = params_cfg.get("labeler_type", "triple_barrier") if 'params_cfg' in locals() else "triple_barrier"
    if labeler_type == "mean_reversion":
        from strategies.btc_1h_mean_reversion.labeler import MeanReversionLabeler
        labeler = MeanReversionLabeler(
            oversold_bb_threshold=params_cfg.get("oversold_bb_threshold", 0.2),
            oversold_rsi_threshold=params_cfg.get("oversold_rsi_threshold", 30.0),
            profit_atr_mult=args.upper_barrier,
            loss_atr_mult=args.lower_barrier,
            max_holding_period=args.max_holding,
            oversold_mode=params_cfg.get("oversold_mode", "or"),
        )
        logger.info("MeanReversion 라벨링 시작...")
    else:
        labeler = TripleBarrierLabeler(
            upper_multiplier=args.upper_barrier,
            lower_multiplier=args.lower_barrier,
            max_holding_period=args.max_holding,
        )
        logger.info("Triple Barrier 라벨링 시작...")
    df["label"] = labeler.generate_labels(df)

    # 2.5. Gap 구간 제외 (라벨을 NaN으로 설정 → 아래 NaN 제거에서 자동 제외)
    exclude_gaps = args.exclude_gaps and not args.no_exclude_gaps
    if exclude_gaps and "is_gap_filled" in df.columns:
        gap_mask = df["is_gap_filled"].fillna(False).astype(bool)
        gap_count = gap_mask.sum()

        if gap_count > 0:
            # gap 행 자체 + gap 이후 buffer 봉 (라벨이 gap 가격에 의존하므로)
            exclude_mask = gap_mask.copy()
            gap_indices = df.index[gap_mask]
            for idx in gap_indices:
                buffer_end = min(idx + args.gap_buffer_bars, len(df) - 1)
                exclude_mask.iloc[idx:buffer_end + 1] = True

            excluded_total = exclude_mask.sum()
            df.loc[exclude_mask, "label"] = np.nan
            logger.info(
                f"Gap 제외: gap {gap_count}행 + buffer → 총 {excluded_total}행 제외"
            )

    # NaN 제거 (피처 + 라벨)
    valid_cols = feature_names + ["label", "timestamp"]
    valid_cols = [c for c in valid_cols if c in df.columns]
    before_len = len(df)
    df = df.dropna(subset=feature_names + ["label"])
    df = df.reset_index(drop=True)
    logger.info(f"NaN 제거: {before_len} → {len(df)}행")

    # 라벨 분포 출력
    label_counts = df["label"].value_counts().sort_index()
    logger.info(f"라벨 분포: {dict(label_counts)}")

    # 2.5. 상관관계 기반 피처 제거
    if args.corr_threshold > 0:
        before_count = len(feature_names)
        feature_names = FeatureEngine.remove_correlated_features(
            df, feature_names, threshold=args.corr_threshold
        )
        removed = before_count - len(feature_names)
        if removed > 0:
            logger.info(
                f"상관관계 제거 (threshold={args.corr_threshold}): "
                f"{before_count} → {len(feature_names)}개 ({removed}개 제거)"
            )

    # 3. Walk-Forward 학습
    # --load-params: 기존 best_params.json 로드 → Optuna 스킵
    loaded_params = None
    if args.load_params:
        params_path = os.path.join(save_dir, "best_params.json")
        if os.path.exists(params_path):
            import json as _json
            with open(params_path, "r", encoding="utf-8") as f:
                loaded_params = _json.load(f)
            logger.info(f"기존 파라미터 로드: {params_path}")
        else:
            logger.warning(f"best_params.json 없음: {params_path} — Optuna 실행")

    n_trials = 0 if (args.no_optuna or loaded_params) else args.optuna_trials
    trainer = WalkForwardTrainer(
        min_train_months=args.min_train_months,
        val_months=args.val_months,
        embargo_bars=args.embargo_bars,
        n_optuna_trials=n_trials,
        use_sliding_window=args.sliding_window,
        sliding_window_months=args.sliding_window_months,
    )

    logger.info("Walk-Forward 학습 시작...")
    result = trainer.run(df, feature_names, override_params=loaded_params)

    # 4. 과적합 체크 (trainer.run()에서 이미 gap 계산 및 필터링 완료, 여기서는 로그만 보충)
    overfit_count = 0
    for fm in result["folds_metrics"]:
        ov = ModelEvaluator.check_overfitting(fm["train_f1_macro"], fm["val_f1_macro"])
        fm["overfit_check"] = ov
        if ov["is_overfit"]:
            overfit_count += 1
            logger.warning(
                f"Fold {fm['fold']}: 과적합 의심 "
                f"(gap={ov['gap']:.4f}, train={fm['train_f1_macro']:.4f}, "
                f"val={fm['val_f1_macro']:.4f})"
            )

    total_folds = len(result["folds_metrics"])
    if overfit_count == total_folds:
        logger.warning(f"전체 {total_folds}개 fold가 과적합. 피처/하이퍼파라미터 재검토 필요.")

    # 5. 모델 저장
    os.makedirs(save_dir, exist_ok=True)
    model_path = trainer.save_model(
        model=result["model"],
        params=result["params"],
        feature_names=result["feature_names"],
        folds_metrics=result["folds_metrics"],
        feature_importance=result["feature_importance"],
        save_dir=save_dir,
        best_fold_idx=result["best_fold_idx"],
        best_val_f1=result["best_val_f1"],
        fold_models=result.get("fold_models"),
    )

    # 6. 결과 출력
    print("\n" + "=" * 60)
    print("LightGBM 학습 완료")
    print("=" * 60)
    print(f"전략: {args.strategy} | 심볼: {symbol} | 타임프레임: {timeframe}")
    print(f"피처 수: {len(feature_names)}")
    print(f"Fold 수: {len(result['folds_metrics'])}")
    selected_fm = result["folds_metrics"][result["best_fold_idx"]]
    print(
        f"선택 Fold: {result['best_fold_idx']} "
        f"(Val F1: {result['best_val_f1']:.4f}, "
        f"gap: {selected_fm['overfit_gap']:.4f})"
    )

    avg_val_f1 = sum(fm["val_f1_macro"] for fm in result["folds_metrics"]) / len(
        result["folds_metrics"]
    )
    print(f"평균 Val F1(macro): {avg_val_f1:.4f}")
    print(f"과적합 fold: {overfit_count}/{total_folds}")

    print(f"\n모델 저장: {model_path}")
    print(f"피처 이름: {save_dir}/feature_names.json")
    print(f"하이퍼파라미터: {save_dir}/best_params.json")
    print(f"학습 메타: {save_dir}/training_meta.json")

    # 피처 중요도 상위 10개
    importance = result["feature_importance"]
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n피처 중요도 Top 10:")
    for name, imp in top_features:
        print(f"  {name}: {imp}")


if __name__ == "__main__":
    main()
