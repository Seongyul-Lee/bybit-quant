"""LightGBM 모델 학습 CLI 스크립트.

사용법:
    python train_lgbm.py --symbol BTCUSDT --timeframe 1h
    python train_lgbm.py --symbol BTCUSDT --timeframe 1h --optuna-trials 100
    python train_lgbm.py --symbol BTCUSDT --timeframe 1h --no-optuna
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from src.utils.logger import setup_logger
from strategies.lgbm_classifier.features import FeatureEngine
from strategies.lgbm_classifier.labeler import TripleBarrierLabeler
from strategies.lgbm_classifier.trainer import WalkForwardTrainer
from strategies.lgbm_classifier.evaluator import ModelEvaluator

logger = setup_logger("train_lgbm")


def main() -> None:
    """학습 CLI 진입점."""
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(description="LightGBM 모델 학습")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="심볼 (기본: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="타임프레임 (기본: 1h)")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Optuna 시행 수 (기본: 50)")
    parser.add_argument("--no-optuna", action="store_true", help="Optuna 튜닝 비활성화")
    parser.add_argument("--min-train-months", type=int, default=6, help="최소 학습 기간 (월)")
    parser.add_argument("--val-months", type=int, default=1, help="검증 기간 (월)")
    parser.add_argument("--embargo-bars", type=int, default=24, help="Embargo 봉 수")
    parser.add_argument(
        "--upper-barrier", type=float, default=2.0, help="상단 배리어 ATR 배수"
    )
    parser.add_argument(
        "--lower-barrier", type=float, default=1.0, help="하단 배리어 ATR 배수"
    )
    parser.add_argument(
        "--max-holding", type=int, default=24, help="최대 보유 기간 (봉 수)"
    )

    args = parser.parse_args()

    # 데이터 로드
    data_path = f"data/processed/{args.symbol}_{args.timeframe}_features.parquet"
    if not os.path.exists(data_path):
        logger.error(f"데이터 파일 없음: {data_path}")
        logger.info("먼저 데이터를 수집하고 processor를 실행하세요.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    logger.info(f"데이터 로드: {args.symbol} {args.timeframe} — {len(df)}행")

    # 1. 피처 계산
    logger.info("피처 계산 시작...")
    feature_engine = FeatureEngine(config={})
    df = feature_engine.compute_all_features(df)
    feature_names = feature_engine.get_feature_names()
    logger.info(f"피처 수: {len(feature_names)}")

    # 2. 라벨 생성
    logger.info("Triple Barrier 라벨링 시작...")
    labeler = TripleBarrierLabeler(
        upper_multiplier=args.upper_barrier,
        lower_multiplier=args.lower_barrier,
        max_holding_period=args.max_holding,
    )
    df["label"] = labeler.generate_labels(df)

    # NaN 제거 (피처 + 라벨)
    valid_cols = feature_names + ["label", "timestamp"]
    valid_cols = [c for c in valid_cols if c in df.columns]
    before_len = len(df)
    df = df.dropna(subset=feature_names + ["label"])
    logger.info(f"NaN 제거: {before_len} → {len(df)}행")

    # 라벨 분포 출력
    label_counts = df["label"].value_counts().sort_index()
    logger.info(f"라벨 분포: {dict(label_counts)}")

    # 3. Walk-Forward 학습
    n_trials = 0 if args.no_optuna else args.optuna_trials
    trainer = WalkForwardTrainer(
        min_train_months=args.min_train_months,
        val_months=args.val_months,
        embargo_bars=args.embargo_bars,
        n_optuna_trials=n_trials,
    )

    logger.info("Walk-Forward 학습 시작...")
    result = trainer.run(df, feature_names)

    # 4. 과적합 체크
    for fm in result["folds_metrics"]:
        ov = ModelEvaluator.check_overfitting(fm["train_f1_macro"], fm["val_f1_macro"])
        fm["overfit_check"] = ov
        if ov["is_overfit"]:
            logger.warning(
                f"Fold {fm['fold']}: 과적합 의심 "
                f"(gap={ov['gap']:.4f}, train={fm['train_f1_macro']:.4f}, "
                f"val={fm['val_f1_macro']:.4f})"
            )

    # 5. 모델 저장
    save_dir = "strategies/lgbm_classifier/models"
    model_path = trainer.save_model(
        model=result["model"],
        params=result["params"],
        feature_names=result["feature_names"],
        folds_metrics=result["folds_metrics"],
        feature_importance=result["feature_importance"],
        save_dir=save_dir,
    )

    # 6. 결과 출력
    print("\n" + "=" * 60)
    print("LightGBM 학습 완료")
    print("=" * 60)
    print(f"심볼: {args.symbol} | 타임프레임: {args.timeframe}")
    print(f"피처 수: {len(feature_names)}")
    print(f"Fold 수: {len(result['folds_metrics'])}")

    avg_val_f1 = sum(fm["val_f1_macro"] for fm in result["folds_metrics"]) / len(
        result["folds_metrics"]
    )
    print(f"평균 Val F1(macro): {avg_val_f1:.4f}")

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
