"""자동화된 학습 → OOS 검증 실험 스크립트.

사용법:
    python run_experiment.py --config experiments/run1.json
    또는 직접 파라미터 지정:
    python run_experiment.py --threshold 0.55 --upper-barrier 2.5 --lower-barrier 2.5 \
        --sl 0.02 --tp 0.02 --optuna-trials 100
"""

import argparse
import json
import os
import subprocess
import sys
import time

import yaml


def update_config(threshold, sl_pct, tp_pct, upper_barrier, lower_barrier, max_holding):
    """config.yaml을 업데이트."""
    config_path = "strategies/btc_1h_momentum/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["params"]["confidence_threshold"] = threshold
    config["params"]["upper_barrier_multiplier"] = upper_barrier
    config["params"]["lower_barrier_multiplier"] = lower_barrier
    config["params"]["max_holding_period"] = max_holding
    config["risk"]["stop_loss_pct"] = sl_pct
    config["risk"]["take_profit_pct"] = tp_pct

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def run_training(args_list):
    """train_lgbm.py 실행."""
    cmd = [sys.executable, "train_lgbm.py"] + args_list
    print(f"\n>>> 학습 실행: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=1800)
    return result.returncode


def run_oos():
    """oos_validation.py 실행."""
    cmd = [sys.executable, "oos_validation.py"]
    print(f"\n>>> OOS 검증 실행")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        # LightGBM warnings 등은 무시
        pass
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="실험 자동화")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--upper-barrier", type=float, default=2.5)
    parser.add_argument("--lower-barrier", type=float, default=2.5)
    parser.add_argument("--sl", type=float, default=0.02)
    parser.add_argument("--tp", type=float, default=0.02)
    parser.add_argument("--max-holding", type=int, default=24)
    parser.add_argument("--optuna-trials", type=int, default=100)
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--use-all-features", action="store_true")
    parser.add_argument("--corr-threshold", type=float, default=0.9)
    parser.add_argument("--run-name", type=str, default="run")

    args = parser.parse_args()

    print("=" * 60)
    print(f"실험: {args.run_name}")
    print("=" * 60)
    print(f"Threshold: {args.threshold}")
    print(f"Barrier: {args.upper_barrier}/{args.lower_barrier}")
    print(f"SL/TP: {args.sl*100}%/{args.tp*100}%")
    print(f"Max Hold: {args.max_holding}")
    print(f"Optuna: {'OFF' if args.no_optuna else f'{args.optuna_trials} trials'}")
    print(f"Features: {'all' if args.use_all_features else 'selected'}")

    # 1. config 업데이트
    update_config(args.threshold, args.sl, args.tp,
                  args.upper_barrier, args.lower_barrier, args.max_holding)

    # 2. 학습 실행
    train_args = [
        "--upper-barrier", str(args.upper_barrier),
        "--lower-barrier", str(args.lower_barrier),
        "--max-holding", str(args.max_holding),
        "--corr-threshold", str(args.corr_threshold),
    ]
    if args.no_optuna:
        train_args.append("--no-optuna")
    else:
        train_args.extend(["--optuna-trials", str(args.optuna_trials)])
    if args.use_all_features:
        train_args.append("--use-all-features")

    rc = run_training(train_args)
    if rc != 0:
        print(f"학습 실패 (exit code: {rc})")
        return

    # 3. OOS 검증
    run_oos()


if __name__ == "__main__":
    main()
