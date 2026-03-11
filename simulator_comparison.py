"""시뮬레이터 괴리 분석: high/low 기준 vs 종가 기준 SL/TP.

실험 1: 종가 기준 SL/TP (vectorbt 방식)
실험 2: high/low 기준 SL/TP (oos_validation 방식)
동일한 신호, 동일한 전체 기간에서 비교.
"""

import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml

from strategies.lgbm_classifier.features import FeatureEngine


def simulate_highlow(df_period, signals_period, sl_pct, tp_pct, max_hold,
                     position_pct=0.05, fee_per_side=0.00055):
    """OOS 검증 방식: high/low 기준 SL/TP 체크."""
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

    return _summarize(trades, position_pct, fee_per_side)


def simulate_close_only(df_period, signals_period, sl_pct, tp_pct, max_hold,
                        position_pct=0.05, fee_per_side=0.00055):
    """vectorbt 방식: 종가 기준 SL/TP 체크."""
    close = df_period["close"].values
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
            # 종가 기준만 체크
            if close[j] >= tp_price:
                exit_bar = j
                exit_return = (close[j] - entry_price) / entry_price
                exit_type = "tp"
                break
            elif close[j] <= sl_price:
                exit_bar = j
                exit_return = (close[j] - entry_price) / entry_price
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

    return _summarize(trades, position_pct, fee_per_side)


def _summarize(trades, position_pct, fee_per_side):
    if not trades:
        return {"trades": 0, "pf": 0, "total_return": 0, "win_rate": 0,
                "mdd": 0, "tp_count": 0, "sl_count": 0, "timeout_count": 0,
                "avg_win": 0, "avg_loss": 0, "trades_detail": []}

    returns = np.array([t["return"] for t in trades])
    wins = returns > 0

    cumulative = 1.0
    equity_curve = [1.0]
    for r in returns:
        pnl = position_pct * (r - 2 * fee_per_side)
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

    avg_win = returns[wins].mean() * 100 if wins.any() else 0
    avg_loss = returns[~wins].mean() * 100 if (~wins).any() else 0

    return {
        "trades": len(trades),
        "win_rate": wins.mean() * 100,
        "pf": pf,
        "total_return": (cumulative - 1) * 100,
        "mdd": mdd,
        "tp_count": types.count("tp"),
        "sl_count": types.count("sl"),
        "timeout_count": types.count("timeout"),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    with open("strategies/lgbm_classifier/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    params = config.get("params", {})
    risk = config.get("risk", {})

    CONFIDENCE_THRESHOLD = params.get("confidence_threshold", 0.48)
    SL_PCT = risk.get("stop_loss_pct", 0.015)
    TP_PCT = risk.get("take_profit_pct", 0.015)
    MAX_HOLD = params.get("max_holding_period", 16)
    POSITION_PCT = risk.get("max_position_pct", 0.05)
    FEE_PER_SIDE = config.get("execution", {}).get("fee_rate", 0.00055)
    ENSEMBLE_FOLDS = params.get("ensemble_folds", None)
    MODELS_DIR = params.get("models_dir", "strategies/lgbm_classifier/models")

    # 모델 로드
    if ENSEMBLE_FOLDS:
        models = []
        for fold_idx in ENSEMBLE_FOLDS:
            path = os.path.join(MODELS_DIR, f"fold_{fold_idx:02d}.txt")
            models.append(lgb.Booster(model_file=path))
        print(f"앙상블 모델: {len(models)}개 fold ({ENSEMBLE_FOLDS})")
    else:
        model = lgb.Booster(model_file="strategies/lgbm_classifier/models/latest.txt")
        models = None

    with open("strategies/lgbm_classifier/models/feature_names.json") as f:
        feature_names = json.load(f)

    # 데이터 & 신호
    df = pd.read_parquet("data/processed/BTCUSDT_1h_features.parquet")
    engine = FeatureEngine(config={})
    df_feat = engine.compute_all_features(df)

    X = df_feat[feature_names]
    valid_mask = ~X.isna().any(axis=1)

    signals = pd.Series(0, index=df.index, dtype=int)

    if models:
        preds = [m.predict(X[valid_mask]) for m in models]
        proba = np.mean(preds, axis=0)
    else:
        proba = model.predict(X[valid_mask])

    # 적응형 threshold
    funding_filter = params.get("funding_filter", {})
    if funding_filter.get("enabled", False) and "funding_rate_zscore" in df_feat.columns:
        fr_zscore = df_feat["funding_rate_zscore"].values
        zscore_thresholds = funding_filter.get("zscore_thresholds", [])
        adaptive_thr = np.full(len(df), 999.0)
        for rule in sorted(zscore_thresholds, key=lambda x: x["zscore_below"], reverse=True):
            mask = fr_zscore < rule["zscore_below"]
            adaptive_thr[mask] = rule["confidence"]
        adaptive_thr[np.isnan(fr_zscore)] = CONFIDENCE_THRESHOLD
        signals.loc[valid_mask] = np.where(proba >= adaptive_thr[valid_mask], 1, 0)
        print(f"펀딩비 적응형 threshold 적용")
    else:
        signals.loc[valid_mask] = np.where(proba >= CONFIDENCE_THRESHOLD, 1, 0)

    print(f"\n전체 신호: 매수={int((signals==1).sum())}, 비매수={int((signals==0).sum())}")
    print(f"SL: {SL_PCT*100}% / TP: {TP_PCT*100}% / Max Hold: {MAX_HOLD}")
    print(f"Position: {POSITION_PCT*100}% / Fee: {FEE_PER_SIDE*100}%")

    # 구간 정의
    ts = pd.to_datetime(df["timestamp"])
    with open("strategies/lgbm_classifier/models/training_meta.json") as f:
        meta = json.load(f)

    if ENSEMBLE_FOLDS:
        latest_fold = max(ENSEMBLE_FOLDS)
        fm = meta["folds_metrics"][latest_fold]
        val_end_str = fm["val_period"].split(" ~ ")[-1].strip()
    else:
        best_fold_idx = meta.get("best_fold_idx", -1)
        fm = meta["folds_metrics"][best_fold_idx]
        val_end_str = fm["val_period"].split(" ~ ")[-1].strip()

    val_end_ts = pd.Timestamp(val_end_str)
    if val_end_ts.tzinfo is None:
        val_end_ts = val_end_ts.tz_localize("UTC")
    oos_boundary = pd.Timestamp("2026-01-19", tz="UTC")

    periods = {
        "전체 기간": (ts.iloc[0], ts.iloc[-1]),
        "In-Sample": (ts.iloc[0], val_end_ts),
        "Post-Validation": (val_end_ts, oos_boundary),
        "Strict OOS": (oos_boundary, ts.iloc[-1]),
    }

    # 두 방식 비교
    print("\n" + "=" * 90)
    print("시뮬레이터 비교: high/low 기준 vs 종가 기준 SL/TP")
    print("=" * 90)

    for sim_name, sim_func in [("HIGH/LOW 기준 (OOS방식)", simulate_highlow),
                                ("종가 기준 (vectorbt방식)", simulate_close_only)]:
        print(f"\n--- {sim_name} ---")
        print(f"{'구간':<20} {'거래':>5} {'TP':>4} {'SL':>4} {'TO':>4} "
              f"{'승률':>6} {'PF':>6} {'수익률':>8} {'MDD':>7} "
              f"{'평균승':>7} {'평균패':>7}")
        print("-" * 90)

        for name, (start, end) in periods.items():
            mask = (ts >= start) & (ts < end)
            idx = df.index[mask]
            if len(idx) == 0:
                print(f"{name:<20} 데이터 없음")
                continue

            result = sim_func(
                df.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                signals.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                SL_PCT, TP_PCT, MAX_HOLD, POSITION_PCT, FEE_PER_SIDE,
            )

            if result["trades"] == 0:
                print(f"{name:<20} 거래 없음")
                continue

            r = result
            print(f"{name:<20} {r['trades']:>5} {r['tp_count']:>4} {r['sl_count']:>4} "
                  f"{r['timeout_count']:>4} {r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
                  f"{r['total_return']:>+7.2f}% {r['mdd']:>6.2f}% "
                  f"{r['avg_win']:>+6.2f}% {r['avg_loss']:>+6.2f}%")

    print("\n" + "=" * 90)
    print("분석 포인트:")
    print("  1. 동일 신호에서 SL/TP 체크 방식만 다름")
    print("  2. high/low: 봉 내 고가/저가로 SL/TP hit → 고정 수익/손실")
    print("  3. 종가: 종가가 SL/TP 가격을 넘어야 청산 → 실제 종가 수익/손실")
    print("  4. 종가 방식은 SL 슬리피지 발생 (종가 > SL 가격 → 손실 > SL%)")
    print("=" * 90)


if __name__ == "__main__":
    main()
