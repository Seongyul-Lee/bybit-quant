"""시뮬레이터 괴리 분석 v2: vectorbt 슬리피지(0.2%) 반영.

vectorbt backtest.py 설정:
  - fees=0.00055 (편도 수수료)
  - slippage=0.002 (0.2% 슬리피지)
  - size=max_position_pct (5%)
  - 종가 기반 SL/TP
  - exits=False → SL/TP에만 의존하지만 max_hold 없음!

핵심 차이점 발견:
  1. vectorbt에는 max_holding_period가 없음 → timeout 없이 SL/TP까지 무한 홀드
  2. slippage 0.2% 적용
  3. 수수료 구조 차이
"""

import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml

from strategies.lgbm_classifier.features import FeatureEngine


def simulate(df_period, signals_period, sl_pct, tp_pct, max_hold,
             position_pct=0.05, fee_per_side=0.00055, slippage=0.0,
             use_highlow=True):
    """통합 시뮬레이터.

    Args:
        use_highlow: True=봉내 high/low 체크, False=종가만 체크
        slippage: 슬리피지 비율 (편도, 진입+청산 각각 적용)
        max_hold: 0이면 무한 홀드 (SL/TP만으로 청산)
    """
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

        entry_price = close[i] * (1 + slippage)  # 슬리피지: 더 비싸게 진입
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)

        exit_bar = None
        exit_return = 0
        exit_type = "timeout"

        search_end = n if max_hold == 0 else min(i + 1 + max_hold, n)

        for j in range(i + 1, search_end):
            if use_highlow:
                if high[j] >= tp_price and low[j] <= sl_price:
                    exit_bar = j
                    exit_price = sl_price * (1 - slippage)
                    exit_return = (exit_price - entry_price) / entry_price
                    exit_type = "sl"
                    break
                elif high[j] >= tp_price:
                    exit_bar = j
                    exit_price = tp_price * (1 - slippage)
                    exit_return = (exit_price - entry_price) / entry_price
                    exit_type = "tp"
                    break
                elif low[j] <= sl_price:
                    exit_bar = j
                    exit_price = sl_price * (1 - slippage)
                    exit_return = (exit_price - entry_price) / entry_price
                    exit_type = "sl"
                    break
            else:
                if close[j] >= tp_price:
                    exit_bar = j
                    exit_price = close[j] * (1 - slippage)
                    exit_return = (exit_price - entry_price) / entry_price
                    exit_type = "tp"
                    break
                elif close[j] <= sl_price:
                    exit_bar = j
                    exit_price = close[j] * (1 - slippage)
                    exit_return = (exit_price - entry_price) / entry_price
                    exit_type = "sl"
                    break

        if exit_bar is None:
            if max_hold == 0:
                # 무한 홀드인데 SL/TP에 안 걸림 → 마지막 봉에서 청산
                exit_bar = n - 1
            else:
                exit_bar = min(i + max_hold, n - 1)
            exit_price = close[exit_bar] * (1 - slippage)
            exit_return = (exit_price - entry_price) / entry_price
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
                "avg_win": 0, "avg_loss": 0}

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
    avg_hold = np.mean([t["holding"] for t in trades])

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
        "avg_hold": avg_hold,
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

    funding_filter = params.get("funding_filter", {})
    if funding_filter.get("enabled", False) and "funding_rate_zscore" in df_feat.columns:
        fr_zscore = df_feat["funding_rate_zscore"].values
        zscore_thresholds = funding_filter.get("zscore_thresholds", [])
        adaptive_thr = np.full(len(df), 999.0)
        for rule in sorted(zscore_thresholds, key=lambda x: x["zscore_below"], reverse=True):
            adaptive_thr[fr_zscore < rule["zscore_below"]] = rule["confidence"]
        adaptive_thr[np.isnan(fr_zscore)] = CONFIDENCE_THRESHOLD
        signals.loc[valid_mask] = np.where(proba >= adaptive_thr[valid_mask], 1, 0)
    else:
        signals.loc[valid_mask] = np.where(proba >= CONFIDENCE_THRESHOLD, 1, 0)

    print(f"전체 신호: 매수={int((signals==1).sum())}")
    print(f"SL: {SL_PCT*100}% / TP: {TP_PCT*100}% / Max Hold: {MAX_HOLD}")

    # 시나리오 정의
    scenarios = [
        ("A. OOS방식 (high/low, 수수료만, maxhold=16)",
         dict(use_highlow=True, slippage=0.0, fee_per_side=FEE_PER_SIDE, max_hold=MAX_HOLD)),
        ("B. 종가기준 + 수수료만 + maxhold=16",
         dict(use_highlow=False, slippage=0.0, fee_per_side=FEE_PER_SIDE, max_hold=MAX_HOLD)),
        ("C. 종가기준 + 슬리피지0.2% + maxhold=16",
         dict(use_highlow=False, slippage=0.002, fee_per_side=FEE_PER_SIDE, max_hold=MAX_HOLD)),
        ("D. 종가기준 + 슬리피지0.2% + maxhold=무한 (vectorbt)",
         dict(use_highlow=False, slippage=0.002, fee_per_side=FEE_PER_SIDE, max_hold=0)),
        ("E. high/low + 슬리피지0.2% + maxhold=16 (보수적 실거래)",
         dict(use_highlow=True, slippage=0.002, fee_per_side=FEE_PER_SIDE, max_hold=MAX_HOLD)),
    ]

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
        "전체": (ts.iloc[0], ts.iloc[-1]),
        "IS": (ts.iloc[0], val_end_ts),
        "PV": (val_end_ts, oos_boundary),
        "OOS": (oos_boundary, ts.iloc[-1]),
    }

    print("\n" + "=" * 100)
    print("시뮬레이터 시나리오 비교")
    print("=" * 100)

    for scenario_name, kwargs in scenarios:
        print(f"\n--- {scenario_name} ---")
        print(f"{'구간':<8} {'거래':>5} {'TP':>4} {'SL':>4} {'TO':>4} "
              f"{'승률':>6} {'PF':>6} {'수익률':>8} {'MDD':>7} "
              f"{'평균승':>7} {'평균패':>7} {'평균보유':>6}")
        print("-" * 100)

        for name, (start, end) in periods.items():
            mask = (ts >= start) & (ts < end)
            idx = df.index[mask]
            if len(idx) == 0:
                print(f"{name:<8} 데이터 없음")
                continue

            result = simulate(
                df.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                signals.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                SL_PCT, TP_PCT, position_pct=POSITION_PCT,
                **kwargs,
            )

            if result["trades"] == 0:
                print(f"{name:<8} 거래 없음")
                continue

            r = result
            print(f"{name:<8} {r['trades']:>5} {r['tp_count']:>4} {r['sl_count']:>4} "
                  f"{r['timeout_count']:>4} {r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
                  f"{r['total_return']:>+7.2f}% {r['mdd']:>6.2f}% "
                  f"{r['avg_win']:>+6.2f}% {r['avg_loss']:>+6.2f}% {r['avg_hold']:>5.1f}h")

    print("\n" + "=" * 100)
    print("시나리오 해석:")
    print("  A: 현재 OOS 검증 방식 (가장 낙관적)")
    print("  B: A에서 SL/TP 체크를 종가로 변경 → high/low vs 종가 효과 분리")
    print("  C: B에 슬리피지 추가 → 거래 비용 효과 분리")
    print("  D: C에서 max_hold 제거 → vectorbt와 동일 조건 (괴리 원인 확인)")
    print("  E: A에 슬리피지 추가 → 실거래 최선 추정치")
    print("=" * 100)


if __name__ == "__main__":
    main()
