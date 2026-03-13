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


def simulate_period_v2(
    df_period, signals_period, confidences_period,
    sl_pcts_period, tp_pcts_period,
    max_hold=24, base_position_pct=0.20,
    fee_per_side=0.00055, slippage_per_side=0.0,
):
    """양방향 + 동적 SL/TP + 동적 포지셔닝 시뮬레이션.

    Args:
        df_period: OHLCV 데이터프레임.
        signals_period: 1(롱) / -1(숏) / 0(대기) 시리즈.
        confidences_period: 0.0~1.0 시리즈 (포지셔닝 스케일).
        sl_pcts_period: 각 봉의 SL 비율 시리즈 (동적).
        tp_pcts_period: 각 봉의 TP 비율 시리즈 (동적).
        max_hold: 최대 보유 기간 (봉 수).
        base_position_pct: 기본 포지션 비율.
        fee_per_side: 편도 수수료.
        slippage_per_side: 편도 슬리피지.

    Returns:
        dict: 거래 통계 (롱/숏 분리 메트릭 포함).
    """
    close = df_period["close"].values
    high = df_period["high"].values
    low = df_period["low"].values
    sigs = signals_period.values
    confs = confidences_period.values
    sls = sl_pcts_period.values
    tps = tp_pcts_period.values
    n = len(close)

    trades = []
    i = 0

    while i < n:
        if sigs[i] == 0:
            i += 1
            continue

        direction = int(sigs[i])  # 1 or -1
        entry_price = close[i]
        sl_pct = sls[i]
        tp_pct = tps[i]

        # 포지션 크기 (confidence 비례)
        scale = 0.25 + 0.75 * confs[i]
        position_pct = base_position_pct * scale

        # SL/TP 가격 계산
        if direction == 1:  # 롱
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:  # 숏
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        exit_bar = None
        exit_return = 0
        exit_type = "timeout"

        for j in range(i + 1, min(i + 1 + max_hold, n)):
            if direction == 1:  # 롱
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price
            else:  # 숏
                hit_tp = low[j] <= tp_price
                hit_sl = high[j] >= sl_price

            if hit_tp and hit_sl:
                exit_bar = j
                exit_return = -sl_pct  # 동시 터치: 보수적으로 SL
                exit_type = "sl"
                break
            elif hit_tp:
                exit_bar = j
                exit_return = tp_pct
                exit_type = "tp"
                break
            elif hit_sl:
                exit_bar = j
                exit_return = -sl_pct
                exit_type = "sl"
                break

        if exit_bar is None:
            exit_bar = min(i + max_hold, n - 1)
            if direction == 1:
                exit_return = (close[exit_bar] - entry_price) / entry_price
            else:
                exit_return = (entry_price - close[exit_bar]) / entry_price
            exit_type = "timeout"

        trades.append({
            "direction": direction,
            "entry_bar": i,
            "exit_bar": exit_bar,
            "return": exit_return,
            "position_pct": position_pct,
            "type": exit_type,
        })

        i = exit_bar + 1

    if not trades:
        return {
            "trades": 0, "pf": 0, "total_return": 0, "win_rate": 0,
            "mdd": 0, "tp_count": 0, "sl_count": 0, "timeout_count": 0,
            "long_trades": 0, "short_trades": 0,
            "long_win_rate": 0, "short_win_rate": 0,
            "long_pf": 0, "short_pf": 0,
            "avg_position_pct": 0,
        }

    # 수익률 계산 (동적 포지션 크기)
    cumulative = 1.0
    equity_curve = [1.0]
    for t in trades:
        total_cost = 2 * (fee_per_side + slippage_per_side)
        pnl = t["position_pct"] * (t["return"] - total_cost)
        cumulative *= (1 + pnl)
        equity_curve.append(cumulative)

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = dd.min() * 100

    returns = np.array([t["return"] for t in trades])
    wins = returns > 0
    gross_profit = returns[wins].sum() if wins.any() else 0
    gross_loss = abs(returns[~wins].sum()) if (~wins).any() else 0.001
    pf = gross_profit / gross_loss

    types = [t["type"] for t in trades]

    # 롱/숏 분리 메트릭
    long_trades = [t for t in trades if t["direction"] == 1]
    short_trades = [t for t in trades if t["direction"] == -1]

    def _side_metrics(side_trades):
        if not side_trades:
            return 0, 0.0, 0.0
        r = np.array([t["return"] for t in side_trades])
        w = r > 0
        wr = w.mean() * 100 if len(r) > 0 else 0.0
        gp = r[w].sum() if w.any() else 0
        gl = abs(r[~w].sum()) if (~w).any() else 0.001
        return len(side_trades), wr, gp / gl

    ln, lwr, lpf = _side_metrics(long_trades)
    sn, swr, spf = _side_metrics(short_trades)

    avg_pos = np.mean([t["position_pct"] for t in trades])

    return {
        "trades": len(trades),
        "win_rate": wins.mean() * 100,
        "pf": pf,
        "total_return": (cumulative - 1) * 100,
        "mdd": mdd,
        "tp_count": types.count("tp"),
        "sl_count": types.count("sl"),
        "timeout_count": types.count("timeout"),
        "long_trades": ln,
        "short_trades": sn,
        "long_win_rate": lwr,
        "short_win_rate": swr,
        "long_pf": lpf,
        "short_pf": spf,
        "avg_position_pct": avg_pos,
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

    # 모드 판별
    mode = params.get("mode", "classifier")

    signals = pd.Series(0, index=df.index, dtype=int)

    if models:
        raw_preds = [m.predict(X[valid_mask]) for m in models]
        proba = np.mean(raw_preds, axis=0)
    else:
        proba = model.predict(X[valid_mask])

    if mode == "regressor":
        # === 회귀 모드: 예측값 기반 양방향 시그널 ===
        min_pred_threshold = params.get("min_pred_threshold", 0.005)
        max_position_scale = params.get("max_position_scale", 2.0)
        sl_atr_mult = params.get("sl_atr_mult", 2.0)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        min_sl_pct = params.get("min_sl_pct", 0.01)
        max_sl_pct = params.get("max_sl_pct", 0.05)
        min_tp_pct = params.get("min_tp_pct", 0.01)
        max_tp_pct = params.get("max_tp_pct", 0.08)

        # 예측값 중심화: expanding mean을 빼서 양방향 시그널 가능하게 함
        demean_window = params.get("pred_demean_window", 720)
        pred_series = pd.Series(proba)
        expanding_mean = pred_series.rolling(window=demean_window, min_periods=100).mean()
        proba_centered = (pred_series - expanding_mean).values.copy()
        # rolling mean이 NaN인 초기 구간은 전체 expanding mean 사용
        nan_mask = np.isnan(proba_centered)
        if nan_mask.any():
            global_mean = pred_series.iloc[:demean_window].mean()
            proba_centered[nan_mask] = proba[nan_mask] - global_mean
        print(f"예측값 중심화: demean_window={demean_window}, "
              f"centered mean={proba_centered.mean():.6f}, std={proba_centered.std():.6f}")
        proba = proba_centered

        abs_preds = np.abs(proba)

        # 펀딩비 적응형 threshold (수익률 단위)
        funding_filter = params.get("funding_filter", {})
        if funding_filter.get("enabled", False) and "funding_rate_zscore" in df_feat.columns:
            fr_zscore = df_feat.loc[valid_mask, "funding_rate_zscore"].values
            zscore_thresholds = funding_filter.get("zscore_thresholds", [])
            adaptive_thr = np.full(len(proba), 999.0)
            for rule in sorted(zscore_thresholds, key=lambda x: x["zscore_below"], reverse=True):
                mask = fr_zscore < rule["zscore_below"]
                adaptive_thr[mask] = rule["confidence"]
            adaptive_thr[np.isnan(fr_zscore)] = min_pred_threshold
            print(f"펀딩비 적응형 threshold 적용: {zscore_thresholds}")
        else:
            adaptive_thr = np.full(len(proba), min_pred_threshold)

        # OI 필터
        oi_filter = params.get("oi_filter", {})
        if oi_filter.get("enabled", False) and "oi_zscore" in df_feat.columns:
            oi_block = oi_filter.get("block_zscore")
            if oi_block is not None:
                oi_z = df_feat.loc[valid_mask, "oi_zscore"].values
                block_mask = (oi_z >= oi_block) & ~np.isnan(oi_z)
                adaptive_thr[block_mask] = 999.0
                print(f"OI 필터 적용: block_zscore >= {oi_block}")

        # 양방향 시그널 생성
        long_mask = (proba > 0) & (abs_preds >= adaptive_thr)
        short_mask = (proba < 0) & (abs_preds >= adaptive_thr)
        signal_values = np.zeros(len(proba), dtype=int)
        signal_values[long_mask] = 1
        signal_values[short_mask] = -1
        signals.loc[valid_mask] = signal_values

        # confidence 시리즈
        confidences = pd.Series(0.0, index=df.index, dtype=float)
        conf_values = np.minimum(abs_preds / (min_pred_threshold * max_position_scale), 1.0)
        conf_values[signal_values == 0] = 0.0
        confidences.loc[valid_mask] = conf_values

        # 동적 SL/TP 시리즈
        fallback_sl = risk.get("stop_loss_pct", 0.021)
        fallback_tp = risk.get("take_profit_pct", 0.021)
        if "atr_14" in df_feat.columns:
            atr_pct = (df_feat["atr_14"] / df_feat["close"]).values
            sl_pcts = pd.Series(
                np.clip(atr_pct * sl_atr_mult, min_sl_pct, max_sl_pct),
                index=df.index,
            )
            tp_pcts = pd.Series(
                np.clip(atr_pct * tp_atr_mult, min_tp_pct, max_tp_pct),
                index=df.index,
            )
            # NaN 행에 fallback
            sl_pcts = sl_pcts.fillna(fallback_sl)
            tp_pcts = tp_pcts.fillna(fallback_tp)
        else:
            sl_pcts = pd.Series(fallback_sl, index=df.index)
            tp_pcts = pd.Series(fallback_tp, index=df.index)

    else:
        # === 분류 모드 (기존 로직 — 변경 없음) ===
        proba_series = pd.Series(np.nan, index=df.index)
        proba_series.loc[valid_mask] = proba

        # 펀딩비 적응형 threshold
        funding_filter = params.get("funding_filter", {})
        if funding_filter.get("enabled", False) and "funding_rate_zscore" in df_feat.columns:
            fr_zscore = df_feat["funding_rate_zscore"].values
            zscore_thresholds = funding_filter.get("zscore_thresholds", [])
            adaptive_thr = np.full(len(df), 999.0)
            for rule in sorted(zscore_thresholds, key=lambda x: x["zscore_below"], reverse=True):
                mask = fr_zscore < rule["zscore_below"]
                adaptive_thr[mask] = rule["confidence"]
            adaptive_thr[np.isnan(fr_zscore)] = CONFIDENCE_THRESHOLD
            print(f"펀딩비 적응형 threshold 적용: {zscore_thresholds}")
        else:
            adaptive_thr = np.full(len(df), CONFIDENCE_THRESHOLD)

        # OI 필터
        oi_filter = params.get("oi_filter", {})
        if oi_filter.get("enabled", False) and "oi_zscore" in df_feat.columns:
            oi_block = oi_filter.get("block_zscore")
            if oi_block is not None:
                oi_z = df_feat["oi_zscore"].values
                block_mask = (oi_z >= oi_block) & ~np.isnan(oi_z)
                adaptive_thr[block_mask] = 999.0
                print(f"OI 필터 적용: block_zscore >= {oi_block}")

        signals.loc[valid_mask] = np.where(proba >= adaptive_thr[valid_mask], 1, 0)

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
    print(f"OOS 검증 (mode={mode})")
    print("=" * 80)

    if mode == "regressor":
        print(f"min_pred_threshold: {params.get('min_pred_threshold', 0.005)}")
        print(f"동적 SL/TP: sl_atr_mult={params.get('sl_atr_mult', 2.0)}, tp_atr_mult={params.get('tp_atr_mult', 3.0)}")
    else:
        print(f"confidence_threshold: {CONFIDENCE_THRESHOLD}")
        print(f"SL: {SL_PCT*100}% / TP: {TP_PCT*100}%")
    print(f"Max Hold: {MAX_HOLD}")
    print(f"Post-Val: {val_end_ts} ~ {oos_boundary}")
    print()

    # 시그널 분포
    if mode == "regressor":
        print(f"전체 시그널: 롱={int((signals==1).sum())}, 숏={int((signals==-1).sum())}, 대기={int((signals==0).sum())}")
    else:
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

        if mode == "regressor":
            print(f"{'구간':<30} {'거래':>5} {'롱':>4} {'숏':>4} {'승률':>6} {'PF':>6} {'수익률':>8} {'MDD':>7}")
        else:
            print(f"{'구간':<30} {'거래':>5} {'승률':>6} {'PF':>6} {'수익률':>8} {'MDD':>7}")
        print("-" * 80)

        results = {}
        for name, (start, end) in periods.items():
            mask = (ts >= start) & (ts < end)
            idx = df.index[mask]
            if len(idx) == 0:
                print(f"{name:<30} 데이터 없음")
                continue

            if mode == "regressor":
                result = simulate_period_v2(
                    df.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                    signals.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                    confidences.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                    sl_pcts.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                    tp_pcts.iloc[idx[0]:idx[-1]+1].reset_index(drop=True),
                    MAX_HOLD, POSITION_PCT, FEE_PER_SIDE,
                    slippage_per_side=slippage,
                )
            else:
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
            if mode == "regressor":
                print(f"{name:<30} {r['trades']:>5} {r['long_trades']:>4} {r['short_trades']:>4} "
                      f"{r['win_rate']:>5.1f}% {r['pf']:>6.2f} "
                      f"{r['total_return']:>+7.2f}% {r['mdd']:>6.2f}%")
            else:
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
        return all_scenario_results, {
            "passed": False, "conservative": {"pv_pf": 0, "pv_return": 0, "pv_trades": 0},
            "strict_oos": {},
        }

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

    # 회귀 모드: 롱/숏 분리 정보
    if mode == "regressor" and pv_result.get("long_trades", 0) + pv_result.get("short_trades", 0) > 0:
        print(f"\n  --- PV 롱/숏 분리 (보수적) ---")
        print(f"  롱: {pv_result.get('long_trades', 0)}건, 승률 {pv_result.get('long_win_rate', 0):.1f}%, PF {pv_result.get('long_pf', 0):.2f}")
        print(f"  숏: {pv_result.get('short_trades', 0)}건, 승률 {pv_result.get('short_win_rate', 0):.1f}%, PF {pv_result.get('short_pf', 0):.2f}")

    all_passed = all(checks.values())
    print(f"\n최종 결과: {'SUCCESS' if all_passed else 'FAIL'}")
    print("=" * 80)

    # 구조화된 결과 (프로그래밍 API용)
    structured = {
        "passed": all_passed,
        "conservative": {
            "is_pf": is_pf,
            "pv_pf": pv_pf,
            "pv_return": pv_return,
            "pv_trades": pv_trades,
            "pf_drop": pf_drop,
        },
        "strict_oos": {},
    }

    strict_result = conservative.get("Strict OOS (2026-01-19~)", {})
    if strict_result:
        structured["strict_oos"] = {
            "pf": strict_result.get("pf", 0),
            "trades": strict_result.get("trades", 0),
            "total_return": strict_result.get("total_return", 0),
        }

    return all_scenario_results, structured


if __name__ == "__main__":
    import json as _json

    parser = argparse.ArgumentParser(description="OOS 검증")
    parser.add_argument("--strategy", type=str, default="btc_1h_momentum",
                        help="전략 이름 (기본: btc_1h_momentum)")
    parser.add_argument("--json", action="store_true",
                        help="JSON 형식으로 결과 출력")
    args = parser.parse_args()
    result = run_oos_validation(strategy_name=args.strategy)

    if args.json and isinstance(result, tuple) and len(result) == 2:
        _, structured = result
        print("\n__JSON_RESULT__")
        print(_json.dumps(structured, ensure_ascii=False))
    elif isinstance(result, tuple):
        pass  # 일반 출력 이미 완료
