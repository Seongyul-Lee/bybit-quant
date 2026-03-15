"""
펀딩비 차익거래 Phase 0: 데이터 수집 + 탐색적 분석.

실행: python -m analysis.funding_arb_eda
  --skip-collection: 데이터 수집 건너뛰기 (이미 최신화된 경우)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.stattools import acf, pacf

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collector import BybitDataCollector

# ── 상수 ──────────────────────────────────────────────

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "bybit"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

SYMBOLS = {
    "BTC": {
        "ccxt": "BTC/USDT:USDT",
        "bybit": "BTCUSDT",
        "dir": "BTCUSDTUSDT",
    },
    "ETH": {
        "ccxt": "ETH/USDT:USDT",
        "bybit": "ETHUSDT",
        "dir": "ETHUSDTUSDT",
    },
}

# 비용 모델
COST_SPOT_TAKER = 0.0010   # 현물 taker 0.10%
COST_PERP_TAKER = 0.00055  # 선물 taker 0.055%
COST_ROUNDTRIP = (COST_SPOT_TAKER + COST_PERP_TAKER) * 2  # 왕복 0.31%

FUNDING_PER_DAY = 3  # 하루 3회 결제


# ══════════════════════════════════════════════════════
# 작업 1: 데이터 수집
# ══════════════════════════════════════════════════════

def update_funding_rates() -> None:
    """기존 펀딩비 데이터를 최신화한다."""
    collector = BybitDataCollector(testnet=False)

    for name, sym in SYMBOLS.items():
        fr_path = DATA_RAW / sym["dir"] / "funding_rate.parquet"
        if fr_path.exists():
            existing = pd.read_parquet(fr_path)
            last_ts = existing["timestamp"].max()
            print(f"[{name}] 기존 펀딩비 {len(existing)}건, 마지막: {last_ts}")
            since = last_ts.isoformat()
        else:
            existing = pd.DataFrame()
            since = "2024-01-01T00:00:00Z"
            print(f"[{name}] 기존 데이터 없음 — 전체 수집")

        new_df = collector.fetch_funding_rate_bulk(sym["ccxt"], since=since)
        if new_df.empty:
            print(f"[{name}] 신규 펀딩비 없음")
            continue

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        print(f"[{name}] 펀딩비 업데이트: {len(existing)} → {len(combined)}건")
        combined.to_parquet(fr_path, index=False, compression="snappy")


def fetch_long_short_ratio(symbol: str = "BTCUSDT", period: str = "4h",
                           limit: int = 500) -> pd.DataFrame:
    """Bybit 롱/숏 비율 히스토리 수집 (공개 API)."""
    url = "https://api.bybit.com/v5/market/account-ratio"
    all_data: list[dict] = []
    cursor = None

    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            "period": period,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()

        if data["retCode"] != 0:
            print(f"  API 오류: {data['retMsg']}")
            break

        records = data["result"]["list"]
        if not records:
            break

        all_data.extend(records)

        cursor = data["result"].get("nextPageCursor")
        if not cursor:
            break

        time.sleep(0.3)

    if not all_data:
        return pd.DataFrame(columns=["timestamp", "buy_ratio", "sell_ratio"])

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["buy_ratio"] = df["buyRatio"].astype(float)
    df["sell_ratio"] = df["sellRatio"].astype(float)
    df = df[["timestamp", "buy_ratio", "sell_ratio"]].sort_values("timestamp").reset_index(drop=True)
    return df


def collect_long_short_ratios() -> None:
    """BTC, ETH 롱/숏 비율을 수집하여 저장한다."""
    for name, sym in SYMBOLS.items():
        print(f"[{name}] 롱/숏 비율 수집 중...")
        ls_path = DATA_RAW / sym["dir"] / "long_short_ratio_4h.parquet"

        df = fetch_long_short_ratio(sym["bybit"])
        if df.empty:
            print(f"[{name}] 롱/숏 비율 데이터 없음")
            continue

        print(f"[{name}] 롱/숏 비율 {len(df)}건 수집 ({df['timestamp'].min()} ~ {df['timestamp'].max()})")
        df.to_parquet(ls_path, index=False, compression="snappy")


# ══════════════════════════════════════════════════════
# 작업 2: 탐색적 분석
# ══════════════════════════════════════════════════════

def load_funding_rate(symbol_name: str) -> pd.DataFrame:
    """펀딩비 parquet 로드."""
    sym = SYMBOLS[symbol_name]
    path = DATA_RAW / sym["dir"] / "funding_rate.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_ohlcv_1h(symbol_name: str) -> pd.DataFrame:
    """1h OHLCV 월별 parquet을 병합하여 로드."""
    sym = SYMBOLS[symbol_name]
    ohlcv_dir = DATA_RAW / sym["dir"] / "1h"
    files = sorted(ohlcv_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_oi(symbol_name: str) -> pd.DataFrame:
    """OI parquet 로드."""
    sym = SYMBOLS[symbol_name]
    path = DATA_RAW / sym["dir"] / "open_interest_1h.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_long_short_ratio(symbol_name: str) -> pd.DataFrame:
    """롱/숏 비율 parquet 로드."""
    sym = SYMBOLS[symbol_name]
    path = DATA_RAW / sym["dir"] / "long_short_ratio_4h.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── 2-1: 기초 통계 ──────────────────────────────────

def basic_stats(fr: pd.DataFrame) -> dict:
    """펀딩비 기초 통계를 계산한다."""
    rates = fr["funding_rate"]
    stats = {
        "count": len(rates),
        "mean": rates.mean(),
        "median": rates.median(),
        "std": rates.std(),
        "min": rates.min(),
        "max": rates.max(),
        "skewness": rates.skew(),
        "kurtosis": rates.kurtosis(),
        "positive_pct": (rates > 0).mean() * 100,
        "annual_simple": rates.mean() * FUNDING_PER_DAY * 365 * 100,  # %
    }
    return stats


def monthly_avg(fr: pd.DataFrame) -> pd.Series:
    """월별 평균 펀딩비."""
    fr = fr.copy()
    fr["month"] = fr["timestamp"].dt.to_period("M")
    return fr.groupby("month")["funding_rate"].mean()


# ── 2-2: 자기상관 분석 ──────────────────────────────

def autocorrelation_analysis(fr: pd.DataFrame, max_lag: int = 10) -> dict:
    """ACF, PACF 계산."""
    rates = fr["funding_rate"].dropna().values
    acf_vals = acf(rates, nlags=max_lag, fft=True)
    pacf_vals = pacf(rates, nlags=min(max_lag, 5))
    return {"acf": acf_vals, "pacf": pacf_vals}


# ── 2-3: 연속 양수/음수 구간 ────────────────────────

def consecutive_runs(fr: pd.DataFrame) -> dict:
    """연속 양수/음수 구간 길이 분포를 계산한다."""
    signs = (fr["funding_rate"] > 0).astype(int)
    # 부호 전환 시점 감지
    changes = signs.diff().fillna(1).abs().cumsum()
    groups = signs.groupby(changes)

    positive_runs = []
    negative_runs = []
    for _, g in groups:
        if len(g) == 0:
            continue
        val = g.iloc[0]
        run_len = len(g)
        if val == 1:
            positive_runs.append(run_len)
        else:
            negative_runs.append(run_len)

    pos = np.array(positive_runs) if positive_runs else np.array([0])
    neg = np.array(negative_runs) if negative_runs else np.array([0])

    return {
        "positive": {
            "count": len(positive_runs),
            "mean": pos.mean(),
            "median": np.median(pos),
            "p75": np.percentile(pos, 75),
            "max": pos.max(),
        },
        "negative": {
            "count": len(negative_runs),
            "mean": neg.mean(),
            "median": np.median(neg),
            "p75": np.percentile(neg, 75),
            "max": neg.max(),
        },
    }


# ── 2-4/2-5: 상관관계 분석 ──────────────────────────

def resample_ohlcv_8h(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """1h OHLCV → 8h 리샘플링. 펀딩비 결제 시점(0,8,16 UTC)에 맞춤."""
    df = ohlcv.set_index("timestamp")
    resampled = df.resample("8h", origin="epoch").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    resampled = resampled.reset_index()
    return resampled


def resample_oi_8h(oi: pd.DataFrame) -> pd.DataFrame:
    """1h OI → 8h 리샘플링 (마지막 값)."""
    if oi.empty:
        return pd.DataFrame()
    # OI 컬럼 이름 확인
    oi_col = [c for c in oi.columns if c not in ("timestamp", "symbol")]
    if not oi_col:
        return pd.DataFrame()
    oi_col = oi_col[0]
    df = oi.set_index("timestamp")
    resampled = df[[oi_col]].resample("8h", origin="epoch").last().dropna()
    resampled = resampled.reset_index()
    resampled.rename(columns={oi_col: "open_interest"}, inplace=True)
    return resampled


def build_correlation_dataset(fr: pd.DataFrame, ohlcv_8h: pd.DataFrame,
                              oi_8h: pd.DataFrame, ls: pd.DataFrame) -> pd.DataFrame:
    """8h 단위 상관관계 분석용 데이터셋 구축."""
    merged = fr[["timestamp", "funding_rate"]].copy()

    # OHLCV 파생 변수
    if not ohlcv_8h.empty:
        ohlcv_8h = ohlcv_8h.copy()
        ohlcv_8h["return_8h"] = ohlcv_8h["close"].pct_change()
        ohlcv_8h["return_24h"] = ohlcv_8h["close"].pct_change(3)
        ohlcv_8h["volatility_24h"] = ohlcv_8h["return_8h"].rolling(3).std()
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            ohlcv_8h[["timestamp", "return_8h", "return_24h", "volatility_24h"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    # OI 파생 변수
    if not oi_8h.empty:
        oi_8h = oi_8h.copy()
        oi_8h["oi_change_8h"] = oi_8h["open_interest"].pct_change()
        oi_8h["oi_change_24h"] = oi_8h["open_interest"].pct_change(3)
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            oi_8h[["timestamp", "oi_change_8h", "oi_change_24h"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    # 롱/숏 비율
    if not ls.empty:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            ls[["timestamp", "buy_ratio"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    return merged.dropna()


def compute_correlations(merged: pd.DataFrame) -> dict[str, float]:
    """펀딩비와 각 변수의 상관계수."""
    cols = [c for c in merged.columns if c not in ("timestamp", "funding_rate")]
    corrs = {}
    for c in cols:
        r = merged["funding_rate"].corr(merged[c])
        corrs[c] = r
    return dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))


# ── 2-6: 비용 모델 ──────────────────────────────────

def cost_model(avg_fr: float) -> dict:
    """손익분기 계산."""
    daily_fr = avg_fr * FUNDING_PER_DAY
    if daily_fr > 0:
        breakeven_days = COST_ROUNDTRIP / daily_fr
    else:
        breakeven_days = float("inf")
    return {
        "roundtrip_cost": COST_ROUNDTRIP * 100,
        "daily_fr_pct": daily_fr * 100,
        "breakeven_days": breakeven_days,
    }


# ── 2-7: 분기별 수익 ────────────────────────────────

def quarterly_returns(fr: pd.DataFrame) -> pd.DataFrame:
    """분기별 펀딩비 누적 수익 (비용 차감 전)."""
    df = fr.copy()
    df["quarter"] = df["timestamp"].dt.to_period("Q")
    quarterly = df.groupby("quarter")["funding_rate"].sum() * 100  # %
    return quarterly


# ══════════════════════════════════════════════════════
# 작업 3: 피처 데이터셋 생성
# ══════════════════════════════════════════════════════

def build_feature_dataset(symbol_name: str) -> pd.DataFrame:
    """8h 단위 통합 피처 데이터셋 생성."""
    fr = load_funding_rate(symbol_name)
    ohlcv = load_ohlcv_1h(symbol_name)
    oi = load_oi(symbol_name)
    ls = load_long_short_ratio(symbol_name)

    # 기본: 펀딩비
    df = fr[["timestamp", "funding_rate"]].copy()

    # 타겟: 다음 펀딩비
    df["next_funding_rate"] = df["funding_rate"].shift(-1)

    # 펀딩비 파생 피처
    df["fr_ma_3"] = df["funding_rate"].rolling(3).mean()
    df["fr_ma_7"] = df["funding_rate"].rolling(7).mean()
    df["fr_ma_21"] = df["funding_rate"].rolling(21).mean()
    df["fr_std_7"] = df["funding_rate"].rolling(7).std()
    fr_std_21 = df["funding_rate"].rolling(21).std()
    df["fr_zscore"] = (df["funding_rate"] - df["fr_ma_21"]) / fr_std_21.replace(0, np.nan)

    # OHLCV 8h 리샘플링 후 병합
    ohlcv_8h = resample_ohlcv_8h(ohlcv)
    if not ohlcv_8h.empty:
        ohlcv_8h["return_8h"] = ohlcv_8h["close"].pct_change()
        ohlcv_8h["return_24h"] = ohlcv_8h["close"].pct_change(3)
        ohlcv_8h["volatility_24h"] = ohlcv_8h["return_8h"].rolling(3).std()

        # RSI 14 (1h 봉 기준, 마지막 값)
        ohlcv_1h = ohlcv.copy()
        delta = ohlcv_1h["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        ohlcv_1h["rsi_14"] = 100 - 100 / (1 + rs)
        rsi_8h = ohlcv_1h.set_index("timestamp")[["rsi_14"]].resample("8h", origin="epoch").last().dropna().reset_index()

        df = pd.merge_asof(
            df.sort_values("timestamp"),
            ohlcv_8h[["timestamp", "return_8h", "return_24h", "volatility_24h"]].sort_values("timestamp"),
            on="timestamp", direction="backward",
        )
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            rsi_8h.sort_values("timestamp"),
            on="timestamp", direction="backward",
        )

    # OI 8h
    oi_8h = resample_oi_8h(oi)
    if not oi_8h.empty:
        oi_8h["oi_change_8h"] = oi_8h["open_interest"].pct_change()
        oi_8h["oi_change_24h"] = oi_8h["open_interest"].pct_change(3)
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            oi_8h[["timestamp", "oi_change_8h", "oi_change_24h"]].sort_values("timestamp"),
            on="timestamp", direction="backward",
        )

    # 롱/숏 비율
    if not ls.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            ls[["timestamp", "buy_ratio", "sell_ratio"]].sort_values("timestamp"),
            on="timestamp", direction="backward",
        )

    # 시간 피처
    df["hour_utc"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df


# ══════════════════════════════════════════════════════
# 작업 4: 결과 출력
# ══════════════════════════════════════════════════════

def print_table(headers: list[str], rows: list[list[str]],
                col_widths: list[int] | None = None) -> None:
    """간단한 텍스트 테이블 출력."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]

    def row_str(cells: list[str]) -> str:
        return "│" + "│".join(str(c).center(w) for c, w in zip(cells, col_widths)) + "│"

    top = "┌" + "┬".join("─" * w for w in col_widths) + "┐"
    mid = "├" + "┼".join("─" * w for w in col_widths) + "┤"
    bot = "└" + "┴".join("─" * w for w in col_widths) + "┘"

    print(top)
    print(row_str(headers))
    print(mid)
    for r in rows:
        print(row_str(r))
    print(bot)


def run_analysis() -> dict:
    """전체 분석을 실행하고 결과를 딕셔너리로 반환한다."""
    results: dict = {}

    for name in SYMBOLS:
        fr = load_funding_rate(name)
        ohlcv = load_ohlcv_1h(name)
        oi = load_oi(name)
        ls = load_long_short_ratio(name)

        res: dict = {}
        res["stats"] = basic_stats(fr)
        res["monthly"] = monthly_avg(fr)
        res["acf"] = autocorrelation_analysis(fr)
        res["runs"] = consecutive_runs(fr)
        res["cost"] = cost_model(res["stats"]["mean"])
        res["quarterly"] = quarterly_returns(fr)

        # 상관관계
        ohlcv_8h = resample_ohlcv_8h(ohlcv)
        oi_8h = resample_oi_8h(oi)
        merged = build_correlation_dataset(fr, ohlcv_8h, oi_8h, ls)
        res["correlations"] = compute_correlations(merged)

        results[name] = res

    return results


def print_results(results: dict) -> dict:
    """결과를 포맷팅하여 출력하고, Phase 1 진입 판단 결과를 반환한다."""

    print("\n" + "=" * 56)
    print("  펀딩비 차익거래 Phase 0: 탐색적 분석 결과")
    print("=" * 56)

    # ── 1. 기초 통계
    print("\n1. 기초 통계")
    w = [12, 14, 14]
    print_table(
        ["메트릭", "BTC", "ETH"],
        [
            ["건수", f"{results['BTC']['stats']['count']:,}", f"{results['ETH']['stats']['count']:,}"],
            ["평균 FR", f"{results['BTC']['stats']['mean']*100:.4f}%", f"{results['ETH']['stats']['mean']*100:.4f}%"],
            ["중위 FR", f"{results['BTC']['stats']['median']*100:.4f}%", f"{results['ETH']['stats']['median']*100:.4f}%"],
            ["표준편차", f"{results['BTC']['stats']['std']*100:.4f}%", f"{results['ETH']['stats']['std']*100:.4f}%"],
            ["최소", f"{results['BTC']['stats']['min']*100:.4f}%", f"{results['ETH']['stats']['min']*100:.4f}%"],
            ["최대", f"{results['BTC']['stats']['max']*100:.4f}%", f"{results['ETH']['stats']['max']*100:.4f}%"],
            ["왜도", f"{results['BTC']['stats']['skewness']:.2f}", f"{results['ETH']['stats']['skewness']:.2f}"],
            ["첨도", f"{results['BTC']['stats']['kurtosis']:.2f}", f"{results['ETH']['stats']['kurtosis']:.2f}"],
            ["양수비율", f"{results['BTC']['stats']['positive_pct']:.1f}%", f"{results['ETH']['stats']['positive_pct']:.1f}%"],
            ["연간추정", f"{results['BTC']['stats']['annual_simple']:.1f}%", f"{results['ETH']['stats']['annual_simple']:.1f}%"],
        ],
        col_widths=w,
    )

    # ── 2. 자기상관
    print("\n2. 자기상관 (예측 가능성)")
    lags_to_show = [1, 2, 3, 5, 7, 10]
    rows = []
    for lag in lags_to_show:
        btc_acf = results["BTC"]["acf"]["acf"]
        eth_acf = results["ETH"]["acf"]["acf"]
        if lag < len(btc_acf):
            rows.append([
                f"{lag} ({lag*8}h)",
                f"{btc_acf[lag]:.3f}",
                f"{eth_acf[lag]:.3f}",
            ])
    print_table(["Lag", "BTC ACF", "ETH ACF"], rows, [12, 10, 10])

    print("\n  편자기상관 (PACF):")
    pacf_rows = []
    for lag in range(1, 6):
        btc_p = results["BTC"]["acf"]["pacf"]
        eth_p = results["ETH"]["acf"]["pacf"]
        if lag < len(btc_p):
            pacf_rows.append([f"{lag}", f"{btc_p[lag]:.3f}", f"{eth_p[lag]:.3f}"])
    print_table(["Lag", "BTC PACF", "ETH PACF"], pacf_rows, [8, 10, 10])

    btc_lag1 = results["BTC"]["acf"]["acf"][1]
    eth_lag1 = results["ETH"]["acf"]["acf"][1]
    acf_pass = btc_lag1 > 0.5 and eth_lag1 > 0.5
    print(f"\n  검증: lag-1 ACF > 0.5 → BTC {btc_lag1:.3f} {'✅' if btc_lag1 > 0.5 else '❌'}, "
          f"ETH {eth_lag1:.3f} {'✅' if eth_lag1 > 0.5 else '❌'}")

    # ── 3. 연속 양수 구간
    print("\n3. 연속 양수/음수 구간 (회 = 8시간 단위)")
    for name in ["BTC", "ETH"]:
        runs = results[name]["runs"]
        print(f"\n  [{name}] 양수 구간:")
        print(f"    구간수: {runs['positive']['count']}, "
              f"평균: {runs['positive']['mean']:.1f}회 ({runs['positive']['mean']*8/24:.1f}일), "
              f"중위: {runs['positive']['median']:.0f}회 ({runs['positive']['median']*8/24:.1f}일), "
              f"75%ile: {runs['positive']['p75']:.0f}회 ({runs['positive']['p75']*8/24:.1f}일), "
              f"최대: {runs['positive']['max']}회 ({runs['positive']['max']*8/24:.1f}일)")
        print(f"  [{name}] 음수 구간:")
        print(f"    구간수: {runs['negative']['count']}, "
              f"평균: {runs['negative']['mean']:.1f}회 ({runs['negative']['mean']*8/24:.1f}일), "
              f"중위: {runs['negative']['median']:.0f}회 ({runs['negative']['median']*8/24:.1f}일)")

    # ── 4. 비용 모델
    print("\n4. 비용 모델")
    for name in ["BTC", "ETH"]:
        c = results[name]["cost"]
        print(f"  [{name}] 왕복 비용: {c['roundtrip_cost']:.2f}%, "
              f"일 펀딩비: {c['daily_fr_pct']:.4f}%, "
              f"손익분기: {c['breakeven_days']:.1f}일")

    btc_be = results["BTC"]["cost"]["breakeven_days"]
    btc_pos_median = results["BTC"]["runs"]["positive"]["median"] * 8 / 24
    be_pass = btc_pos_median > btc_be
    print(f"\n  검증: 손익분기({btc_be:.1f}일) < 연속양수 중위값({btc_pos_median:.1f}일) → {'✅' if be_pass else '❌'}")

    # ── 5. 분기별 수익
    print("\n5. 시기별 수익 (비용 차감 전, 펀딩비 합산 %)")
    btc_q = results["BTC"]["quarterly"]
    eth_q = results["ETH"]["quarterly"]
    all_quarters = sorted(set(btc_q.index.tolist() + eth_q.index.tolist()))
    q_rows = []
    positive_quarters = 0
    total_quarters = 0
    for q in all_quarters:
        btc_val = btc_q.get(q, 0)
        eth_val = eth_q.get(q, 0)
        q_rows.append([str(q), f"{btc_val:+.2f}%", f"{eth_val:+.2f}%"])
        total_quarters += 1
        if btc_val > 0 and eth_val > 0:
            positive_quarters += 1

    print_table(["분기", "BTC 누적", "ETH 누적"], q_rows, [12, 12, 12])

    # 양쪽 모두 양수인 분기 비율로 판단
    quarter_pass = positive_quarters >= (total_quarters * 2 / 3)
    print(f"\n  BTC+ETH 모두 양수 분기: {positive_quarters}/{total_quarters} "
          f"({'✅' if quarter_pass else '❌'} 2/3 이상)")

    # ── 6. 상관관계
    print("\n6. 피처 상관관계 (|r| 기준 상위)")
    for name in ["BTC", "ETH"]:
        corrs = results[name]["correlations"]
        print(f"\n  [{name}]")
        for var, r in list(corrs.items())[:6]:
            print(f"    funding_rate vs {var:20s}: r = {r:+.3f}")

    # ── 7. Phase 1 진입 판단
    print("\n" + "=" * 56)
    print("  7. Phase 1 진입 판단")
    print("=" * 56)

    # 양수비율 체크
    pos_pct_btc = results["BTC"]["stats"]["positive_pct"]
    pos_pct_eth = results["ETH"]["stats"]["positive_pct"]
    pos_pass = pos_pct_btc > 60 and pos_pct_eth > 60

    checks = {
        "자기상관 lag-1 > 0.5": acf_pass,
        f"양수비율 > 60% (BTC {pos_pct_btc:.1f}%, ETH {pos_pct_eth:.1f}%)": pos_pass,
        f"손익분기({btc_be:.1f}일) < 연속양수 중위값({btc_pos_median:.1f}일)": be_pass,
        f"분기별 양수 {positive_quarters}/{total_quarters} >= 2/3": quarter_pass,
    }

    passed = 0
    for desc, ok in checks.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}")
        if ok:
            passed += 1

    total = len(checks)
    go = passed >= 3
    print(f"\n  결과: {passed}/{total} 통과 → {'Phase 1 진행 ✅' if go else 'Phase 1 보류 ❌'}")

    return {"passed": passed, "total": total, "go": go, "checks": checks}


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="펀딩비 차익거래 Phase 0 EDA")
    parser.add_argument("--skip-collection", action="store_true",
                        help="데이터 수집 건너뛰기")
    args = parser.parse_args()

    # ── 작업 1: 데이터 수집
    if not args.skip_collection:
        print("=" * 56)
        print("  작업 1: 데이터 수집")
        print("=" * 56)
        print("\n[1-1] 펀딩비 최신화...")
        update_funding_rates()
        print("\n[1-2] 롱/숏 비율 수집...")
        collect_long_short_ratios()
    else:
        print("데이터 수집 건너뜀 (--skip-collection)")

    # ── 작업 2: 탐색적 분석
    print("\n분석 실행 중...")
    results = run_analysis()
    judgment = print_results(results)

    # ── 작업 3: 피처 데이터셋 생성
    print("\n" + "=" * 56)
    print("  작업 3: 피처 데이터셋 생성")
    print("=" * 56)

    for name in SYMBOLS:
        feat_df = build_feature_dataset(name)
        out_path = DATA_PROCESSED / f"{SYMBOLS[name]['bybit']}_8h_funding_features.parquet"
        # next_funding_rate이 NaN인 마지막 행 제거
        feat_df = feat_df.dropna(subset=["next_funding_rate"])
        feat_df.to_parquet(out_path, index=False, compression="snappy")
        print(f"  [{name}] {len(feat_df)}건 → {out_path.name}")
        print(f"    컬럼: {list(feat_df.columns)}")
        print(f"    기간: {feat_df['timestamp'].min()} ~ {feat_df['timestamp'].max()}")

    print("\n완료.")


if __name__ == "__main__":
    main()
