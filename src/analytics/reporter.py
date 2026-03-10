"""성과 분석 및 리포트 생성 모듈.

백테스트 결과와 실거래 성과를 분석하고 리포트를 생성한다.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("reporter")

TRADE_LOG_PATH = "reports/trades/trade_log.csv"


class Reporter:
    """성과 분석 및 리포트 생성 클래스.

    일별 PnL, 누적 수익률, 샤프 비율, MDD 등을 계산하고
    JSON/HTML 리포트를 생성한다.
    """

    def __init__(self, trade_log_path: str = TRADE_LOG_PATH) -> None:
        """Reporter 초기화.

        Args:
            trade_log_path: 거래 이력 CSV 파일 경로.
        """
        self.trade_log_path = trade_log_path

    def generate_daily_report(self, date_str: str) -> dict:
        """특정 날짜의 일일 성과 리포트를 생성.

        Args:
            date_str: 대상 날짜 문자열 (예: "2024-01-15").

        Returns:
            일일 성과 딕셔너리.
        """
        if not os.path.exists(self.trade_log_path):
            logger.warning(f"거래 이력 파일 없음: {self.trade_log_path}")
            return {"date": date_str, "trades_count": 0, "daily_pnl": 0}

        trade_log = pd.read_csv(self.trade_log_path)
        today_trades = trade_log[trade_log["timestamp"].str.startswith(date_str)]

        report = {
            "date": date_str,
            "trades_count": len(today_trades),
            "daily_pnl": float(today_trades["pnl"].sum()) if len(today_trades) > 0 else 0,
            "win_rate": float((today_trades["pnl"] > 0).mean()) if len(today_trades) > 0 else 0,
            "best_trade": float(today_trades["pnl"].max()) if len(today_trades) > 0 else 0,
            "worst_trade": float(today_trades["pnl"].min()) if len(today_trades) > 0 else 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # JSON 저장
        path = f"reports/live/{date_str}_daily.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"일일 리포트 생성: {path}")
        return report

    def save_backtest_result(
        self,
        strategy_name: str,
        params: dict,
        symbol: str,
        timeframe: str,
        period_start: str,
        period_end: str,
        metrics: dict,
    ) -> str:
        """백테스트 결과를 JSON으로 저장.

        Args:
            strategy_name: 전략 클래스 이름.
            params: 전략 파라미터.
            symbol: 심볼.
            timeframe: 타임프레임.
            period_start: 백테스트 시작일.
            period_end: 백테스트 종료일.
            metrics: 성과 지표 딕셔너리 (sharpe_ratio, max_drawdown 등).

        Returns:
            저장된 파일 경로.
        """
        result = {
            "strategy": strategy_name,
            "params": params,
            "symbol": symbol,
            "timeframe": timeframe,
            "period": {"start": period_start, "end": period_end},
            "metrics": metrics,
            "run_at": datetime.now(timezone.utc).isoformat(),
        }

        # 전략 폴더 이름 변환 (CamelCase → snake_case)
        folder_name = self._to_snake_case(strategy_name.replace("Strategy", ""))
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        base_dir = f"strategies/{folder_name}/backtest_results"
        os.makedirs(base_dir, exist_ok=True)

        # 기존 파일 카운트로 run 번호 결정
        existing = [f for f in os.listdir(base_dir) if f.startswith(date_str)]
        run_num = len(existing) + 1

        path = f"{base_dir}/{date_str}_run{run_num}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"백테스트 결과 저장: {path}")
        return path

    # 타임프레임별 연간 봉 수 매핑
    PERIODS_PER_YEAR: dict[str, int] = {
        "1m": 525_600,   # 365 * 24 * 60
        "5m": 105_120,   # 365 * 24 * 12
        "15m": 35_040,   # 365 * 24 * 4
        "30m": 17_520,   # 365 * 24 * 2
        "1h": 8_760,     # 365 * 24
        "2h": 4_380,     # 365 * 12
        "4h": 2_190,     # 365 * 6
        "6h": 1_460,     # 365 * 4
        "12h": 730,      # 365 * 2
        "1d": 365,
        "1w": 52,
    }

    @staticmethod
    def calculate_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        timeframe: str = "1d",
        trade_stats: Optional[dict] = None,
    ) -> dict:
        """수익률 시리즈에서 주요 성과 지표를 계산.

        Args:
            returns: 수익률 시리즈 (봉 단위).
            risk_free_rate: 무위험 수익률 (연간).
            timeframe: 데이터 타임프레임 (연환산 계수 결정에 사용).
            trade_stats: 거래 기반 지표 딕셔너리 (선택).
                키: total_trades, win_rate, profit_factor.
                제공 시 봉 기준 값 대신 거래 기반 값을 사용한다.

        Returns:
            성과 지표 딕셔너리 (total_return, sharpe_ratio, max_drawdown 등).
        """
        periods_per_year = Reporter.PERIODS_PER_YEAR.get(timeframe, 252)
        total_return = float((1 + returns).prod() - 1)

        # 샤프 비율 (연간화 — 타임프레임 기반)
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe_ratio = float(
            np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        ) if excess_returns.std() > 0 else 0.0

        # MDD
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())

        # 거래 기반 지표: trade_stats가 있으면 사용, 없으면 봉 기준 fallback
        if trade_stats:
            total_trades = trade_stats.get("total_trades", 0)
            win_rate = trade_stats.get("win_rate", 0.0)
            profit_factor = trade_stats.get("profit_factor", 0.0)
        else:
            total_trades = len(returns)
            win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0
            gross_profit = float(returns[returns > 0].sum())
            gross_loss = float(abs(returns[returns < 0].sum()))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
        }

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """CamelCase를 snake_case로 변환."""
        import re
        s = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        return s
