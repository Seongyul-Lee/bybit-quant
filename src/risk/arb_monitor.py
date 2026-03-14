"""펀딩비 차익거래 전용 리스크 모니터.

기존 RiskManager, PortfolioRiskManager와 독립적으로 동작.
차익거래 특유의 리스크(베이시스, 델타, 마진, 펀딩비 추세)를 감시.
"""

import time
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("arb_risk_monitor")


class ArbRiskMonitor:
    """펀딩비 차익거래 리스크 모니터.

    매 폴링마다 check_all()을 호출하여 리스크 상태를 반환.
    알림 발송은 main.py에서 처리 (이 클래스는 판단만).

    Attributes:
        config: 리스크 설정 딕셔너리.
    """

    def __init__(self, config: dict) -> None:
        """초기화.

        Args:
            config: strategies/funding_arb/config.yaml의 risk 섹션.
        """
        # 베이시스
        self.basis_warn_pct = config.get("basis_warn_pct", 0.01)
        self.basis_critical_pct = config.get("basis_critical_pct", 0.02)

        # 마진
        self.margin_warn_pct = config.get("margin_warn_pct", 0.50)
        self.margin_critical_pct = config.get("margin_critical_pct", 0.30)

        # 연속 음수 펀딩비
        self.max_consecutive_negative = config.get("max_consecutive_negative", 6)

        # 슬리피지
        self.max_entry_slippage_pct = config.get("max_entry_slippage_pct", 0.005)

        # 누적 손실
        self.max_cumulative_loss_pct = config.get("max_cumulative_loss_pct", 0.03)

        # 내부 상태
        self._consecutive_negative: int = 0
        self._cumulative_funding: float = 0.0
        self._basis_history: list[float] = []

        # 알림 빈도 제한: {alert_type: last_sent_timestamp}
        self._last_alert_time: dict[str, float] = {}
        self._alert_cooldown_sec: float = 3600.0  # 1시간

    def should_send_alert(self, alert_type: str, level: str) -> bool:
        """알림 빈도 제한 체크.

        critical은 항상 전송. warn/info는 1시간에 1회만.

        Args:
            alert_type: 알림 종류 (예: "basis", "margin", "funding_trend").
            level: 알림 레벨 ("warn", "critical").

        Returns:
            True이면 알림 전송 가능.
        """
        if level == "critical":
            return True

        now = time.time()
        last_sent = self._last_alert_time.get(alert_type, 0.0)
        if now - last_sent >= self._alert_cooldown_sec:
            self._last_alert_time[alert_type] = now
            return True
        return False

    def check_basis(
        self, spot_price: float, perp_price: float
    ) -> dict[str, Any]:
        """베이시스 리스크 체크.

        Args:
            spot_price: 현물 가격.
            perp_price: 선물 가격.

        Returns:
            {"basis_pct": float, "level": str, "message": str | None}
        """
        if spot_price <= 0:
            return {"basis_pct": 0, "level": "normal", "message": None}

        basis_pct = (perp_price - spot_price) / spot_price
        self._basis_history.append(basis_pct)
        if len(self._basis_history) > 100:
            self._basis_history = self._basis_history[-100:]

        if abs(basis_pct) > self.basis_critical_pct:
            return {
                "basis_pct": basis_pct,
                "level": "critical",
                "message": (
                    f"베이시스 긴급: {basis_pct:+.2%} "
                    f"(임계 ±{self.basis_critical_pct:.0%})"
                ),
            }
        elif abs(basis_pct) > self.basis_warn_pct:
            return {
                "basis_pct": basis_pct,
                "level": "warn",
                "message": f"베이시스 경고: {basis_pct:+.2%}",
            }
        return {"basis_pct": basis_pct, "level": "normal", "message": None}

    def check_margin(
        self, exchange: Any, symbol_perp: str
    ) -> dict[str, Any]:
        """마진율 체크.

        Bybit V5 API로 포지션의 마진율 조회.
        info 필드에서 positionMM, positionBalance를 직접 추출.

        Args:
            exchange: ccxt.bybit 인스턴스 (선물).
            symbol_perp: 선물 심볼.

        Returns:
            {"margin_ratio": float | None, "level": str, "message": str | None}
        """
        try:
            positions = exchange.fetch_positions([symbol_perp])
            if not positions:
                return {"margin_ratio": None, "level": "unknown", "message": None}

            pos = positions[0]

            # Bybit V5 네이티브 필드에서 직접 추출 (ccxt 래핑보다 안정적)
            info = pos.get("info", {})
            position_mm = float(info.get("positionMM", 0))
            wallet_balance = float(info.get("positionBalance", 0) or 0)

            if wallet_balance > 0:
                margin_usage = position_mm / wallet_balance  # 높을수록 위험
            else:
                margin_usage = 0

            if margin_usage > (1 - self.margin_critical_pct):
                return {
                    "margin_ratio": margin_usage,
                    "level": "critical",
                    "message": f"마진 긴급: 사용률 {margin_usage:.1%} (청산 임박)",
                }
            elif margin_usage > (1 - self.margin_warn_pct):
                return {
                    "margin_ratio": margin_usage,
                    "level": "warn",
                    "message": f"마진 경고: 사용률 {margin_usage:.1%}",
                }
            return {"margin_ratio": margin_usage, "level": "normal", "message": None}

        except Exception as e:
            logger.warning(f"마진 조회 실패 ({symbol_perp}): {e}")
            return {
                "margin_ratio": None,
                "level": "unknown",
                "message": f"마진 조회 실패: {e}",
            }

    def check_funding_trend(self, funding_rate: float) -> dict[str, Any]:
        """펀딩비 추세 체크.

        연속 음수 펀딩비 횟수를 추적.

        Args:
            funding_rate: 최근 결제된 펀딩비.

        Returns:
            {"consecutive_negative": int, "level": str, "message": str | None}
        """
        if funding_rate < 0:
            self._consecutive_negative += 1
        else:
            self._consecutive_negative = 0
        self._cumulative_funding += funding_rate

        if self._consecutive_negative >= self.max_consecutive_negative:
            return {
                "consecutive_negative": self._consecutive_negative,
                "level": "critical",
                "message": (
                    f"연속 음수 펀딩비 {self._consecutive_negative}회 "
                    f"({self._consecutive_negative * 8}시간)"
                ),
            }
        elif self._consecutive_negative >= self.max_consecutive_negative // 2:
            return {
                "consecutive_negative": self._consecutive_negative,
                "level": "warn",
                "message": f"연속 음수 펀딩비 {self._consecutive_negative}회",
            }
        return {
            "consecutive_negative": self._consecutive_negative,
            "level": "normal",
            "message": None,
        }

    def check_entry_slippage(
        self, spot_fill_price: float, perp_fill_price: float
    ) -> dict[str, Any]:
        """진입 슬리피지 체크.

        현물 매수 체결가와 선물 숏 체결가의 차이를 확인.

        Args:
            spot_fill_price: 현물 체결가.
            perp_fill_price: 선물 체결가.

        Returns:
            {"slippage_pct": float, "level": str, "message": str | None}
        """
        if spot_fill_price <= 0:
            return {"slippage_pct": 0, "level": "unknown", "message": None}

        # 이상적: perp > spot (선물 프리미엄 = 추가 수익)
        # 나쁜 경우: spot > perp (현물이 비싸게 체결 = 진입 손실)
        slippage_pct = (spot_fill_price - perp_fill_price) / spot_fill_price

        if abs(slippage_pct) > self.max_entry_slippage_pct:
            return {
                "slippage_pct": slippage_pct,
                "level": "warn",
                "message": (
                    f"진입 슬리피지 {slippage_pct:+.2%}: "
                    f"현물 {spot_fill_price:.2f} / 선물 {perp_fill_price:.2f}"
                ),
            }
        return {"slippage_pct": slippage_pct, "level": "normal", "message": None}

    def check_all(
        self,
        spot_price: float,
        perp_price: float,
        exchange: Any,
        symbol_perp: str,
    ) -> list[dict[str, Any]]:
        """전체 리스크 체크.

        Args:
            spot_price: 현물 가격.
            perp_price: 선물 가격.
            exchange: ccxt.bybit 인스턴스.
            symbol_perp: 선물 심볼.

        Returns:
            경고/긴급 메시지 리스트 (level이 normal이 아닌 것만).
        """
        alerts: list[dict[str, Any]] = []

        basis = self.check_basis(spot_price, perp_price)
        if basis["level"] != "normal":
            alerts.append(basis)

        margin = self.check_margin(exchange, symbol_perp)
        if margin["level"] not in ("normal", "unknown"):
            alerts.append(margin)

        return alerts

    def to_dict(self) -> dict:
        """상태 직렬화."""
        return {
            "consecutive_negative": self._consecutive_negative,
            "cumulative_funding": self._cumulative_funding,
            "basis_history_last10": self._basis_history[-10:],
        }

    def from_dict(self, data: dict) -> None:
        """상태 복원."""
        self._consecutive_negative = data.get("consecutive_negative", 0)
        self._cumulative_funding = data.get("cumulative_funding", 0.0)
        self._basis_history = data.get("basis_history_last10", [])
