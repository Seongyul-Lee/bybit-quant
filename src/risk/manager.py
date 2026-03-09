"""리스크 관리 모듈.

포지션 사이징, 손실 한도, Circuit Breaker 등
모든 리스크 가드레일을 관리한다.
전략 신호보다 항상 우선한다.
"""

import os
from typing import Optional

import yaml

from src.utils.logger import setup_logger

logger = setup_logger("risk")


class CircuitBreaker:
    """시스템 이상 또는 극단적 시장 상황에서 거래를 자동 중단.

    연속 손실 한도 초과 또는 변동성 임계값 초과 시 발동된다.
    리셋은 수동으로만 가능하다 (자동 리셋 금지).

    Attributes:
        consecutive_losses: 현재 연속 손실 횟수.
        is_tripped: Circuit Breaker 발동 여부.
    """

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        volatility_threshold: float = 0.05,
    ) -> None:
        """CircuitBreaker 초기화.

        Args:
            max_consecutive_losses: 연속 손실 허용 최대 횟수.
            volatility_threshold: 1시간 변동성 임계값 (5% = 0.05).
        """
        self.max_consecutive_losses = max_consecutive_losses
        self.volatility_threshold = volatility_threshold
        self.consecutive_losses: int = 0
        self.is_tripped: bool = False

    def record_trade(self, pnl: float) -> None:
        """거래 결과를 기록하고 연속 손실 카운터 업데이트.

        Args:
            pnl: 해당 거래의 손익.
        """
        if pnl < 0:
            self.consecutive_losses += 1
            logger.warning(f"연속 손실 {self.consecutive_losses}회")
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trip(f"연속 손실 {self.consecutive_losses}회 한도 초과")

    def check_volatility(self, current_volatility: float) -> None:
        """현재 시장 변동성을 확인하고 임계값 초과 시 발동.

        Args:
            current_volatility: 현재 1시간 변동성.
        """
        if current_volatility > self.volatility_threshold:
            self.trip(f"시장 변동성 {current_volatility:.2%} > 임계값 {self.volatility_threshold:.2%}")

    def trip(self, reason: str) -> None:
        """Circuit Breaker를 발동한다.

        Args:
            reason: 발동 사유.
        """
        self.is_tripped = True
        logger.critical(f"[Circuit Breaker 발동] 사유: {reason}")

    def reset(self) -> None:
        """Circuit Breaker를 수동 리셋한다. 자동 리셋 금지."""
        self.is_tripped = False
        self.consecutive_losses = 0
        logger.info("Circuit Breaker 수동 리셋 완료")


class RiskManager:
    """리스크 관리 총괄 클래스.

    포지션 사이징 계산, 손실 한도 체크, Circuit Breaker 관리,
    레버리지 제한 등 모든 리스크 가드레일을 통합 관리한다.

    Attributes:
        params: 리스크 파라미터 딕셔너리.
        circuit_breaker: CircuitBreaker 인스턴스.
    """

    def __init__(self, config_path: str = "config/risk_params.yaml") -> None:
        """RiskManager 초기화.

        Args:
            config_path: 리스크 파라미터 YAML 파일 경로.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.params: dict = yaml.safe_load(f)

        self.circuit_breaker = CircuitBreaker(
            max_consecutive_losses=self.params["circuit_breaker"]["max_consecutive_losses"],
            volatility_threshold=self.params["circuit_breaker"]["volatility_threshold"],
        )
        logger.info(f"RiskManager 초기화 완료: {config_path}")

    def calculate_position_size(
        self,
        portfolio_value: float,
        win_rate: float,
        reward_risk_ratio: float,
        kelly_fraction: float = 0.5,
    ) -> float:
        """Half Kelly Criterion 기반 포지션 사이즈 계산.

        Kelly = win_rate - (1 - win_rate) / reward_risk_ratio
        Half Kelly = Kelly * kelly_fraction

        Args:
            portfolio_value: 총 포트폴리오 가치.
            win_rate: 승률 (0~1).
            reward_risk_ratio: 보상/위험 비율.
            kelly_fraction: Kelly 비율 (기본 0.5 = Half Kelly).

        Returns:
            계산된 포지션 사이즈 (금액).
        """
        kelly = win_rate - (1 - win_rate) / reward_risk_ratio
        kelly = max(0.0, kelly)
        position_size = portfolio_value * kelly * kelly_fraction

        # 최대 포지션 비율 상한 적용
        max_size = portfolio_value * self.params["position"]["max_position_pct"]
        position_size = min(position_size, max_size)

        logger.info(f"포지션 사이즈: {position_size:.2f} (Kelly={kelly:.4f})")
        return position_size

    def calculate_atr_position_size(
        self,
        portfolio_value: float,
        atr: float,
        risk_per_trade: Optional[float] = None,
    ) -> float:
        """ATR 기반 변동성 조절 포지션 사이징.

        변동성(ATR)이 클수록 포지션을 줄인다.

        Args:
            portfolio_value: 총 포트폴리오 가치.
            atr: 현재 ATR 값.
            risk_per_trade: 거래당 위험 비율. None이면 설정 파일 기본값 사용.

        Returns:
            계산된 포지션 사이즈 (수량 단위).
        """
        risk_pct = risk_per_trade or self.params["trade"]["risk_per_trade_pct"]
        dollar_risk = portfolio_value * risk_pct
        if atr <= 0:
            return 0.0
        position_size = dollar_risk / atr

        max_size = portfolio_value * self.params["position"]["max_position_pct"]
        return min(position_size, max_size)

    def check_all(
        self,
        daily_pnl: float,
        portfolio_value: float,
        current_positions: int,
        current_volatility: float = 0.0,
    ) -> tuple[bool, str]:
        """주문 실행 전 모든 리스크 조건을 순서대로 체크.

        체크 순서:
        1. Circuit Breaker 발동 여부
        2. 일일 손실 한도 초과
        3. 동시 포지션 수 초과
        4. 변동성 체크

        Args:
            daily_pnl: 금일 누적 손익.
            portfolio_value: 총 포트폴리오 가치.
            current_positions: 현재 보유 포지션 수.
            current_volatility: 현재 1시간 변동성.

        Returns:
            (통과 여부, 사유) 튜플.
            통과 시 (True, "OK"), 실패 시 (False, 사유 문자열).
        """
        # 1. Circuit Breaker
        if self.circuit_breaker.is_tripped:
            return False, "Circuit Breaker 발동 상태"

        # 2. 일일 손실 한도
        if daily_pnl < 0:
            loss_pct = abs(daily_pnl) / portfolio_value
            if loss_pct > self.params["loss_limits"]["daily_loss_limit_pct"]:
                logger.warning(f"일일 손실 한도 초과: {loss_pct:.2%}")
                return False, f"일일 손실 한도 초과 ({loss_pct:.2%})"

        # 3. 동시 포지션 수
        max_positions = self.params["position"]["max_concurrent_positions"]
        if current_positions >= max_positions:
            return False, f"최대 동시 포지션 수 도달 ({current_positions}/{max_positions})"

        # 4. 변동성 체크
        if current_volatility > 0:
            self.circuit_breaker.check_volatility(current_volatility)
            if self.circuit_breaker.is_tripped:
                return False, "변동성 임계값 초과로 Circuit Breaker 발동"

        return True, "OK"

    def get_stop_take_profit(
        self,
        entry_price: float,
        side: str,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> tuple[float, float]:
        """손절/익절 가격 계산.

        Args:
            entry_price: 진입 가격.
            side: 포지션 방향 ("long" | "short").
            stop_loss_pct: 손절 비율. None이면 설정 기본값.
            take_profit_pct: 익절 비율. None이면 설정 기본값.

        Returns:
            (stop_loss_price, take_profit_price) 튜플.
        """
        sl = stop_loss_pct or self.params["trade"]["default_stop_loss_pct"]
        tp = take_profit_pct or self.params["trade"]["default_take_profit_pct"]

        if side == "long":
            stop_loss = entry_price * (1 - sl)
            take_profit = entry_price * (1 + tp)
        else:
            stop_loss = entry_price * (1 + sl)
            take_profit = entry_price * (1 - tp)

        return stop_loss, take_profit
