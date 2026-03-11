"""전략 기본 인터페이스 모듈.

모든 전략은 BaseStrategy를 상속하고 generate_signal을 구현해야 한다.
백테스팅 엔진과 실거래 엔진이 동일한 인터페이스로 전략을 호출한다.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """전략 기본 추상 클래스.

    모든 전략은 이 클래스를 상속하고 generate_signal 메서드를 구현해야 한다.
    백테스팅과 실거래에서 동일한 인터페이스를 보장한다.

    Attributes:
        config: 전략 파라미터 딕셔너리.
        strategy_name: 전략 이름.
        symbol: 거래 심볼.
        timeframe: 타임프레임.
    """

    def __init__(self, config: dict) -> None:
        """BaseStrategy 초기화.

        Args:
            config: 전략 설정 딕셔너리 (파라미터, 심볼, 타임프레임 등).
        """
        self.config = config
        self.strategy_name: str = config.get("strategy_name", "unknown")
        self.symbol: str = config.get("symbol", "BTCUSDT")
        self.timeframe: str = config.get("timeframe", "1h")

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """매매 신호 + 확신도를 반환.

        Args:
            df: OHLCV + 피처 데이터프레임.
                최소 컬럼: timestamp, open, high, low, close, volume.

        Returns:
            (signal, probability) 튜플.
            signal: 1(매수) 또는 0(비매수).
            probability: 매수 확률 (0.0 ~ 1.0). 자본 배분에 활용.
        """
        pass

    @abstractmethod
    def generate_signals_vectorized(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """벡터화 신호 + 확신도 시리즈 반환 (백테스트 전용).

        Args:
            df: OHLCV + 피처 데이터프레임.

        Returns:
            (signal_series, probability_series) 튜플.
        """
        pass

    def get_params(self) -> dict:
        """전략 파라미터를 딕셔너리로 반환.

        백테스트 결과 저장 시 파라미터를 기록하기 위해 사용한다.

        Returns:
            전략 설정 딕셔너리.
        """
        return self.config
