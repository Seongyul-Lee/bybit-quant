"""BTC 1h 평균회귀 회귀 전략 (v2).

LGBMRegressorStrategy를 상속하여 BTC 평균회귀 전략을 구현.
v1과 달리 ForwardReturn 라벨링 + 회귀로 양방향 시그널 생성.
모델이 피처(bb_position, rsi_14 등)에서 과매도/과매수 패턴을 자체 학습.
"""

from strategies._common.regressor_strategy import LGBMRegressorStrategy


class BTCMeanReversionV2Strategy(LGBMRegressorStrategy):
    """BTC 1h 평균회귀 회귀 전략.

    LGBMRegressorStrategy의 모든 기능을 상속.
    추가 커스터마이징이 필요하면 여기서 오버라이드.
    """

    pass
