"""BTC 1h 모멘텀 회귀 전략 (v2).

LGBMRegressorStrategy를 상속하여 BTC 모멘텀 전략을 구현.
기본 클래스에서 양방향 시그널 + 동적 SL/TP + 동적 포지셔닝을 모두 처리하므로,
이 파일은 config 로딩과 전략 등록만 담당.
"""

from strategies._common.regressor_strategy import LGBMRegressorStrategy


class BTCMomentumV2Strategy(LGBMRegressorStrategy):
    """BTC 1h 모멘텀 회귀 전략.

    LGBMRegressorStrategy의 모든 기능을 상속.
    추가 커스터마이징이 필요하면 여기서 오버라이드.
    """

    pass
