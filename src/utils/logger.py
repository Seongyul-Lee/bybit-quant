"""로깅 설정 모듈.

날짜별 시스템 로그와 에러 전용 로그를 RotatingFileHandler로 관리한다.
"""

import logging
import os
from datetime import date
from logging.handlers import RotatingFileHandler


def setup_logger(name: str) -> logging.Logger:
    """모듈별 로거를 설정하고 반환.

    로그 파일:
    - logs/system/{날짜}.log — INFO 이상 (최대 10MB, 30개 백업)
    - logs/errors/{날짜}_errors.log — ERROR 이상 (최대 5MB, 30개 백업)

    Args:
        name: 로거 이름 (보통 모듈명).

    Returns:
        설정된 logging.Logger 인스턴스.
    """
    logger = logging.getLogger(name)

    # 이미 핸들러가 설정되어 있으면 중복 방지
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (디렉토리 존재 시에만)
    system_dir = "logs/system"
    error_dir = "logs/errors"

    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    system_handler = RotatingFileHandler(
        filename=os.path.join(system_dir, f"{date.today()}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=30,
        encoding="utf-8",
    )
    system_handler.setLevel(logging.INFO)
    system_handler.setFormatter(formatter)
    logger.addHandler(system_handler)

    error_handler = RotatingFileHandler(
        filename=os.path.join(error_dir, f"{date.today()}_errors.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=30,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger
