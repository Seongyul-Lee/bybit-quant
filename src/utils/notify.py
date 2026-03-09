"""텔레그램 알림 모듈.

거래 실행, Circuit Breaker 발동, 일일 성과 리포트 등
주요 이벤트를 텔레그램으로 전송한다.
"""

import asyncio
import os
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger("notify")


class TelegramNotifier:
    """텔레그램 봇을 통한 알림 발송 클래스.

    Attributes:
        token: 텔레그램 봇 토큰.
        chat_id: 메시지를 보낼 채팅 ID.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        """TelegramNotifier 초기화.

        Args:
            token: 텔레그램 봇 토큰. None이면 환경변수에서 로드.
            chat_id: 텔레그램 채팅 ID. None이면 환경변수에서 로드.
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._bot = None

    def _get_bot(self):
        """텔레그램 Bot 인스턴스를 지연 초기화."""
        if self._bot is None:
            from telegram import Bot
            self._bot = Bot(token=self.token)
        return self._bot

    async def send(self, message: str) -> bool:
        """텔레그램 메시지 전송.

        Args:
            message: 전송할 메시지 (Markdown 지원).

        Returns:
            전송 성공 여부.
        """
        if not self.token or not self.chat_id:
            logger.warning("텔레그램 설정 누락 (토큰 또는 채팅 ID)")
            return False

        try:
            bot = self._get_bot()
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown",
            )
            logger.info(f"텔레그램 전송 완료: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"텔레그램 전송 실패: {e}")
            return False

    async def send_daily_report(self, stats: dict) -> bool:
        """일일 성과 리포트를 텔레그램으로 전송.

        Args:
            stats: 성과 통계 딕셔너리.
                필수 키: date, daily_pnl, total_return, current_mdd, trades_today.

        Returns:
            전송 성공 여부.
        """
        message = (
            f"*일일 성과 리포트*\n"
            f"날짜: {stats.get('date', 'N/A')}\n\n"
            f"일일 PnL: `{stats.get('daily_pnl', 0):+.2f} USDT`\n"
            f"누적 수익률: `{stats.get('total_return', 0):+.2%}`\n"
            f"현재 MDD: `{stats.get('current_mdd', 0):.2%}`\n"
            f"오늘 거래 수: `{stats.get('trades_today', 0)}`"
        )
        return await self.send(message)

    async def send_alert(self, level: str, message: str) -> bool:
        """우선순위별 알림 전송.

        Args:
            level: 알림 레벨 ("critical" | "warning" | "info").
            message: 알림 내용.

        Returns:
            전송 성공 여부.
        """
        prefix_map = {
            "critical": "[긴급]",
            "warning": "[경고]",
            "info": "[정보]",
        }
        prefix = prefix_map.get(level, "[알림]")
        return await self.send(f"{prefix} {message}")

    def send_sync(self, message: str) -> bool:
        """동기 방식으로 텔레그램 메시지 전송 (이벤트 루프가 없는 경우).

        Args:
            message: 전송할 메시지.

        Returns:
            전송 성공 여부.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 이벤트 루프가 실행 중이면 새 스레드에서 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.send(message))
                    return future.result(timeout=10)
            else:
                return loop.run_until_complete(self.send(message))
        except RuntimeError:
            return asyncio.run(self.send(message))
