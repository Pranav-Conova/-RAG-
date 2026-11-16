import threading
import time
from collections import deque
from collections.abc import Callable
from functools import lru_cache


class RateLimiter:
    """Sliding-window limiter: at most `per_minute` calls in any 60 s window.

    The NVIDIA free tier returns 429 above its limit, so we throttle client-side
    and keep 429 retries for the rare overshoot.
    """

    def __init__(
        self,
        per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be >= 1")
        self.per_minute = per_minute
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.per_minute:
                    self._calls.append(now)
                    return
                self._sleep(60.0 - (now - self._calls[0]))


@lru_cache
def shared_limiter(per_minute: int) -> RateLimiter:
    """One limiter per process, shared by LLM, embedding and rerank calls."""
    return RateLimiter(per_minute)
