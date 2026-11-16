import pytest

from asof.providers.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_allows_up_to_limit_without_sleeping():
    clock = FakeClock()
    rl = RateLimiter(3, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        rl.acquire()
    assert clock.slept == []


def test_waits_for_oldest_call_to_leave_window():
    clock = FakeClock()
    rl = RateLimiter(2, clock=clock, sleep=clock.sleep)
    rl.acquire()
    clock.t = 10.0
    rl.acquire()
    rl.acquire()  # third call must wait until t=60 when the first call expires
    assert clock.slept == [pytest.approx(50.0)]
    assert clock.t == pytest.approx(60.0)


def test_rejects_invalid_limit():
    with pytest.raises(ValueError):
        RateLimiter(0)
