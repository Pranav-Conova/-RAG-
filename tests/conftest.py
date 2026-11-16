import pytest

from asof.config import Settings
from asof.providers.ratelimit import RateLimiter


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]  # ignore any local .env in unit tests
        nvidia_api_key="nvapi-test",
        embed_dim=4,
    )


@pytest.fixture
def no_limit() -> RateLimiter:
    return RateLimiter(10_000)
