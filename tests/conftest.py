"""Fixtures for testing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

SEED = 564687
rng = np.random.default_rng(SEED)


@pytest.fixture
def make_random_matrix():
    """Generate a random matrix for use in tests.

    The fixture returns a function that can be called to generate a random matrix
    >>> def my_test(random_matrix):
    >>>     matrix = random_matrix(seed=1234)
    or use the global seed
    >>>     matrix = random_matrix()
    """

    def _random_matrix(seed: int | None = None, size=(100, 100)) -> np.ndarray:
        if seed is not None:
            # Seed is supplied so use a generator with that seed...
            rng_to_use = np.random.default_rng(seed)
        else:
            # ... else use the global generator
            rng_to_use = rng
        return np.asarray(rng_to_use.random(size))

    return _random_matrix


def pytest_configure() -> None:
    """Configure custom variables for pytest.

    **NB**: pytest automatically calls this hook when the conftest is loaded.
    """
    pytest.abs_tolerance = 1e-7
    pytest.rel_tolerance = 1e-7
