"""Fixtures for testing."""
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"

SEED = 564687
rng = np.random.default_rng(SEED)


@pytest.fixture
def random_matrix() -> np.ndarray:
    """Generate a random matrix for use in tests."""
    return np.asarray(rng.random((100, 100)))


def pytest_configure() -> None:
    """Configure custom variables for pytest"""
    pytest.abs_tolerance = 1e-7
    pytest.rel_tolerance = 1e-7
