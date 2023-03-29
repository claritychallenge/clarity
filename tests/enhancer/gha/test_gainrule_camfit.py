"""Tests for enhancer.gha.gainrule_camfit module"""
import numpy as np  # noqa: I001
import pytest

from clarity.enhancer.gha.gainrule_camfit import compute_proportion_overlap, isothr

# freq_inter_sh,; gains,; gainrule_camfit_linear,; gainrule_camfit_compr


@pytest.mark.parametrize(
    "a1, a2, b1, b2, expected",
    [
        (0.0, 1.0, 0.0, 1.0, 1.0),
        (0.0, 1.0, 2.0, 3.0, 0.0),
        (0.0, 10.0, 5.0, 15.0, 0.5),
        (0.0, 10.0, 5.0, 10.0, 1.0),
        (10.0, 20.0, 5.0, 15.0, 0.5),
        (10.0, 20.0, 19.0, 29.0, 0.1),
        (2.0, 4.0, 0.0, 10.0, 0.2),
    ],
)
def test_compute_proportion_overlap(a1, a2, b1, b2, expected):
    """test that the proportion overlap is computed correctly"""
    assert compute_proportion_overlap(a1, a2, b1, b2) == pytest.approx(expected)


def test_isothr():
    """test that the isothr is computed correctly"""
    inputs_1 = [30, 40, 50, 60, 70, 80]
    inputs_2 = [35, 45, 50, 60, 70, 80]  # <- same as inputs_1 as all rounded up to 50
    inputs_3 = [35, 55, 50, 60, 70, 80]
    outputs_1 = isothr(inputs_1)
    outputs_2 = isothr(inputs_2)
    outputs_3 = isothr(inputs_3)

    assert len(inputs_1) == len(outputs_1)
    assert np.all(outputs_1 == outputs_2)  # they should be the same
    assert not np.all(outputs_1 == outputs_3)  # they should be the different

    assert np.sum(outputs_1) == pytest.approx(237.52941176470588)
    assert np.sum(outputs_2) == pytest.approx(237.52941176470588)
    assert np.sum(outputs_3) == pytest.approx(235.02941176470588)


@pytest.mark.skip(reason="not implemented")
def test_freq_inter_sh():
    """test that the freq_inter_sh is computed correctly"""


@pytest.mark.skip(reason="not implemented")
def test_gains():
    """test that the gains are computed correctly"""


@pytest.mark.skip(reason="not implemented")
def test_gainrule_camfit_linear():
    """test that the linear gain rule runs correctly"""


@pytest.mark.skip(reason="not implemented")
def test_gainrule_camfit_compr():
    """test that the compr gain rule runs correctly"""
