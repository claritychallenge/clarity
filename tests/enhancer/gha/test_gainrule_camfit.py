"""Tests for enhancer.gha.gainrule_camfit module"""
import numpy as np
import pytest

from clarity.enhancer.gha.gainrule_camfit import (
    compute_proportion_overlap,
    gainrule_camfit_compr,
    gainrule_camfit_linear,
    gains,
    isothr,
)
from clarity.utils.audiogram import Audiogram


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
    assert compute_proportion_overlap(a1, a2, b1, b2) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


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

    assert np.sum(outputs_1) == pytest.approx(
        237.52941176470588, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(outputs_2) == pytest.approx(
        237.52941176470588, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(outputs_3) == pytest.approx(
        235.02941176470588, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_gains():
    """test that the gains are computed correctly"""
    compr_thr_inputs = [30, 40, 50]
    compr_thr_gains = [35, 35, 50]
    compressions_ratios = [1.0, 2.0, 3.0]
    levels = [0.0, 10.0, 100.0]
    expected_outputs = np.array(
        [[35.0, 55.0, 83.33333333], [35.0, 50.0, 76.66666667], [35.0, 5.0, 16.66666667]]
    )
    n_gains = len(compr_thr_inputs)
    uncorrected_gains = gains(
        compr_thr_inputs=compr_thr_inputs,
        compr_thr_gains=compr_thr_gains,
        compression_ratios=compressions_ratios,
        levels=levels,
    )
    assert uncorrected_gains.shape == (n_gains, n_gains)
    assert uncorrected_gains == pytest.approx(
        expected_outputs, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_gainrule_camfit_linear():
    """test that the linear gain rule runs correctly"""
    audiogram = Audiogram(
        levels=np.array([30, 40, 50, 60, 70, 80]),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 8000]),
    )

    sFitmodel = {
        "frequencies": [250, 500, 1000, 2000, 4000, 8000],
        "levels": [30, 40, 50, 60, 70, 80],
    }
    sGt, noisegate_level, noisegate_slope, insertion_gains = gainrule_camfit_linear(
        audiogram_left=audiogram,
        audiogram_right=audiogram,
        sFitmodel=sFitmodel,
        noisegatelevels=45,
        noisegateslope=1,
        max_output_level=100,
    )

    assert sGt.shape == (6, 6, 2)
    assert noisegate_level.shape == (6, 2)
    assert noisegate_slope.shape == (6, 2)
    assert insertion_gains.shape == (6, 2)

    assert sGt.sum() == pytest.approx(
        1589.2, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert noisegate_level.sum() == pytest.approx(
        540.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert noisegate_slope.sum() == pytest.approx(
        12.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert insertion_gains.sum() == pytest.approx(
        284.8, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_gainrule_camfit_compr():
    """test that the compr gain rule runs correctly"""
    audiogram_left = Audiogram(
        levels=np.array([30, 40, 50, 60, 70, 80]),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 8000]),
    )
    audiogram_right = Audiogram(
        levels=np.array([30, 40, 50, 60, 70, 80]),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 8000]),
    )

    sFitmodel = {
        "frequencies": [250, 500, 1000, 2000, 4000, 8000],
        "levels": [30, 40, 50, 60, 70, 80],
        "edge_frequencies": [250, 8000],
    }
    sGt, noisegate_levels, noisegate_slope = gainrule_camfit_compr(
        audiogram_left=audiogram_left,
        audiogram_right=audiogram_right,
        sFitmodel=sFitmodel,
        noisegatelevels=45,
        noisegateslope=1,
        level=0,
        max_output_level=100,
    )

    assert sGt.shape == (6, 6, 2)
    assert noisegate_levels.shape == (6, 2)
    assert noisegate_slope.shape == (6, 2)

    assert sGt.sum() == pytest.approx(
        2140.1762291737905, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert noisegate_levels.sum() == pytest.approx(
        540.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert noisegate_slope.sum() == pytest.approx(
        12.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "ng_levels, ng_slope, level, max_output_level, out_sgt, out_ng_level, out_ng_slope",
    [
        (45, 1, 0, 100, 2140.1762291737905, 540.0, 12.0),
        (55, 1, 0, 100, 2140.1762291737905, 660.0, 12.0),
        (45, 2, 0, 100, 2140.1762291737905, 540.0, 24.0),
        # In next example level is changed to 30, the level must be in sFitmodel levels
        (45, 1, 30, 100, 2140.1762291737905, 540.0, 12.0),
        (45, 1, 0, 120, 2383.502544963264, 540.0, 12.0),
    ],
)
def test_gainrule_camfit_compr_varying_params(
    ng_levels, ng_slope, level, max_output_level, out_sgt, out_ng_level, out_ng_slope
):
    """test that the compr gain rule runs correctly for different parameters"""
    audiogram = Audiogram(
        levels=np.array([30, 40, 50, 60, 70, 80]),
        frequencies=np.array([250, 500, 1000, 2000, 4000, 8000]),
    )

    sFitmodel = {
        "frequencies": [250, 500, 1000, 2000, 4000, 8000],
        "levels": [30, 40, 50, 60, 70, 80],
        "edge_frequencies": [250, 8000],
    }
    sGt, noisegate_levels, noisegate_slope = gainrule_camfit_compr(
        audiogram_left=audiogram,
        audiogram_right=audiogram,
        sFitmodel=sFitmodel,
        noisegatelevels=ng_levels,
        noisegateslope=ng_slope,
        level=level,
        max_output_level=max_output_level,
    )

    assert sGt.sum() == pytest.approx(
        out_sgt, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert noisegate_levels.sum() == pytest.approx(
        out_ng_level, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert noisegate_slope.sum() == pytest.approx(
        out_ng_slope, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
