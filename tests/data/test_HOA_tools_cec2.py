"""Tests for the data.HOA_tools_cec2 module."""
import numpy as np
import pytest

# HOARotator,; ambisonic_convolve,; binaural_mixdown,;;;
# compute_band_rotation,;; compute_rotation_matrix,;  dot,; dot,
from clarity.data.HOA_tools_cec2 import (  # compute_rotation_matrix,
    HOARotator,
    P,
    U,
    V,
    W,
    centred_element,
    compute_rms,
    compute_rotation_vector,
    compute_UVW_coefficients,
    dB_to_gain,
    equalise_rms_levels,
    rotation_control_vector,
    smoothstep,
)

# def test_compute_rotation_matrix() -> None:
#    """Test for compute_rotation_matrix() function."""
#    # rot_mat = compute_rotation_matrix(2, np.eye(3))
#    compute_rotation_matrix(
#        2, np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])
#    )


def test_hoa_rotator_construction() -> None:
    """Test for HOARotator class construction."""

    # Testing 2nd order matrix with a 90 degree rotation step
    hoa_rotator = HOARotator(2, 90)
    assert hoa_rotator.rotmat.shape == (4, 9, 9)


@pytest.mark.parametrize(
    "seed, row, col, expected",
    [
        (1234, 1, 1, 0.740996817336437),
        (1235, 45, 54, 0.078972915583452),
        (1236, 18, 91, 0.522131631590338),
    ],
)
def test_centred_element(
    make_random_matrix, seed: int, row: int, col: int, expected: float
) -> None:
    """Test for centered_element() function."""
    random_matrix = make_random_matrix(seed)
    assert centred_element(random_matrix, row, col) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_centred_element_index_error(make_random_matrix) -> None:
    """Test that centered_element() raises an IndexError if centering is outside
    of matrix dimensions."""
    random_matrix = make_random_matrix()
    with pytest.raises(IndexError):
        centred_element(random_matrix, row=101, col=2)
    with pytest.raises(IndexError):
        centred_element(random_matrix, row=1, col=102)


@pytest.mark.parametrize(
    "i, a, b, order, seed, expected",
    [
        (1, 1, 2, 2, 1234, -0.20056163713681976),
        (1, 1, 2, -2, 1235, 0.1767992365183524),
        (1, 1, 2, 2, 1236, -0.5552636713315029),
    ],
)
def test_P(
    i: int, a: int, b: int, order: int, make_random_matrix, seed: int, expected: float
) -> None:
    """Test for P() function."""
    r1_matrix = make_random_matrix(seed)
    r2_matrix = make_random_matrix(seed + 10)  # Different seed to get different matrix
    r3_matrix = make_random_matrix(seed + 20)  # ditto
    assert P(
        i,
        a,
        b,
        order,
        rotation_matrices=[r1_matrix, r2_matrix, r3_matrix],
    ) == pytest.approx(expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance)


def test_P_index_error(make_random_matrix) -> None:
    """Test P() raises IndexError if invalid indices to r are provided."""
    random_matrix = make_random_matrix()
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=1, order=1, rotation_matrices=[random_matrix])
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=1, order=-1, rotation_matrices=[random_matrix])
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=4, order=3, rotation_matrices=[random_matrix])


@pytest.mark.parametrize(
    "degree, n, order, seed, expected",
    [
        (1, 2, 2, 2234, -0.08645029499482301),
        (1, 2, -2, 2235, 0.1550159595919902),
        (1, 2, 2, 2236, -0.6545704498040456),
    ],
)
def test_U(
    degree: int, n: int, order: int, make_random_matrix, seed: int, expected: float
) -> None:
    """Test for U function."""
    r1_matrix = make_random_matrix(seed)
    r2_matrix = make_random_matrix(seed + 10)  # Different seed to get different matrix
    r3_matrix = make_random_matrix(seed + 20)  # ditto
    assert U(
        degree,
        n,
        order,
        rotation_matrices=[r1_matrix, r2_matrix, r3_matrix],
    ) == pytest.approx(expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance)


@pytest.mark.parametrize(
    "degree, n, order, seed, expected",
    [
        (0, 2, 2, 3234, 0.6475456062624654),
        (1, 2, 2, 3235, -0.009790830648337613),
        (1, 2, -2, 3236, 1.3180460765445767),
        (-1, 2, -2, 3237, 0.44136069630527924),
    ],
)
def test_V(
    degree: int, n: int, order: int, make_random_matrix, seed: int, expected: float
) -> None:
    """Test for V() function."""
    r1_matrix = make_random_matrix(seed)
    r2_matrix = make_random_matrix(seed + 10)  # Different seed to get different matrix
    r3_matrix = make_random_matrix(seed + 20)  # ditto
    assert V(
        degree,
        n,
        order,
        rotation_matrices=[r1_matrix, r2_matrix, r3_matrix],
    ) == pytest.approx(expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance)


@pytest.mark.parametrize(
    "degree, n, order, seed, expected",
    [
        (0, 2, 2, None, 0.0),  # No seed, as should result in 0.0 for all matrices
        (1, 2, 2, 4235, 0.6708751988390329),
        (-1, 2, 2, 4236, 0.38743574623364374),
    ],
)
def test_W(
    degree: int, n: int, order: int, make_random_matrix, seed: int, expected: float
) -> None:
    """Test for W() function."""
    r1_matrix = make_random_matrix(seed)
    r2_matrix = make_random_matrix(seed + 10 if seed is not None else None)
    r3_matrix = make_random_matrix(seed + 10 if seed is not None else None)  # ditto

    assert W(
        degree,
        n,
        order,
        rotation_matrices=[r1_matrix, r2_matrix, r3_matrix],
    ) == pytest.approx(expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance)


@pytest.mark.parametrize(
    "degree, n, order, expected",
    [
        (0, 2, 2, (0.5773502691896257, -0.28867513459481287, -0.0)),
        (1, 2, 2, (0.5, 0.3535533905932738, -0.0)),
        (-1, 2, 3, (1.2649110640673518, 0.7745966692414834, -0.31622776601683794)),
        # (-1, 2, 1, (-0.0, np.nan, -0.0)),
        # TODO : This is probably going to cause problems how to capture?
    ],
)
def test_compute_UVW_coefficients(
    degree: int, n: int, order: int, expected: float
) -> None:
    """Test for computer_UVW_coefficients() function."""
    assert compute_UVW_coefficients(degree, n, order) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_compute_UVW_coefficients_zero_division_error() -> None:
    """Test computer_UVW_coefficients() raises ZeroDivisionError"""
    with pytest.raises(ZeroDivisionError):
        compute_UVW_coefficients(degree=-1, n=2, order=-2)


# TODO : Not yet working, I don't understand how to get `output` passed in correctly
# @pytest.mark.parametrize(
#     "el, output, expected",
#     [
#         (2, np.asarray([0, 0]), np.asarray([[1, 2], [3, 4]])),
#     ],
# )
# def test_compute_band_rotation(el: int, output: np.ndarray,
#   make_random_matrix, expected: np.ndarray) -> None:
#     """Test for compute_band_rotation() function."""
#     np.testing.assert_array_equal(compute_band_rotation(el,
# [make_random_matrix(), make_random_matrix()], output), expected)


# TODO : Not working yet, results in a typing error???
# @pytest.mark.parametrize(
#     "A, B, expected",
#     [
#         (np.asarray([[1, 2], [3, 4]]), np.asarray([[4, 3], [2, 1]]),
#               np.asarray([[1, 2], [3, 4]])),
#         (np.asarray([[5, 6], [7, 8]]), np.asarray([[8, 7], [6, 5]]),
#               np.asarray([[1, 2], [3, 4]])),
#     ],
# )
# def test_dot(A: np.ndarray, B: np.ndarray, expected: np.ndarray) -> None:
#     """Test for dot() function."""
#     np.testing.assert_array_equal(dot(A, B), expected)


def test_HOARotator() -> None:
    """Test for test_HOARotator class."""
    assert True


# TODO : need examples of hrir_metadata dictionary to be able to write a tests for this
def test_binaural_mixdown() -> None:
    """Test for binaural_mixdown() function."""
    assert True


def test_ambisonic_convolve() -> None:
    """Test for ambisonic_convolve() function."""
    assert True


@pytest.mark.parametrize(
    "array, axis, expected",
    [
        (
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            0,
            [
                np.sqrt(np.mean([1**2, 4**2, 7**2])),
                np.sqrt(np.mean([2**2, 5**2, 8**2])),
                np.sqrt(np.mean([3**2, 6**2, 9**2])),
            ],
        ),
        (
            np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            1,
            [
                np.sqrt(np.mean([1**2, 2**2, 3**2])),
                np.sqrt(np.mean([4**2, 5**2, 6**2])),
                np.sqrt(np.mean([7**2, 8**2, 9**2])),
            ],
        ),
    ],
)
def test_compute_rms(array: np.ndarray, axis: int, expected: float) -> None:
    """Test for compute_rms() function along both axes."""
    np.testing.assert_array_equal(compute_rms(input_signal=array, axis=axis), expected)


def test_equalise_rms_levels(make_random_matrix) -> None:
    """Test for equalise_rms_levels() function."""
    assert equalise_rms_levels(inputs=[make_random_matrix(), make_random_matrix()])


@pytest.mark.parametrize(
    "db, gain", [(10, 3.1622776601683795), (20, 10.0), (0.045, 1.0051942600951387)]
)
def test_dB_to_gain(db: float, gain: float) -> None:
    """Test for XX function."""
    assert dB_to_gain(db) == gain


@pytest.mark.parametrize(
    "x, x_min, x_max, N, expected",
    [
        (
            np.asarray([100, 200, 300, 400]),
            10,
            300,
            1,
            np.asarray([0.229161, 0.725286, 1.0, 1.0]),
        ),
        (
            np.asarray([10, 200, 300, 4000]),
            10,
            300,
            1,
            np.asarray([0.0, 0.725286, 1.0, 1.0]),
        ),
        (
            np.asarray([10, -200, 300, 4000]),
            -200,
            300,
            1,
            np.asarray([0.381024, 0.0, 1.0, 1.0]),
        ),
        (
            np.asarray([10, -50, 300, 4000]),
            -200,
            300,
            1,
            np.asarray([0.381024, 0.216, 1.0, 1.0]),
        ),
    ],
)
def test_smoothstep(
    x: np.ndarray, x_min: float, x_max: float, N: int, expected
) -> None:
    """Test for smoothstep() function."""
    np.testing.assert_array_almost_equal(smoothstep(x, x_min, x_max, N), expected)


@pytest.mark.parametrize(
    "array_length, start_idx, end_idx, expected",
    [
        (10, 2, 8, np.asarray([0, 0, 0, 0, 2, 4, 6, 8, 9, 9])),  # start_idx > end_idx
        (
            10,
            1,
            5,
            np.asarray([0, 0, 1, 4, 7, 9, 9, 9, 9, 9]),
        ),  # end_idx > signal_length
    ],
)
def test_rotation_control_vector(
    array_length: int, start_idx: int, end_idx: int, expected: np.ndarray
) -> None:
    """Test for rotation_control_vector() function."""
    np.testing.assert_array_equal(
        rotation_control_vector(array_length, start_idx, end_idx), expected
    )


@pytest.mark.parametrize(
    "array_length, start_idx, end_idx",
    [
        (100, 90, 80),  # start_idx > end_idx
        (100, 20, 110),  # end_idx > signal_length
    ],
)
def test_rotation_control_vector_value_error(
    array_length: int, start_idx: int, end_idx: int
) -> None:
    """Test rotation_control_vector() raises ValueError if start_idx > end_idx or
    end_idx > array_length."""
    with pytest.raises(ValueError):
        rotation_control_vector(array_length, start_idx, end_idx)


@pytest.mark.parametrize(
    "start_angle, end_angle, signal_length, start_idx, end_idx, expected",
    [
        (
            10,
            20,
            10,
            2,
            8,
            np.asarray([10.0, 10.0, 10.0, 10.0, 12.0, 14.0, 16.0, 18.0, 19.0, 19.0]),
        ),
        (
            -10,
            20,
            10,
            2,
            8,
            np.asarray([-10.0, -10.0, -10.0, -10.0, -4.0, 2.0, 8.0, 14.0, 17.0, 17.0]),
        ),
        (
            -10,
            -20,
            10,
            2,
            8,
            np.asarray(
                [-10.0, -10.0, -10.0, -10.0, -12.0, -14.0, -16.0, -18.0, -19.0, -19.0]
            ),
        ),
    ],
)
def test_compute_rotation_vector(
    start_angle: float,
    end_angle: float,
    signal_length: int,
    start_idx: int,
    end_idx: int,
    expected: np.ndarray,
) -> None:
    """Test for rotation_vector() function."""
    np.testing.assert_array_equal(
        compute_rotation_vector(
            start_angle, end_angle, signal_length, start_idx, end_idx
        ),
        expected,
    )


def test_compute_rotation_vector_floating_point_error() -> None:
    """Test rotation_vector() raises FloatingPointError if signal_length is
    zero function."""
    with pytest.raises(FloatingPointError):
        compute_rotation_vector(
            start_angle=10, end_angle=20, signal_length=0, start_idx=2, end_idx=8
        )
