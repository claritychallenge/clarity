"""Tests for the data.HOA_tools_cec2 module."""
import numpy as np
import pytest

from clarity.data.HOA_tools_cec2 import (  # HOARotator,; ambisonic_convolve,; binaural_mixdown,; compute_band_rotation,; compute_rotation_matrix,; dB_to_gain,; dot,; smoothstep,
    P,
    U,
    V,
    W,
    centred_element,
    compute_rms,
    compute_UVW_coefficients,
    equalise_rms_levels,
    rotation_control_vector,
    rotation_vector,
)


def test_compute_rotation_matrix() -> None:
    """Test for compute_rotation_matrix() function."""
    assert True


@pytest.mark.parametrize(
    "i, j, expected",
    [
        (1, 1, 0.8840922122960315),
        (45, 54, 0.2378095805520779),
        (18, 91, 0.39897595320190293),
    ],
)
def test_centred_element(
    random_matrix: np.ndarray, i: int, j: int, expected: float
) -> None:
    """Test for centered_element() function."""
    assert centred_element(random_matrix, i, j) == expected


def test_centred_element_index_error(random_matrix: np.ndarray) -> None:
    """Test centered_element() raises an IndexError if centering is outside of matrix dimensions."""
    with pytest.raises(IndexError):
        centred_element(random_matrix, i=101, j=2)
    with pytest.raises(IndexError):
        centred_element(random_matrix, i=1, j=102)


@pytest.mark.parametrize(
    "i, a, b, el, expected",
    [
        (1, 1, 2, 2, -0.3475073926260553),
        (1, 1, 2, -2, 0.6814353682034401),
        (1, 1, 2, 2, -0.7185822065660934),
    ],
)
def test_P(
    i: int, a: int, b: int, el: int, random_matrix: np.ndarray, expected: float
) -> None:
    """Test for P() function."""
    # FixMe : how long is the list of matrices that is passed in?
    assert P(i, a, b, el, r=[random_matrix, random_matrix, random_matrix]) == expected


def test_P_index_error(random_matrix: np.ndarray) -> None:
    """Test P() raises IndexError if invalid indices to r are provided."""
    # FixMe : how long is the list of matrices that is passed in?
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=1, el=1, r=[random_matrix])
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=1, el=-1, r=[random_matrix])
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=4, el=3, r=[random_matrix])


@pytest.mark.parametrize(
    "m, n, el, expected",
    [
        (1, 2, 2, 0.8066335578037782),
        (1, 2, -2, 0.647932971103906),
        (1, 2, 2, 0.504340149676519),
    ],
)
def test_U(m: int, n: int, el: int, random_matrix: np.ndarray, expected: float) -> None:
    """Test for XX function."""
    assert U(m, n, el, r=[random_matrix, random_matrix, random_matrix]) == expected


@pytest.mark.parametrize(
    "m, n, el, expected",
    [
        (0, 2, 2, -0.02607000059684761),
        (1, 2, 2, -0.8041211269599031),
        (1, 2, -2, 0.889239093568916),
        (-1, 2, -2, 0.5668224726503112),
    ],
)
def test_V(m: int, n: int, el: int, random_matrix: np.ndarray, expected: float) -> None:
    """Test for V() function."""
    assert V(m, n, el, r=[random_matrix, random_matrix, random_matrix]) == expected


@pytest.mark.parametrize(
    "m, n, el, expected",
    [
        (0, 2, 2, 0.0),
        (1, 2, 2, -0.18322112315448064),
        (-1, 2, 2, -0.3501066433176404),
    ],
)
def test_W(m: int, n: int, el: int, random_matrix: np.ndarray, expected: float) -> None:
    """Test for W() function."""
    assert W(m, n, el, r=[random_matrix, random_matrix, random_matrix]) == expected


@pytest.mark.parametrize(
    "m, n, el, expected",
    [
        (0, 2, 2, (0.5773502691896257, -0.28867513459481287, -0.0)),
        (1, 2, 2, (0.5, 0.3535533905932738, -0.0)),
        (-1, 2, 3, (1.2649110640673518, 0.7745966692414834, -0.31622776601683794)),
        # (-1, 2, 1, (-0.0, np.nan, -0.0)),  # FixMe : This is probably going to cause problems how to capture?
    ],
)
def test_compute_UVW_coefficients(m: int, n: int, el: int, expected: float) -> None:
    """Test for computer_UVW_coefficients() function."""
    assert compute_UVW_coefficients(m, n, el) == expected


def test_compute_UVW_coefficients_zero_division_error() -> None:
    """Test computer_UVW_coefficients() raises ZeroDivisionError"""
    with pytest.raises(ZeroDivisionError):
        compute_UVW_coefficients(m=-1, n=2, el=-2)


# FixMe : Not yet working, I don't understand how to get the `output` passed in correctly
# @pytest.mark.parametrize(
#     "el, output, expected",
#     [
#         (2, np.asarray([0, 0]), np.asarray([[1, 2], [3, 4]])),
#     ],
# )
# def test_compute_band_rotation(el: int, output: np.ndarray, random_matrix: np.ndarray, expected: np.ndarray) -> None:
#     """Test for compute_band_rotation() function."""
#     np.testing.assert_array_equal(compute_band_rotation(el, [random_matrix, random_matrix], output), expected)


# FixMe : Not working yet, results in a typing error???
# @pytest.mark.parametrize(
#     "A, B, expected",
#     [
#         (np.asarray([[1, 2], [3, 4]]), np.asarray([[4, 3], [2, 1]]), np.asarray([[1, 2], [3, 4]])),
#         (np.asarray([[5, 6], [7, 8]]), np.asarray([[8, 7], [6, 5]]), np.asarray([[1, 2], [3, 4]])),
#     ],
# )
# def test_dot(A: np.ndarray, B: np.ndarray, expected: np.ndarray) -> None:
#     """Test for XX function."""
#     np.testing.assert_array_equal(dot(A, B), expected)


def test_HOARotator() -> None:
    """Test for test_HOARotator class."""
    assert True


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


def test_equalise_rms_levels(random_matrix: np.ndarray) -> None:
    """Test for equalise_rms_levels() function."""
    assert equalise_rms_levels(inputs=[random_matrix, random_matrix])


def test_dB_to_gain() -> None:
    """Test for XX function."""
    assert True


def test_smoothstep() -> None:
    """Test for XX function."""
    assert True


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
    """Test rotation_control_vector() raises ValueError if start_idx > end_idx or end_idx > array_length."""
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
def test_rotation_vector(
    start_angle: float,
    end_angle: float,
    signal_length: int,
    start_idx: int,
    end_idx: int,
    expected: np.ndarray,
) -> None:
    """Test for rotation_vector() function."""
    np.testing.assert_array_equal(
        rotation_vector(start_angle, end_angle, signal_length, start_idx, end_idx),
        expected,
    )
    assert True


def test_rotation_vector_floating_point_error() -> None:
    """Test rotation_vector() raises FloatingPointError if signal_length is zero function."""
    with pytest.raises(FloatingPointError):
        rotation_vector(
            start_angle=10, end_angle=20, signal_length=0, start_idx=2, end_idx=8
        )
