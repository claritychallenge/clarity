"""Tests for the data.HOA_tools_cec2 module."""
import numpy as np
import pytest
from numba.typed import List as TypedList  # pylint: disable=no-name-in-module

# HOARotator,; ambisonic_convolve,; binaural_mixdown,;;;
# compute_band_rotation,;; compute_rotation_matrix,;  dot,; dot,
from clarity.data.HOA_tools_cec2 import (
    HOARotator,
    P,
    U,
    V,
    W,
    ambisonic_convolve,
    binaural_mixdown,
    centred_element,
    compute_band_rotation,
    compute_rms,
    compute_rotation_matrix,
    compute_rotation_vector,
    compute_UVW_coefficients,
    convert_a_to_b_format,
    dB_to_gain,
    dot,
    equalise_rms_levels,
    rotation_control_vector,
    smoothstep,
)


def test_convert_a_to_b_format() -> None:
    """Test for convert_a_to_b_format() function."""
    np.random.seed(1234)
    signal = np.random.random((4, 100))
    result = convert_a_to_b_format(
        signal[0, :], signal[1, :], signal[2, :], signal[3, :]
    )
    assert result.shape == (4, 100)
    assert np.sum(np.abs(signal)) == pytest.approx(
        208.78666127520782, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_convert_a_to_b_format_index_error() -> None:
    """Test convert_a_to_b_format() raises IndexError if inputs not same length."""
    with pytest.raises(ValueError):
        convert_a_to_b_format(np.zeros(100), np.zeros(100), np.zeros(100), np.zeros(80))


def test_compute_rotation_matrix() -> None:
    """Test for compute_rotation_matrix() function."""
    order = 2
    rot_mat_dim = (order + 1) ** 2

    # Test for identity matrix
    rot_mat = compute_rotation_matrix(order, np.eye(3))
    assert rot_mat == pytest.approx(
        np.eye(rot_mat_dim), rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )

    # Test for 90 degree rotation about x-axis
    rot_mat = compute_rotation_matrix(
        2, np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    )
    assert rot_mat.shape == (rot_mat_dim, rot_mat_dim)
    # For this matrix, all values should be either 0, 1 or -1
    assert np.all(
        np.isclose(rot_mat, 0.0) + np.isclose(rot_mat, 1.0) + np.isclose(rot_mat, -1.0)
    )


def test_hoa_rotator_construction() -> None:
    """Test for HOARotator class construction."""

    # Testing 2nd order matrix with a 90 degree rotation step
    hoa_rotator = HOARotator(2, 90)
    rot_mat = hoa_rotator.rotmat
    assert rot_mat.shape == (4, 9, 9)
    rot90 = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert rot_mat[1, :, :] == pytest.approx(compute_rotation_matrix(2, rot90))
    # For multiples of 90 degrees all entries should be either 0, 1 or -1
    assert np.all(
        np.isclose(rot_mat, 0.0) + np.isclose(rot_mat, 1.0) + np.isclose(rot_mat, -1.0)
    )


def test_hoa_rotate_no_rotation() -> None:
    """Test the HOARotator rotate method"""

    # 1st order ambinsonic signal, ie 4 channels
    np.random.seed(1234)
    signal = np.random.random((100, 4))
    # equal length signal specify rotation for each sample
    rotations = np.zeros(100)
    # Apply the rotation
    hoa_rotator = HOARotator(1, 90)
    result = hoa_rotator.rotate(signal, rotations)
    assert result.shape == (100, 4)
    assert result == pytest.approx(
        signal, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "seed, expected",
    [
        (1234, 34.9827354617758),
        (1235, 37.63083581519224),
        (1236, 35.83739821236641),
    ],
)
def test_hoa_rotate(seed: int, expected: float) -> None:
    """Test the HOARotator rotate method"""

    # 1st order ambinsonic signal, ie 4 channels
    np.random.seed(seed)
    signal = np.random.random((20, 4))
    rotations = np.random.random(20)
    hoa_rotator = HOARotator(1, 90)
    result = hoa_rotator.rotate(signal, rotations)
    assert np.sum(np.abs(result)) == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "seed, row, col, expected",
    [
        (1234, 1, 1, 0.740996817336437),
        (1235, 45, 24, 0.2617612929455504),
        (1236, 18, 47, 0.2706397841499083),
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
        rotation_matrices=TypedList([r1_matrix, r2_matrix, r3_matrix]),
    ) == pytest.approx(expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance)


def test_P_index_error(make_random_matrix) -> None:
    """Test P() raises IndexError if invalid indices to r are provided."""
    random_matrix = make_random_matrix()
    rotation_matrices = TypedList([random_matrix])
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=1, order=1, rotation_matrices=rotation_matrices)
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=1, order=-1, rotation_matrices=rotation_matrices)
    with pytest.raises(IndexError):
        assert P(i=1, a=5, b=4, order=3, rotation_matrices=rotation_matrices)


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
        rotation_matrices=TypedList([r1_matrix, r2_matrix, r3_matrix]),
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
        rotation_matrices=TypedList([r1_matrix, r2_matrix, r3_matrix]),
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
        rotation_matrices=TypedList([r1_matrix, r2_matrix, r3_matrix]),
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
    assert compute_UVW_coefficients(degree, n, order) == pytest.approx(expected)


def test_compute_UVW_coefficients_zero_division_error() -> None:
    """Test computer_UVW_coefficients() raises ZeroDivisionError"""
    with pytest.raises(ZeroDivisionError):
        compute_UVW_coefficients(degree=-1, n=2, order=-2)


def test_compute_band_rotation() -> None:
    """Test for compute_band_rotation() function."""

    # This is not a very useful test as it is just one special case
    # but this function is tested implicitly in other tests
    order = 2
    rot_mats = TypedList([np.eye(1), np.eye(3), np.eye(5)])
    output = np.eye(9)
    new_output, new_rotation_matrices = compute_band_rotation(order, rot_mats, output)
    # output and rotation matrices should be unchanged
    assert new_output.shape == (9, 9)
    assert new_output == pytest.approx(
        np.eye(9), rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    for new_rot_mat, expected_rot_mat in zip(new_rotation_matrices, rot_mats):
        assert new_rot_mat == pytest.approx(
            expected_rot_mat, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
        )


@pytest.mark.parametrize(
    "A, B, expected",
    [
        (
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            np.asarray([[4.0, 3.0], [2.0, 1.0]]),
            np.asarray([[8.0, 5.0], [20.0, 13.0]]),
        ),
        (
            np.asarray([[5.0, 6.0], [7.0, 8.0]]),
            np.asarray([[8.0, 7.0], [6.0, 5.0]]),
            np.asarray([[76.0, 65.0], [104.0, 89.0]]),
        ),
    ],
)
def test_dot(A: np.ndarray, B: np.ndarray, expected: np.ndarray) -> None:
    """Test for dot() function."""
    result = dot(A, B)
    assert result == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


# TODO : need examples of hrir_metadata dictionary to be able to write a tests for this
def test_binaural_mixdown() -> None:
    """Test for binaural_mixdown() function."""
    np.random.seed(1234)
    hrir = {"M_data": np.random.random((100, 6, 2))}  # 6 random hrir filters
    hrir_filters = [0, 2, 3, 5]  # 4 channels selected of possible 6
    n_hrir_filters = len(hrir_filters)
    hrir_metadata = {
        "selected_channels": hrir_filters,
        "matrix": np.eye(n_hrir_filters),  # no rotation
    }
    signals = np.random.random((100, n_hrir_filters))

    result = binaural_mixdown(signals, hrir, hrir_metadata)
    assert result.shape == (199, 2)
    assert np.sum(np.abs(result)) == pytest.approx(
        19647.32682819124, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_ambisonic_convolve() -> None:
    """Test for ambisonic_convolve() function."""
    np.random.seed(1234)
    signal = np.random.randn(100)
    hoa_irs = np.random.random((100, 64))
    order = 2
    result = ambisonic_convolve(signal, hoa_irs, order=order)
    # channel
    assert result.shape == (199, (order + 1) ** 2)
    assert np.sum(np.abs(result)) == pytest.approx(
        3359.04412270433, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_ambisonic_convolve_error() -> None:
    """Test for ambisonic_convolve() function."""
    signal = np.random.randn(100)
    hoa_irs = np.random.random((100, 64))
    order = 10  # too high for number of hoa_ir channels
    with pytest.raises(ValueError):
        ambisonic_convolve(signal, hoa_irs, order=order)


@pytest.mark.parametrize(
    "array, axis, expected",
    [
        (
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            0,
            [
                np.sqrt(np.mean([1.0**2, 4.0**2, 7.0**2])),
                np.sqrt(np.mean([2.0**2, 5.0**2, 8.0**2])),
                np.sqrt(np.mean([3.0**2, 6.0**2, 9.0**2])),
            ],
        ),
        (
            np.asarray([[1.0, 2.0, 3.0], [4, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            1,
            [
                np.sqrt(np.mean([1.0**2, 2.0**2, 3.0**2])),
                np.sqrt(np.mean([4.0**2, 5.0**2, 6.0**2])),
                np.sqrt(np.mean([7.0**2, 8.0**2, 9.0**2])),
            ],
        ),
    ],
)
def test_compute_rms(array: np.ndarray, axis: int, expected: float) -> None:
    """Test for compute_rms() function along both axes."""
    result = compute_rms(input_signal=array, axis=axis)
    assert result == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_equalise_rms_levels() -> None:
    """Test for equalise_rms_levels() function."""
    np.random.seed(1234)
    signal_1 = np.random.random((100, 2)) * 100
    signal_2 = np.random.random((100, 2)) * 200
    signal_3 = np.random.random((100, 2)) * 200
    results = equalise_rms_levels(inputs=[signal_1, signal_2, signal_3])
    # Check all have same level after equalisation
    rms_1 = compute_rms(results[0], axis=0)
    rms_2 = compute_rms(results[1], axis=0)
    rms_3 = compute_rms(results[2], axis=0)
    assert rms_1[0] == pytest.approx(
        rms_3[0], rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert rms_2[0] == pytest.approx(
        rms_3[0], rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "db, gain", [(10, 3.1622776601683795), (20, 10.0), (0.045, 1.0051942600951387)]
)
def test_dB_to_gain(db: float, gain: float) -> None:
    """Test for test_dB_to_gain function."""
    result = dB_to_gain(db)
    assert result == pytest.approx(
        gain, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


@pytest.mark.parametrize(
    "x, x_min, x_max, N, expected",
    [
        (
            np.asarray([100, 200, 300, 400]),
            10.0,
            300.0,
            1,
            np.asarray([0.22916068719504698, 0.725286, 1.0, 1.0]),
        ),
        (
            np.asarray([10, 200, 300, 4000]),
            10.0,
            300.0,
            1,
            np.asarray([0.0, 0.725286, 1.0, 1.0]),
        ),
        (
            np.asarray([10, -200, 300, 4000]),
            -200.0,
            300.0,
            1,
            np.asarray([0.381024, 0.0, 1.0, 1.0]),
        ),
        (
            np.asarray([10, -50, 300, 4000]),
            -200.0,
            300.0,
            1,
            np.asarray([0.381024, 0.216, 1.0, 1.0]),
        ),
    ],
)
def test_smoothstep(
    x: np.ndarray, x_min: float, x_max: float, N: int, expected
) -> None:
    """Test for smoothstep() function."""
    result = smoothstep(x, x_min, x_max, N)
    assert result == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


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

    result = rotation_control_vector(array_length, start_idx, end_idx)
    assert result == pytest.approx(
        expected, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
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
