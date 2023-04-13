"""Tools to support higher order ambisonic processing."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numba import njit  # type: ignore # <-- silence mypy no attribute error
from numba.typed import List as TypedList  # pylint: disable=no-name-in-module
from numpy import ndarray
from scipy.signal import convolve
from scipy.spatial.transform import Rotation as R
from scipy.special import comb

logger = logging.getLogger(__name__)


def convert_a_to_b_format(
    front_left_up: np.ndarray,
    front_right_down: np.ndarray,
    back_left_down: np.ndarray,
    back_right_up: np.ndarray,
):
    """Converts 1st order A format audio into 1st order B format.
    For more information on ambisonic formats see Gerzon, Michael A.
    “Ambisonics. Part two: Studio techniques.” (1975).

    Args:
        front_left_up (np.ndarray): Front-left-up audio
        front_right_down (np.ndarray): Front-right-down audio
        back_left_down (np.ndarray): Back-left-down audio
        back_right_up (np.ndarray): Back-right-up audio

    Raises:
        TypeError: input must be numpy array
        ValueError: all inputs must have same dimensions

    Returns:
        nd.array: 4xN array containing B-format audio. indexed w,x,y,z
    """

    shapes = [
        front_left_up.shape,
        front_right_down.shape,
        back_left_down.shape,
        back_right_up.shape,
    ]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All inputs need to have same dimensions")

    w = 0.5 * sum([front_left_up, front_right_down, back_left_down, back_right_up])
    x = 0.5 * (front_left_up - back_left_down) + (front_right_down - back_right_up)
    y = 0.5 * (front_left_up - back_right_up) - (front_right_down - back_left_down)
    z = 0.5 * (front_left_up - back_left_down) + (back_right_up - front_right_down)

    return np.stack([w, x, y, z])


# Code for generation ambisonic rotation matrices
@njit
def compute_rotation_matrix(n: int, foa_rotmat: ndarray) -> ndarray:
    """Generate a rotation matrix to rotate HOA soundfield.

    Based on [1]_ and [2]_. Operates on HOA of a given order rotates by azimuth theta
    and elevation phi.

    Args:
        order (int): order of ambisonic soundfield
        foa_rotmat (arraylike): rotation matrix to expand

    Returns:
        np.ndarray: HOA rotation matrix

    References:
    .. [1] Ivanic J, Ruedenberg K (1996) Rotation Matrices for Real Spherical Harmonics.
           Direct Determination J. Phys. Chem. 1996, 100(15):6342–6347. Available at
           <https://pubs.acs.org/doi/10.1021/jp953350u> and
           <https://doi.org/10.1021/JP9833350>

    .. [2] Gorzel M, Allen A, Kelly I, Kammerl J, Gungormusler, A, Yeh H, Boland F
           Efficient Encoding and Decoding of Binaural Sound with Resonance Audio
           In Proc. of the AES International Conference on Immersive and Interactive
           Audio, March 2019. Available at
           <https://www.aes.org/e-lib/browse.cfm?elib=20446>
    """
    m = (n + 1) ** 2
    # construct rotamat for given order
    n_vector = np.arange(m)
    rot_mat = np.eye(m)
    rot_mat[1:4, 1:4] = foa_rotmat
    upper_index = np.cumsum(2 * n_vector + 1)
    lower_index = np.roll(upper_index, 1)
    lower_index[0] = 0

    sub_matrices = [np.eye(i * 2 + 1) for i in np.arange(n + 1)]
    sub_matrices[1] = foa_rotmat
    typed_sub_matrices = TypedList()
    for x in sub_matrices:
        typed_sub_matrices.append(x)

    if n > 1:
        for i in np.arange(2, n + 1):
            rot_mat, typed_sub_matrices = compute_band_rotation(
                i, typed_sub_matrices, rot_mat
            )

    return rot_mat


@njit
def centred_element(reference: np.ndarray, row: int, col: int):
    """Get value from centered element indexing.

    Args:
        reference (matrix): reference input matrix
        row (int): row index
        col (int): column index

    Returns:
        Any: matrix element
    """
    offset = int((reference.shape[0] - 1) / 2)
    if row > reference.shape[0] or col > reference.shape[1]:
        raise IndexError
    output = reference[row + offset, col + offset]
    return output


@njit
def P(
    i: int,
    a: int,
    b: int,
    order: int,
    rotation_matrices: TypedList,
) -> TypedList:
    """P function for rotation matrix calculation.

    Args:
        i (int): index
        a (int): 'a' value
        b (int): 'b' value
        order (int): order
        r (list(matrix)): rotation matrices

    Returns:
        float: P value
    """
    p = 0.0
    if b == order:
        p = (
            centred_element(rotation_matrices[1], i, 1)
            * centred_element(rotation_matrices[order - 1], a, order - 1)
        ) - (
            centred_element(rotation_matrices[1], i, -1)
            * centred_element(rotation_matrices[order - 1], a, -order + 1)
        )
    elif b == -order:
        p = (
            centred_element(rotation_matrices[1], i, 1)
            * centred_element(rotation_matrices[order - 1], a, -order + 1)
        ) + (
            centred_element(rotation_matrices[1], i, -1)
            * centred_element(rotation_matrices[order - 1], a, order - 1)
        )
    else:
        p = centred_element(rotation_matrices[1], i, 0) * centred_element(
            rotation_matrices[order - 1], a, b
        )

    return p


@njit
def U(
    degree: int,
    n: int,
    order: int,
    rotation_matrices: TypedList,
) -> TypedList:
    """U coefficient initialiser for rotation matrix calculation.

    Args:
        rotation_degree (int): Upper parameters of spherical harmonic
                               component Y and n the lower.
        n (int): index
        order (int): order
        rotation_matrices (list(matrix)): rotation matrices

    Returns:
        float: U value
    """
    return P(0, degree, n, order, rotation_matrices)


@njit
def V(
    degree: int,
    n: int,
    order: int,
    rotation_matrices: TypedList,
) -> TypedList:
    """V coefficient initialiser for rotation matrix calculation.

    Args:
        degree (int): valid inputs are `int(|m|) <= order`.
        n (int): index
        order (int): order
        rotation_matrices (list(matrix)): rotation matrices

    Returns:
        float: V value
    """
    d = 0.0
    if degree == 0:
        v_coeff = P(1, 1, n, order, rotation_matrices) + P(
            -1, -1, n, order, rotation_matrices
        )
    elif degree > 0:
        if degree == 1:
            d = 1.0
        v_coeff = P(1, degree - 1, n, order, rotation_matrices) * np.sqrt(1 + d) - P(
            -1, -degree + 1, n, order, rotation_matrices
        ) * (1 - d)
    else:
        if degree == -1:
            d = 1.0
        v_coeff = P(1, degree + 1, n, order, rotation_matrices) * (1 - d) + P(
            -1, -degree - 1, n, order, rotation_matrices
        ) * np.sqrt(1 + d)
    return v_coeff


@njit
def W(degree: int, n: int, order: int, rotation_matrices: TypedList) -> TypedList:
    """W coefficient initialiser for rotation matrix calculation.

    Args:
        degree (int): degree
        n (int): index
        order (int): order
        rotation_matrices (list(matrix)): rotation matrices

    Returns:
        float: W value
    """
    if degree == 0:
        w_value = 0.0
    elif degree > 0:
        w_value = P(1, degree + 1, n, order, rotation_matrices) + P(
            -1, -degree - 1, n, order, rotation_matrices
        )
    else:
        w_value = P(1, degree - 1, n, order, rotation_matrices) - P(
            -1, -degree + 1, n, order, rotation_matrices
        )
    return w_value


@njit
def compute_UVW_coefficients(degree, n, order):
    """Compute U, V and W coefficients for rotation matrix calculation.

    Args:
        m (index): degree
        n (index): index
        el (index): order

    Returns:
        tuple: u, v, w
    """
    d = 1 if degree == 0 else 0
    denom = (
        float(2 * order * (2 * order - 1))
        if np.abs(n) == order
        else float((order + n) * (order - n))
    )

    inverse_denom = 1.0 / denom
    u = np.sqrt(float((order + degree) * (order - degree)) * inverse_denom)
    v = (
        0.5
        * np.sqrt(
            (1.0 + d)
            * float(order + np.abs(degree) - 1)
            * (float(order + np.abs(degree)))
            * inverse_denom
        )
        * (1.0 - 2.0 * d)
    )
    w = (
        -0.5
        * np.sqrt(
            float(order - np.abs(degree) - 1)
            * float(order - np.abs(degree))
            * inverse_denom
        )
        * (1.0 - d)
    )

    return u, v, w


@njit
def compute_band_rotation(order: int, rotation_matrices: TypedList, output):
    """Compute submatrix for rotation matrix.

    Args:
        order (int): order of submatrix
        rotationmatrices (list(matrix)): previous and current submatrices
        output (matrix): output destination

    Returns:
        matrix: rotation submatrix
    """
    # print(f'entering band rotation with l = {el}')
    for row, m in enumerate(np.arange(-order, order + 1, 1)):
        for col, n in enumerate(np.arange(-order, order + 1, 1)):
            u, v, w = compute_UVW_coefficients(m, n, order)
            if np.abs(u) > 0.0:
                u *= U(m, n, order, rotation_matrices)
            if np.abs(v) > 0.0:
                v *= V(m, n, order, rotation_matrices)
            if np.abs(w) > 0.0:
                w *= W(m, n, order, rotation_matrices)
            rotation_matrices[order][row, col] = u + v + w

    starting_index = order * order

    output[
        starting_index : (starting_index + (order * 2 + 1)),
        starting_index : (starting_index + (order * 2 + 1)),
    ] = rotation_matrices[order]

    return output, rotation_matrices


@njit
def dot(A, B):
    """Wraps np.dot for numba #@njit.

    Args:
        A (Array)
        B (Array)

    Returns:
        Array: output
    """
    return np.dot(A, B)


class HOARotator:
    """Provides methods for rotating ambisonics."""

    def __init__(self, order, resolution):
        """Initialize HOARotator."""
        self.order = order
        self.resolution = resolution

        thetas = np.arange(0, 360, resolution)
        n_channels = (order + 1) ** 2

        self.rotmat = np.empty((len(thetas), n_channels, n_channels))

        # NOTE: We are rotating about "z" but construct the rotation matrix
        # with from_euler("y",...) because compute_rotation_matrix
        # is based on Google Resonance Audio code that is y-up.
        #
        # NOTE: the inv is only there to reverse the direction and could
        # be done more easily using -x_value.  The reverse is needed because
        # turning the head 10 degrees to the right is equiv to rotating the room
        # 10 degrees to the left. i.e. rotations given as head movements but
        # are being effected as room movements.
        for i, theta in enumerate(thetas):
            foa_rotmat = R.from_euler("y", theta, degrees=True).as_matrix()
            foa_rotmat = np.linalg.inv(foa_rotmat)
            self.rotmat[i, :, :] = compute_rotation_matrix(order, foa_rotmat)

    def rotate(self, signal: ndarray, rotation_vector: ndarray) -> ndarray:
        """Apply rotation to HOA signals using precomputed rotation matrices.

        Args:
            signal (array-like): ambisonic signals
            rotation_vector (array-like): rotation vector (in radians)

        Returns:
            array-like: transformed ambisonic signals
        """
        # Convert to lookup table indices
        # NOTE: Using generators below - using list comprehensions is *much* slower
        theta_i = (rotation_vector / np.pi * 180) / self.resolution  # convert to index
        theta_0 = np.floor(theta_i).astype(int)
        n_entries = self.rotmat.shape[0]  # size of table
        t_r0 = (self.rotmat[t % n_entries] for t in theta_0)
        t_r1 = (self.rotmat[(t + 1) % n_entries] for t in theta_0)
        # Interpolate between the two nearest rotation matrices
        t_interp = (
            t_r0_ + alpha_ * (t_r1_ - t_r0_)
            for t_r0_, t_r1_, alpha_ in zip(t_r0, t_r1, theta_i - theta_0)
        )
        # Apply the interpolated rotation matrices to the signal
        signal = np.array(
            [np.dot(x, _t_interp) for x, _t_interp in zip(signal, t_interp)]
        )
        return signal


def binaural_mixdown(
    ambisonic_signals: ndarray,
    hrir: dict[str, Any],
    hrir_metadata: dict[str, Any],
) -> ndarray:
    """Perform binaural mixdown of ambisonic signals.

    Args:
        ambisonic_signals (array-like): inputs
        hrir_filename (string): name of HRIR file
        hrir_metadata (dict): data for channel selection and ambisonic decoding

    Returns:
        array: stereo audio
    """
    # weights = np.array(hrir_metadata["weights"])
    matrix = np.array(hrir_metadata["matrix"])
    logger.info(f"Decoding signal with shape {ambisonic_signals.shape}")
    logger.info("Decoding to {matrix.shape[0]} positions")

    # Decode to loudspeaker positions

    # # No need to apply max-rE weights given we are using
    # # virtual speakers and headphones
    # y = np.dot(ambisonic_signals * weights, inv_matrix)

    n_chans = ambisonic_signals.shape[1]
    inv_matrix = np.linalg.pinv(matrix[:, 0:n_chans])
    y = np.dot(ambisonic_signals, inv_matrix)

    stereo_audio = np.zeros([y.shape[0] + hrir["M_data"].shape[0] - 1, 2])
    hrir_data = hrir["M_data"][:, hrir_metadata["selected_channels"], :]
    for i in np.arange(y.shape[1]):
        stereo_audio[:, 0] += convolve(y[:, i], hrir_data[:, i, 0])
        stereo_audio[:, 1] += convolve(y[:, i], hrir_data[:, i, 1])

    return stereo_audio


def ambisonic_convolve(
    signal: ndarray, hoa_impulse_responses: ndarray, order: int
) -> ndarray:
    """Convolve HOA Impulse Responses with signals.

    Args:
        signal (ndarray[samples]): the signal to convole
        hoa_impulse_response (ndarray[samples, channels]): the HOA impulse responses
        order (int, optional): ambisonic order.

    Returns:
        np.ndarray[samples, channels]: the convolved signal
    """
    n = (order + 1) ** 2
    if n > hoa_impulse_responses.shape[1]:
        raise ValueError(
            f"Number of channels in impulse response ({hoa_impulse_responses.shape[1]})"
            f" must be >= number of channels required for order {order} ({n})"
        )
    return np.array(
        [
            convolve(impulse_response, signal)
            for impulse_response in hoa_impulse_responses[:, 0:n].T
        ]
    ).T


def compute_rms(input_signal: ndarray, axis: int = 0) -> ndarray:
    """Compute rms values along a given axis.
    Args:
        input_signal (np.ndarray): Input signal
        axis (int): Axis along which to compute the Root Mean Square. 0 (default) or 1.

    Returns:
        float: Root Mean Square for the given axis."""

    return np.sqrt(np.mean(input_signal**2, axis=axis))


def equalise_rms_levels(inputs: list[ndarray]) -> list[ndarray]:
    """Equalise RMS levels.

    Args:
        inputs (array): signals

    Returns:
        array: normalised signals
    """
    rms = compute_rms(np.array(inputs)[:, :, 0], axis=1)
    levels = rms / np.max(rms)
    outputs = [input / level for level, input in zip(levels, inputs)]
    return outputs


def dB_to_gain(x: int | float) -> float:
    """Convert dB to gain.

    Args:
        x (float):

    Returns
        float:
    """
    return 10 ** (0.05 * x)


def smoothstep(
    x: ndarray, x_min: float = 0.0, x_max: float = 1.0, N: int = 1
) -> ndarray:
    """Apply the smoothstep function.

    Args:
        x (np.ndarray): input
        x_min (float, optional): clamp minimum. Defaults to 0.
        x_max (float, optional): clamp maximum. Defaults to 1.
        N (int, optional): smoothing factor. Defaults to 1.

    Returns:
        np.ndarray: smoothstep values
    """
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = sum(
        comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n for n in range(0, N + 1)
    )

    result *= x ** (N + 1)

    return result


def rotation_control_vector(
    array_length: int, start_index: int, end_index: int, smoothness: int = 1
) -> ndarray:
    """Generate mapped rotation control vector for values of theta.

    Args:
        array_length (int): Length of array
        start_index (int): Start position
        end_index (int)
        smoothness (int, optional) Defaults to 1.

    Returns:
        array: mapped rotation control vector
    """
    # assert end_idx > start_idx
    # assert array_length > end_idx
    if start_index > end_index:
        raise ValueError(f"start_index ({start_index}) > end_index ({end_index})")
    if end_index > array_length:
        raise ValueError(f"array_length ({array_length}) > end_index {end_index}")
    x = np.arange(0, array_length)
    idx = smoothstep(x, x_min=start_index, x_max=end_index, N=smoothness)
    return np.array(np.floor(idx * (array_length - 1)), dtype=int)


def compute_rotation_vector(
    start_angle: float,
    end_angle: float,
    signal_length: int,
    start_idx: int,
    end_idx: int,
) -> ndarray:
    """Compute the rotation vector.

    Args:
        start_angle (float)
        end_angle (float)
        signal_length (int)
        start_idx (int)
        end_idx (int)

    Returns:
        np.array: _description_
    """
    turn_direction = -np.sign(start_angle - end_angle)
    with np.errstate(divide="raise"):
        increment = (np.abs(start_angle - end_angle) / signal_length) * turn_direction

    theta = np.arange(start_angle, end_angle, increment)
    idx = rotation_control_vector(signal_length, start_idx, end_idx)
    return theta[idx]
