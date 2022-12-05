"""Tools to support higher order ambisonic processing."""
import logging
from typing import List

import numpy as np
from numba import njit
from numba.typed import List as TypedList
from scipy.signal import convolve
from scipy.spatial.transform import Rotation as R
from scipy.special import comb

logger = logging.getLogger(__name__)


# Code for generation ambisonic rotation matrices


@njit
def compute_rotation_matrix(n: int, foa_rotmat: np.ndarray) -> np.ndarray:
    """Generate a rotation matrix to rotate HOA soundfield.
    Based on 'Rotation Matrices for Real Spherical Harmonics. Direct Determination
    by Recursion' Joseph Ivanic and Klaus Ruedenberg J. Phys. Chem. 1996, 100, 15,
    6342–6347 and Gorzel, M., Allen, A., Kelly, I., Kammerl, J., Gungormusler,
    A., Yeh, H., and Boland, F., "Efficient Encoding and Decoding of Binaural Sound
    with Resonance Audio", In proc. of the AES International Conference on Immersive
    and Interactive Audio, March 2019
    Operates on HOA of a given order rotates by azimuth theta
    and elevation phi

    Args:
        order (int): order of ambisonic soundfield
        foa_rotmat (arraylike): rotation matrix to expand

    Returns:
        np.ndarray: HOA rotation matrix
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
    [typed_sub_matrices.append(x) for x in sub_matrices]

    if n > 1:
        for i in np.arange(2, n + 1):
            rot_mat, typed_sub_matrices = compute_band_rotation(
                i, typed_sub_matrices, rot_mat
            )

    return rot_mat


@njit
def centred_element(r: np.ndarray, i: int, j: int):
    """Get value from centered element indexing.

    Args:
        r (matrix): input matrix
        i (int): row index
        j (int): column index

    Returns:
        Any: matrix element
    """
    offset = int((r.shape[0] - 1) / 2)
    if i > r.shape[0] or j > r.shape[1]:
        raise IndexError
    output = r[i + offset, j + offset]
    return output


@njit
def P(i, a, b, el, r):
    """P function for rotation matrix calculation.

    Args:
        i (int): index
        a (int): 'a' value
        b (int): 'b' value
        el (int): order
        r (list(matrix)): rotation matrices

    Returns:
        float: P value
    """
    p = 0.0
    if b == el:
        p = (centred_element(r[1], i, 1) * centred_element(r[el - 1], a, el - 1)) - (
            centred_element(r[1], i, -1) * centred_element(r[el - 1], a, -el + 1)
        )
    elif b == -el:
        p = (centred_element(r[1], i, 1) * centred_element(r[el - 1], a, -el + 1)) + (
            centred_element(r[1], i, -1) * centred_element(r[el - 1], a, el - 1)
        )
    else:
        p = centred_element(r[1], i, 0) * centred_element(r[el - 1], a, b)

    return p


@njit
def U(m, n, el, r):
    """U coefficient initialiser for rotation matrix calculation.

    Args:
        m (int): degree
        n (int): index
        el (int): order
        r (list(matrix)): rotation matrices

    Returns:
        float: U value
    """
    return P(0, m, n, el, r)


@njit
def V(m, n, el, r):
    """V coefficient initialiser for rotation matrix calculation.

    Args:
        m (int): degree
        n (int): index
        el (int): order
        r (list(matrix)): rotation matrices

    Returns:
        float: V value
    """
    d = 0
    if m == 0:
        return P(1, 1, n, el, r) + P(-1, -1, n, el, r)
    elif m > 0:
        if m == 1:
            d = 1.0
        return P(1, m - 1, n, el, r) * np.sqrt(1 + d) - P(-1, -m + 1, n, el, r) * (
            1 - d
        )
    else:
        if m == -1:
            d = 1.0
        return P(1, m + 1, n, el, r) * (1 - d) + P(-1, -m - 1, n, el, r) * np.sqrt(
            1 + d
        )


@njit
def W(m, n, el, r):
    """W coefficient initialiser for rotation matrix calculation.

    Args:
        m (int): degree
        n (int): index
        el (int): order
        r (list(matrix)): rotation matrices

    Returns:
        float: W value
    """
    if m == 0:
        return 0.0
    elif m > 0:
        return P(1, m + 1, n, el, r) + P(-1, -m - 1, n, el, r)
    else:
        return P(1, m - 1, n, el, r) - P(-1, -m + 1, n, el, r)


@njit
def compute_UVW_coefficients(m, n, el):
    """Compute U, V and W coefficients for rotation matrix calculation.

    Args:
        m (index): degree
        n (index): index
        el (index): order

    Returns:
        tuple: u, v, w
    """
    d = 0
    if m == 0:
        d = 1
    denom = float((el + n) * (el - n))
    if np.abs(n) == el:
        denom = float(2 * el * (2 * el - 1))

    one_over_denom = 1.0 / denom
    u = np.sqrt(float((el + m) * (el - m)) * one_over_denom)
    v = (
        0.5
        * np.sqrt(
            (1.0 + d)
            * float(el + np.abs(m) - 1)
            * (float(el + np.abs(m)))
            * one_over_denom
        )
        * (1.0 - 2.0 * d)
    )
    w = (
        -0.5
        * np.sqrt(float(el - np.abs(m) - 1) * float(el - np.abs(m)) * one_over_denom)
        * (1.0 - d)
    )

    return u, v, w


@njit
def compute_band_rotation(el, rotations, output):
    """Compute submatrix for rotation matrix.

    Args:
        el (int): order of submatrix
        rotations (list(matrix)): previous and current submatrices
        output (matrix): output destination

    Returns:
        matrix: rotation submatrix
    """
    # print(f'entering band rotation with l = {el}')

    for mm, m in enumerate(np.arange(-el, el + 1, 1)):
        for nn, n in enumerate(np.arange(-el, el + 1, 1)):
            u, v, w = compute_UVW_coefficients(m, n, el)
            if np.abs(u) > 0.0:
                uu = U(m, n, el, rotations)
                u *= uu
            if np.abs(v) > 0.0:
                vv = V(m, n, el, rotations)
                v *= vv
            if np.abs(w) > 0.0:
                ww = W(m, n, el, rotations)
                w *= ww
            rotations[el][mm, nn] = u + v + w

    starting_index = el * el

    output[
        starting_index : (starting_index + (el * 2 + 1)),
        starting_index : (starting_index + (el * 2 + 1)),
    ] = rotations[el]

    return output, rotations


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

    def rotate(self, signal, th):
        """Apply rotation to HOA signals using precomputed rotation matrices.

        Args:
            signal (array-like): ambisonic signals
            th (array-like): rotation vector (in radians)

        Returns:
            array-like: transformed ambisonic signals
        """
        # Convert to lookup table indices
        # NOTE: Using generators below - using list comprehensions is *much* slower
        theta_i = (th / np.pi * 180) / self.resolution  # convert to index
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


def binaural_mixdown(ambisonic_signals, hrir, hrir_metadata):
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
    logger.info("decoding signal with shape %s", ambisonic_signals.shape)
    logger.info("Decoding to %s positions", matrix.shape[0])

    # Decode to loudspeaker positions

    # # No need to apply max-rE weights given we are using virtual speakers and headphones
    # y = np.dot(ambisonic_signals * weights, inv_matrix)

    n_chans = ambisonic_signals.shape[1]
    inv_matrix = np.linalg.pinv(matrix[:, 0:n_chans])
    y = np.dot(ambisonic_signals, inv_matrix)

    z = np.zeros([y.shape[0] + hrir["M_data"].shape[0] - 1, 2])
    hrir_data = hrir["M_data"][:, hrir_metadata["selected_channels"], :]
    for i in np.arange(y.shape[1]):
        z[:, 0] += convolve(y[:, i], hrir_data[:, i, 0])
        z[:, 1] += convolve(y[:, i], hrir_data[:, i, 1])

    return z


def ambisonic_convolve(signal: np.ndarray, ir: np.ndarray, order: int) -> np.ndarray:
    """Convolve HOAIRs with signals.

    Args:
        signal (ndarray[samples]): the signal to convole
        ir (ndarray[samples, channels]): the HOA impulse responses
        order (int, optional): ambisonic order.

    Returns:
        np.ndarray[samples, channels]: the convolved signal
    """
    n = (order + 1) ** 2
    return np.array([convolve(ir_, signal) for ir_ in ir[:, 0:n].T]).T


def compute_rms(input_signal: np.ndarray, axis: int = 0):
    """Compute rms values along a given axis.
    Args:
        input_signal (np.ndarray): Input signal
        axis (int): Axis along which to compute the Root Mean Square. 0 (default) or 1.

    Returns:
        float: Root Mean Square for the given axis."""

    return np.sqrt(np.mean(input_signal**2, axis=axis))


def equalise_rms_levels(inputs: List[np.ndarray]) -> List[np.ndarray]:
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


def dB_to_gain(x: float) -> float:
    """Convert dB to gain.

    Args:
        x (float):

    Returns
        float:
    """
    return 10 ** (0.05 * x)


def smoothstep(
    x: np.ndarray, x_min: float = 0, x_max: float = 1, N: int = 1
) -> np.ndarray:
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
    array_length: int, start_idx: int, end_idx: int, smoothness: int = 1
) -> np.ndarray:
    """Generate mapped rotation control vector for values of theta.

    Args:
        array_length (int)
        start_idx (int)
        end_idx (int)
        smoothness (int, optional) Defaults to 1.

    Returns:
        array: mapped rotation control vector
    """
    # assert end_idx > start_idx
    # assert array_length > end_idx
    if start_idx > end_idx:
        raise ValueError(f"start_idx ({start_idx}) > end_idx ({end_idx})")
    if end_idx > array_length:
        raise ValueError(f"array_length ({array_length}) > end_idx {end_idx}")
    x = np.arange(0, array_length)
    idx = smoothstep(x, x_min=start_idx, x_max=end_idx, N=smoothness)
    idx = np.array(np.floor(idx * (array_length - 1)), dtype=int)
    return idx


def rotation_vector(
    start_angle: float,
    end_angle: float,
    signal_length: int,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
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
        try:
            increment = (
                np.abs(start_angle - end_angle) / signal_length
            ) * turn_direction
        except FloatingPointError:
            raise
    theta = np.arange(start_angle, end_angle, increment)
    idx = rotation_control_vector(signal_length, start_idx, end_idx)
    return theta[idx]
