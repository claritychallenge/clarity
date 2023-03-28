import numpy as np
from numba.typed import List as TypedList  # pylint: disable=no-name-in-module
from scipy.spatial.transform import Rotation as R

from clarity.data import HOA_tools_cec2 as hoa


def test_compute_rotation_matrix(regtest):
    theta = 45
    order = 2
    foa = R.from_euler("y", theta, degrees=True).as_matrix()
    rotmat = hoa.compute_rotation_matrix.py_func(order, foa)
    regtest.write(f"Rotation matrix {rotmat}\n")


def test_centred_element(regtest):
    np.random.seed(0)
    matrix = np.random.rand(9, 9)
    centred_element = hoa.centred_element.py_func(matrix, 0, 0)
    regtest.write(f"centered element {centred_element:0.3f}\n")


def test_P(regtest):
    theta = 45
    order = 2
    foa = R.from_euler("y", theta, degrees=True).as_matrix()
    rotmats = TypedList([foa, hoa.compute_rotation_matrix(order, foa)])
    p = hoa.P.py_func(0, 1, 0, order, rotmats)
    regtest.write(f"P value {p:0.3f}\n")


def test_U(regtest):
    m = 0
    n = 0
    theta = 45
    order = 2
    foa = R.from_euler("y", theta, degrees=True).as_matrix()
    rotmats = TypedList([foa, hoa.compute_rotation_matrix(order, foa)])
    u = hoa.U.py_func(m, n, order, rotmats)
    regtest.write(f"U value {u:0.3f}\n")


def test_V(regtest):
    n = 0
    theta = 45
    order = 2
    foa = R.from_euler("y", theta, degrees=True).as_matrix()
    rotmats = TypedList([foa, hoa.compute_rotation_matrix(order, foa)])
    for i in range(3):
        v = hoa.V.py_func(i - 1, n, order, rotmats)
        regtest.write(f"V[{i}] value {v:0.3f}\n")


def test_W(regtest):
    n = 0
    theta = 45
    order = 2
    foa = R.from_euler("y", theta, degrees=True).as_matrix()
    rotmats = TypedList([foa, hoa.compute_rotation_matrix(order, foa)])
    for i in range(3):
        w = hoa.W.py_func(i - 1, n, order, rotmats)
        regtest.write(f"W[{i}] value {w:0.3f}\n")


def test_compute_UVW_coefficients(regtest):
    m = 0
    n = 2
    order = 2
    u, v, w = hoa.compute_UVW_coefficients.py_func(m, n, order)
    regtest.write(f"UVW coefficients {u:0.3f}, {v:0.3f}, {w:0.3f}\n")


def test_compute_band_rotation(regtest):
    theta = 45
    n = 2
    foa_rotmat = R.from_euler("y", theta, degrees=True).as_matrix()
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
    typed_sub_matrices = sub_matrices.copy()

    if n > 1:
        for i in np.arange(2, n + 1):
            rot_mat, typed_sub_matrices = hoa.compute_band_rotation.py_func(
                i, TypedList(typed_sub_matrices), rot_mat
            )
    regtest.write(f"Band rotations {rot_mat}, {sub_matrices}\n")
