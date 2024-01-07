import math
from typing import Tuple

import numpy as np


def vect_to_zrelion(normal: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute rotation angles from an input vector.

    Aligns the input vector with the
    reference [0,0,1] vector, assuming a free Euler angle in Relion format.
    The first Euler angle (Rotation) is assumed 0.

    adapted from Antonio Martinez Sanchez' PySeg:
    https://github.com/anmartinezs/pyseg_system/blob/master/code/pyseg/globals/utils.py

    Parameters
    ----------
    normal : np.ndarray
        Input vector.

    Returns
    -------
    Tuple[float, float, float]
        The Euler angles in Relion format (rotation, tilt, psi).
    """
    # Normalization
    v_m = np.asarray((normal[0], normal[1], normal[2]), dtype=np.float32)
    try:
        n = v_m / (math.sqrt((v_m * v_m).sum()) + 1e-6)
    except ZeroDivisionError:
        print("WARNING (vect_rotation_ref): vector with module 0 cannot be rotated!")
        return 0.0, 0.0, 0.0

    # Computing angles in Extrinsic ZYZ system
    alpha = np.arccos(n[2])
    beta = np.arctan2(n[1], n[0])

    # Transform to Relion system (intrinsic ZY'Z'' where rho is free)
    rot, tilt, psi = (
        0.0,
        unroll_angle(math.degrees(alpha), deg=True),
        unroll_angle(180.0 - math.degrees(beta), deg=True),
    )
    return rot, tilt, psi


def unroll_angle(angle: float, deg: bool = True) -> float:
    """
    Unroll an angle [-infty, infty] to fit range [-180, 180] or (-pi, pi) in radians.

    copied from Antonio Martinez Sanchez' PySeg:
    https://github.com/anmartinezs/pyseg_system/blob/master/code/pyseg/globals/utils.py

    Parameters
    ----------
    angle : float
        Input angle.
    deg : bool, optional
        If True (default), the angle is in degrees, otherwise in radians.

    Returns
    -------
    float
        The unrolled angle.
    """
    fang = float(angle)
    if deg:
        mx_ang, mx_ang2 = 360.0, 180.0
    else:
        mx_ang, mx_ang2 = 2 * np.pi, np.pi
    ang_mod, ang_sgn = np.abs(fang), np.sign(fang)
    ur_ang = ang_mod % mx_ang
    if ur_ang > mx_ang2:
        return -1.0 * ang_sgn * (mx_ang - ur_ang)
    else:
        return ang_sgn * ur_ang


def x_rot_matrix(alpha: float) -> np.ndarray:
    """
    Generate a rotation matrix for rotation around the x-axis.

    Parameters
    ----------
    alpha : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        The rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -1 * np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )


def z_rot_matrix(phi: float) -> np.ndarray:
    """
    Generate a rotation matrix for rotation around the z-axis.

    Parameters
    ----------
    phi : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        The rotation matrix.
    """
    return np.array(
        np.array(
            [
                [np.cos(phi), -1 * np.sin(phi), 0.0],
                [np.sin(phi), np.cos(phi), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )


def y_rot_matrix(theta: float) -> np.ndarray:
    """
    Generate a rotation matrix for rotation around the y-axis.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        The rotation matrix.
    """
    return np.array(
        np.array(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-1 * np.sin(theta), 0.0, np.cos(theta)],
            ]
        )
    )


def zyz_rot_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Generate a rotation matrix for a Z-Y-Z Euler rotation.

    Parameters
    ----------
    phi : float
        First rotation angle (around z-axis) in radians.
    theta : float
        Second rotation angle (around y-axis) in radians.
    psi : float
        Third rotation angle (around z-axis) in radians.

    Returns
    -------
    np.ndarray
        The rotation matrix corresponding to the Z-Y-Z Euler rotation.
    """
    a = np.cos(phi)
    b = np.sin(phi)
    c = np.cos(theta)
    d = np.sin(theta)
    e = np.cos(psi)
    f = np.sin(psi)
    return np.array(
        [
            [e * a * c - b * f, -a * c * f - e * b, a * d],
            [a * f + e * b * c, e * a - b * c * f, b * d],
            [-e * d, d * f, c],
        ]
    )


def zxz_from_rotation_matrix(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Z-X-Z Euler angles from a given rotation matrix.

    Parameters
    ----------
    rotation_matrix : np.ndarray
        The rotation matrix.

    Returns
    -------
    Tuple[float, float, float]
        The Euler angles (phi, theta, psi).
    """
    phi = np.arctan2(rotation_matrix[2, 0], rotation_matrix[2, 1])
    psi = np.arctan2(-rotation_matrix[0, 2], rotation_matrix[1, 2])
    theta = np.arctan2(
        rotation_matrix[2, 1] * np.cos(phi) + rotation_matrix[2, 0] * np.sin(phi),
        rotation_matrix[2, 2],
    )
    return phi, theta, psi


def zyz_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Compute a rotation matrix from Z-Y-Z Euler angles.

    Parameters
    ----------
    phi : float
        First rotation angle (around z-axis) in radians.
    theta : float
        Second rotation angle (around y-axis) in radians.
    psi : float
        Third rotation angle (around z-axis) in radians.

    Returns
    -------
    np.ndarray
        The rotation matrix corresponding to the Z-Y-Z Euler rotation.
    """
    return np.dot(z_rot_matrix(psi), np.dot(y_rot_matrix(theta), z_rot_matrix(phi)))


def zxz_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Compute a rotation matrix from Z-X-Z Euler angles.

    Parameters
    ----------
    phi : float
        First rotation angle (around z-axis) in radians.
    theta : float
        Second rotation angle (around x-axis) in radians.
    psi : float
        Third rotation angle (around z-axis) in radians.

    Returns
    -------
    np.ndarray
        The rotation matrix corresponding to the Z-X-Z Euler rotation.
    """
    return np.dot(z_rot_matrix(psi), np.dot(x_rot_matrix(theta), z_rot_matrix(phi)))


def compute_Euler_angles_for_normals(
    points: np.ndarray, normals: np.ndarray, convention: str = "zxz"
) -> np.ndarray:
    """
    For an array of points and normals, compute and return the Euler angles.

    Parameters
    ----------
    points : np.ndarray
        Array of points.
    normals : np.ndarray
        Array of normal vectors corresponding to the points.
    convention : str, optional
        The convention to use for Euler angles ('zxz' or 'zyz').
        Default is 'zxz'.

    Returns
    -------
    np.ndarray
        Array of computed Euler angles.

    Raises
    ------
    IOError
        If an unknown convention is specified.
    """
    if convention not in ["zxz", "zyz"]:
        raise OSError("Convention not known.")

    new_angles = []
    for point, normal in zip(points, normals):
        point, normal = point.tolist(), normal.tolist()

        # compute zyz angles to align normal with z-axis
        angle = vect_to_zrelion(list(normal))

        if convention == "zyz":
            new_angles.append(angle)
        elif convention == "zxz":
            # Generate rotation matrix from zyz angles
            rot_mat = zyz_rot_matrix(
                phi=np.deg2rad(angle[0]),
                theta=np.deg2rad(angle[1]),
                psi=np.deg2rad(angle[2]),
            )

            # Compute zxz angles from rotation matrix
            zxz_angles = zxz_from_rotation_matrix(rot_mat)

            # Generate rotation matrix from zxz angles
            rot_mat = zxz_to_rotation_matrix(*zxz_angles)

            # Invert rotation matrix
            rot_mat = -rot_mat.T

            # Compute zxz angles from inverted rotation matrix
            zxz_angles_new = np.rad2deg(zxz_from_rotation_matrix(rot_mat))

            # Store zxz angles
            angle = zxz_angles_new.tolist()

            # Adjust tilt angle
            angle[1] = 180 - angle[1]
            new_angles.append(angle)
    new_angles = np.stack(new_angles, axis=0)
    return new_angles
