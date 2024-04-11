import numpy as np
from scipy.linalg import eigh


def eigendecomposition(Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
    """Use scipy.linalg to solve eigenproblem."""
    m = len(Ixx)

    Qo = np.zeros((m, 3, 3), dtype=complex)
    w = np.zeros((m, 3), dtype=complex)

    for i in range(m):
        A = np.array(
            [
                [Ixx[i], Ixy[i], Ixz[i]],
                [Ixy[i], Iyy[i], Iyz[i]],
                [Ixz[i], Iyz[i], Izz[i]],
            ]
        )

        w_i, Qo_i = eigh(A)

        w[i] = w_i[::-1]
        Qo_i[:, [0, 2]] = Qo_i[:, [2, 0]]
        Qo[i] = Qo_i

    return (
        w[:, 0],
        w[:, 1],
        w[:, 2],
        Qo[:, 0, 0],
        Qo[:, 1, 0],
        Qo[:, 2, 0],
        Qo[:, 0, 1],
        Qo[:, 1, 1],
        Qo[:, 2, 1],
        Qo[:, 0, 2],
        Qo[:, 1, 2],
        Qo[:, 2, 2],
    )
