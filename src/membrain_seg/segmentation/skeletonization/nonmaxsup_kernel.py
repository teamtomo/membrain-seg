import numpy as np


def nonmaxsup_kernel(I, Vx, Vy, Vz, M, inter_factor):
    """Initialize F as all-zero matrix."""
    ld = len(M)
    Nx, Ny, Nz = I.shape
    F = np.zeros((Nx, Ny, Nz))
    look_neighborhood(ld, I, Vx, Vy, Vz, M, F, inter_factor)
    return F


def look_neighborhood(ld, I, Vx, Vy, Vz, M, F, inter_factor):
    """Looking 8 adjacent points for trilinear interpolation."""
    # Buffers initialization
    A = np.zeros((8, ld))
    B = np.zeros((8, ld))
    Va = np.zeros(ld)
    Vb = np.zeros(ld)

    # Prepare data for every coordinate
    for j in range(ld):
        x, y, z = M[j]
        lv = I[x, y, z]
        A[0][j] = lv
        B[0][j] = lv

        kx = np.abs(Vx[x, y, z] * inter_factor)
        ky = np.abs(Vy[x, y, z] * inter_factor)
        kz = np.abs(Vz[x, y, z] * inter_factor)

        if Vx[x, y, z] >= 0:
            if Vy[x, y, z] >= 0 and Vz[x, y, z] >= 0:
                A[1][j] = I[x, y, z + 1]
                A[2][j] = I[x, y + 1, z]
                A[3][j] = I[x, y + 1, z + 1]
                A[4][j] = I[x + 1, y, z]
                A[5][j] = I[x + 1, y, z + 1]
                A[6][j] = I[x + 1, y + 1, z]
                A[7][j] = I[x + 1, y + 1, z + 1]

                B[1][j] = I[x, y, z - 1]
                B[2][j] = I[x, y - 1, z]
                B[3][j] = I[x, y - 1, z - 1]
                B[4][j] = I[x - 1, y, z]
                B[5][j] = I[x - 1, y, z - 1]
                B[6][j] = I[x - 1, y - 1, z]
                B[7][j] = I[x - 1, y - 1, z - 1]

            elif Vy[x, y, z] < 0 and Vz[x, y, z] >= 0:
                A[1][j] = I[x, y, z + 1]
                A[2][j] = I[x, y - 1, z]
                A[3][j] = I[x, y - 1, z + 1]
                A[4][j] = I[x + 1, y, z]
                A[5][j] = I[x + 1, y, z + 1]
                A[6][j] = I[x + 1, y - 1, z]
                A[7][j] = I[x + 1, y - 1, z + 1]

                B[1][j] = I[x, y, z - 1]
                B[2][j] = I[x, y + 1, z]
                B[3][j] = I[x, y + 1, z - 1]
                B[4][j] = I[x - 1, y, z]
                B[5][j] = I[x - 1, y, z - 1]
                B[6][j] = I[x - 1, y + 1, z]
                B[7][j] = I[x - 1, y + 1, z - 1]

            elif Vy[x, y, z] >= 0 and Vz[x, y, z] < 0:
                A[1][j] = I[x, y, z - 1]
                A[2][j] = I[x, y + 1, z]
                A[3][j] = I[x, y + 1, z - 1]
                A[4][j] = I[x + 1, y, z]
                A[5][j] = I[x + 1, y, z - 1]
                A[6][j] = I[x + 1, y + 1, z]
                A[7][j] = I[x + 1, y + 1, z - 1]

                B[1][j] = I[x, y, z + 1]
                B[2][j] = I[x, y - 1, z]
                B[3][j] = I[x, y - 1, z + 1]
                B[4][j] = I[x - 1, y, z]
                B[5][j] = I[x - 1, y, z + 1]
                B[6][j] = I[x - 1, y - 1, z]
                B[7][j] = I[x - 1, y - 1, z + 1]

            else:
                A[1][j] = I[x, y, z - 1]
                A[2][j] = I[x, y - 1, z]
                A[3][j] = I[x, y - 1, z - 1]
                A[4][j] = I[x + 1, y, z]
                A[5][j] = I[x + 1, y, z - 1]
                A[6][j] = I[x + 1, y - 1, z]
                A[7][j] = I[x + 1, y - 1, z - 1]

                B[1][j] = I[x, y, z + 1]
                B[2][j] = I[x, y + 1, z]
                B[3][j] = I[x, y + 1, z + 1]
                B[4][j] = I[x - 1, y, z]
                B[5][j] = I[x - 1, y, z + 1]
                B[6][j] = I[x - 1, y + 1, z]
                B[7][j] = I[x - 1, y + 1, z + 1]

        else:
            if Vy[x, y, z] >= 0 and Vz[x, y, z] >= 0:
                A[1][j] = I[x, y, z + 1]
                A[2][j] = I[x, y + 1, z]
                A[3][j] = I[x, y + 1, z + 1]
                A[4][j] = I[x - 1, y, z]
                A[5][j] = I[x - 1, y, z + 1]
                A[6][j] = I[x - 1, y + 1, z]
                A[7][j] = I[x - 1, y + 1, z + 1]

                B[1][j] = I[x, y, z - 1]
                B[2][j] = I[x, y - 1, z]
                B[3][j] = I[x, y - 1, z - 1]
                B[4][j] = I[x + 1, y, z]
                B[5][j] = I[x + 1, y, z - 1]
                B[6][j] = I[x + 1, y - 1, z]
                B[7][j] = I[x + 1, y - 1, z - 1]

            elif Vy[x, y, z] < 0 and Vz[x, y, z] >= 0:
                A[1][j] = I[x, y, z + 1]
                A[2][j] = I[x, y - 1, z]
                A[3][j] = I[x, y - 1, z + 1]
                A[4][j] = I[x - 1, y, z]
                A[5][j] = I[x - 1, y, z + 1]
                A[6][j] = I[x - 1, y - 1, z]
                A[7][j] = I[x - 1, y - 1, z + 1]

                B[1][j] = I[x, y, z - 1]
                B[2][j] = I[x, y + 1, z]
                B[3][j] = I[x, y + 1, z - 1]
                B[4][j] = I[x + 1, y, z]
                B[5][j] = I[x + 1, y, z - 1]
                B[6][j] = I[x + 1, y + 1, z]
                B[7][j] = I[x + 1, y + 1, z - 1]

            elif Vy[x, y, z] >= 0 and Vz[x, y, z] < 0:
                A[1][j] = I[x, y, z - 1]
                A[2][j] = I[x, y + 1, z]
                A[3][j] = I[x, y + 1, z - 1]
                A[4][j] = I[x - 1, y, z]
                A[5][j] = I[x - 1, y, z - 1]
                A[6][j] = I[x - 1, y + 1, z]
                A[7][j] = I[x - 1, y + 1, z - 1]

                B[1][j] = I[x, y, z + 1]
                B[2][j] = I[x, y - 1, z]
                B[3][j] = I[x, y - 1, z + 1]
                B[4][j] = I[x + 1, y, z]
                B[5][j] = I[x + 1, y, z + 1]
                B[6][j] = I[x + 1, y - 1, z]
                B[7][j] = I[x + 1, y - 1, z + 1]

            else:
                A[1][j] = I[x, y, z - 1]
                A[2][j] = I[x, y - 1, z]
                A[3][j] = I[x, y - 1, z - 1]
                A[4][j] = I[x - 1, y, z]
                A[5][j] = I[x - 1, y, z - 1]
                A[6][j] = I[x - 1, y - 1, z]
                A[7][j] = I[x - 1, y - 1, z - 1]

                B[1][j] = I[x, y, z + 1]
                B[2][j] = I[x, y + 1, z]
                B[3][j] = I[x, y + 1, z + 1]
                B[4][j] = I[x + 1, y, z]
                B[5][j] = I[x + 1, y, z + 1]
                B[6][j] = I[x + 1, y + 1, z]
                B[7][j] = I[x + 1, y + 1, z + 1]

    # Trilinear interpolation
    for j in range(ld):
        Va[j] = (
            A[0][j] * (1 - kx) * (1 - ky) * (1 - kz)
            + A[4][j] * kx * (1 - ky) * (1 - kz)
            + A[2][j] * (1 - kx) * ky * (1 - kz)
            + A[1][j] * (1 - kx) * (1 - ky) * kz
            + A[5][j] * kx * (1 - ky) * kz
            + A[3][j] * (1 - kx) * ky * kz
            + A[6][j] * kx * ky * (1 - kz)
            + A[7][j] * kx * ky * kz
        )

        Vb[j] = (
            B[0][j] * (1 - kx) * (1 - ky) * (1 - kz)
            + B[4][j] * kx * (1 - ky) * (1 - kz)
            + B[2][j] * (1 - kx) * ky * (1 - kz)
            + B[1][j] * (1 - kx) * (1 - ky) * kz
            + B[5][j] * kx * (1 - ky) * kz
            + B[3][j] * (1 - kx) * ky * kz
            + B[6][j] * kx * ky * (1 - kz)
            + B[7][j] * kx * ky * kz
        )

    # Mark local maxima
    for j in range(ld):
        x, y, z = M[j]
        lv = I[x, y, z]
        if lv > Va[j] and lv > Vb[j]:
            F[x, y, z] = 1
    return F
