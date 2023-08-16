# Derived from: Python utilities for Focus
# Author: Ricardo Righetto
# E-mail: ricardo.righetto@unibas.ch
# https://github.com/C-CINA/focustools/

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

pi = np.pi  # global PI

# TO-DO:
# Port heavy calculations to Torch or something more efficient than pure NumPy?
# Original implementation used numexpr (see commented code) but that would add one more
# dependency to MemBrain and does not give significant speedups, at least not with the 
# defaults.
# For now, we stick to pure Numpy.


def RadialIndices(
    imsize: tuple = (128, 128),
    rounding: bool = True,
    normalize: bool = False,
    rfft: bool = False,
    xyz: tuple = (0, 0, 0),
    nozero: bool = True,
    nozeroval: float = 1e-3,
):
    """
    Generates a 1D/2D/3D array whose values are the distance counted from the origin. 

    Parameters
    ----------
    imsize : tuple
        The shape of the input ndarray.
    rounding : bool
        Whether the radius values should be rounded to the nearest integer ensuring \
"perfect" radial symmetry.
    normalize : bool
        Whether the radius values should be normalized to the range [0.0,11.0].
    rfft : bool
        Whether to return an array consistent with np.fft.rfftn i.e. exploiting the \
Hermitian symmetry of the Fourier transform of real data.
    xyz : tuple
        Shifts to be applied to the origin specified as (x_shift, y_shift, z_shift). \
Useful when applying phase shifts.
    nozero : bool
        Whether the value of the origin (corresponding to the zero frequency or DC \
component in the Fourier transform) should be set to a small value instead of zero.
    nozeroval : float
        The value to put at the origin if nozero is True.

    Returns
    -------
    rmesh : ndarray
        Array whose values are the distance from the origin.
    amesh : ndarray
        Array whose values are the angle from the x- axis (2D) or from the x,y plane \
(3D)

    Raises
    ------
    ValueError
        If imsize with more than 3 dimensions is given.

    Notes
    -----
    This function is compliant with NumPy fft.fftfreq() and fft.rfftfreq().
    """
    imsize = np.array(imsize)

    # if np.isscalar(imsize):
    #     imsize = [imsize, imsize]

    if len(imsize) > 3:
        raise ValueError(
            "Object should have 2 or 3 dimensions: len(imsize) = %d " % len(imsize)
        )

    xyz = np.flipud(xyz)

    m = np.mod(imsize, 2)  # Check if dimensions are odd or even

    if len(imsize) == 1:
        # The definition below is consistent with numpy np.fft.fftfreq and
        # np.fft.rfftfreq:

        if not rfft:
            xmesh = np.mgrid[
                -imsize[0] // 2 + m[0] - xyz[0] : (imsize[0] - 1) // 2 + 1 - xyz[0]
            ]

        else:
            xmesh = np.mgrid[0 - xyz[0] : imsize[0] // 2 + 1 - xyz[0]]

        rmesh = np.sqrt(xmesh * xmesh)

        amesh = np.zeros(xmesh.shape)

        n = 1  # Normalization factor

    if len(imsize) == 2:
        # The definition below is consistent with numpy np.fft.fftfreq and
        # np.fft.rfftfreq:

        if not rfft:
            [xmesh, ymesh] = np.mgrid[
                -imsize[0] // 2 + m[0] - xyz[0] : (imsize[0] - 1) // 2 + 1 - xyz[0],
                -imsize[1] // 2 + m[1] - xyz[1] : (imsize[1] - 1) // 2 + 1 - xyz[1],
            ]

        else:
            [xmesh, ymesh] = np.mgrid[
                -imsize[0] // 2 + m[0] - xyz[0] : (imsize[0] - 1) // 2 + 1 - xyz[0],
                0 - xyz[1] : imsize[1] // 2 + 1 - xyz[1],
            ]
            xmesh = np.fft.ifftshift(xmesh)

        rmesh = np.sqrt(xmesh * xmesh + ymesh * ymesh)

        amesh = np.arctan2(ymesh, xmesh)

        n = 2  # Normalization factor

    if len(imsize) == 3:
        # The definition below is consistent with numpy np.fft.fftfreq and
        # np.fft.rfftfreq:

        if not rfft:
            [xmesh, ymesh, zmesh] = np.mgrid[
                -imsize[0] // 2 + m[0] - xyz[0] : (imsize[0] - 1) // 2 + 1 - xyz[0],
                -imsize[1] // 2 + m[1] - xyz[1] : (imsize[1] - 1) // 2 + 1 - xyz[1],
                -imsize[2] // 2 + m[2] - xyz[2] : (imsize[2] - 1) // 2 + 1 - xyz[2],
            ]

        else:
            [xmesh, ymesh, zmesh] = np.mgrid[
                -imsize[0] // 2 + m[0] - xyz[0] : (imsize[0] - 1) // 2 + 1 - xyz[0],
                -imsize[1] // 2 + m[1] - xyz[1] : (imsize[1] - 1) // 2 + 1 - xyz[1],
                0 - xyz[2] : imsize[2] // 2 + 1 - xyz[2],
            ]
            xmesh = np.fft.ifftshift(xmesh)
            ymesh = np.fft.ifftshift(ymesh)

        rmesh = np.sqrt(xmesh * xmesh + ymesh * ymesh + zmesh * zmesh)

        amesh = np.arccos(zmesh / rmesh)

        n = 3  # Normalization factor

    if rounding:
        rmesh = np.round(rmesh)

    if normalize:

        a = np.sum(imsize * imsize)

        rmesh = rmesh / (np.sqrt(a) / np.sqrt(n))

    if nozero:
        # Replaces the "zero radius" by a small value to prevent division by zero in
        # other programs
        idx = rmesh == 0
        rmesh[idx] = nozeroval

    return rmesh, np.nan_to_num(amesh)


def CTF(
    imsize: tuple = (128, 128),
    df1: float = 50000.0,
    df2: float = None,
    ast: float = 0.0,
    ampcon: float = 0.07,
    Cs: float = 2.7,
    kV: float = 300.0,
    apix: float = 1.0,
    B: float = 0.0,
    rfft: bool = True,
):
    """
    Generates 1D, 2D or 3D contrast transfer function (CTF) of a TEM.

    Parameters
    ----------
    imsize : tuple
        The shape of the input ndarray.
    df1 : float
        Defocus 1 (or Defocus U in some notations) in Angstroms. Principal defocus \
axis. Underfocus is positive.
    df2 : float
        Defocus 2 (or Defocus V in some notations) in Angstroms. Defocus axis \
orthogonal to the U axis. Only mandatory for astigmatic data.
    ast : float
        Angle for astigmatic data (in degrees).
    ampcon : float
        Amplitude contrast fraction (between 0.0 and 1.0).
    Cs : float
        Spherical aberration (in mm).
    kV : float
        Acceleration voltage of the TEM (in kV).
    apix : float
        Input pixel size in Angstroms.
    B : float
        B-factor in Angstroms**2.
    rfft : bool
        Whether to return an array consistent with np.fft.rfftn i.e. exploiting the \
Hermitian symmetry of the Fourier transform of real data.

    Returns
    -------
    CTFim : ndarray
        Array containing the CTF.

    Notes
    -----
    Follows the CTF definition from Mindell & Grigorieff, JSB (2003) \
(https://doi.org/10.1016/S1047-8477(03)00069-8), which is adopted in FREALIGN/\
cisTEM, RELION and many other packages.
    """
    if not np.isscalar(imsize) and len(imsize) == 1:
        imsize = imsize[0]

    Cs *= 1e7  # Convert Cs to Angstroms

    if df2 is None or np.isscalar(imsize):
        df2 = df1


    # NOTATION FOR DEFOCUS1, DEFOCUS2, ASTIGMATISM BELOW IS INVERTED DUE TO NUMPY
    # CONVENTION:
    # df1, df2 = df2, df1

    ast *= -pi / 180.0

    WL = ElectronWavelength(kV)

    w1 = np.sqrt(1 - ampcon * ampcon)
    w2 = ampcon

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        if np.isscalar(imsize):
            if rfft:
                rmesh = np.fft.rfftfreq(imsize)
            else:
                rmesh = np.fft.fftfreq(imsize)
            amesh = 0.0

        else:
            rmesh, amesh = RadialIndices(imsize, normalize=True, rfft=rfft)

            rmesh2 = rmesh**2 / apix**2

        # From Mindell & Grigorieff, JSB 2003:
        df = 0.5 * (df1 + df2 + (df1 - df2) * np.cos(2.0 * (amesh - ast)))

        Xr = np.nan_to_num(pi * WL * rmesh2 * (df - 0.5 * WL * WL * rmesh2 * Cs))

    CTFim = -w1 * np.sin(Xr) - w2 * np.cos(Xr)

    if B != 0.0:  # Apply B-factor only if necessary:

        CTFim = CTFim * np.exp(-B * (rmesh2) / 4)

    return CTFim


def CorrectCTF(
    img=None,
    df1: float = 50000.0,
    df2: float = None,
    ast: float = 0.0,
    ampcon: float = 0.07,
    invert_contrast: bool = False,
    Cs: float = 2.7,
    kV: float = 300.0,
    apix: float = 1.0,
    phase_flip: bool = False,
    ctf_multiply: bool = False,
    wiener_filter: bool = False,
    C: float = 1.0,
    return_ctf: bool = False,
):
    """
    Applies different types of CTF correction to a 2D or 3D image.

    Parameters
    ----------
    img : 
        The input image to be corrected.
    df1 : float
        Defocus 1 (or Defocus U in some notations) in Angstroms. Principal defocus \
axis. Underfocus is positive.
    df2 : float
        Defocus 2 (or Defocus V in some notations) in Angstroms. Defocus axis \
orthogonal to the U axis. Only mandatory for astigmatic data.
    ast : float
        Angle for astigmatic data (in degrees).
    ampcon : float
        Amplitude contrast fraction (between 0.0 and 1.0).
    invert_contrast: bool
        Whether to invert the contrast of the input image.
    Cs : float
        Spherical aberration (in mm).
    kV : float
        Acceleration voltage of the TEM (in kV).
    apix : float
        Input pixel size in Angstroms.
    phase_flip :  bool
        Correct CTF by phase-flipping only. This corrects phases but leaves the \
amplitudes unchanged.
    ctf_multiply :  bool
        Correct CTF by multiplying the input image FT with the CTF. This corrects \
phases and dampens amplitudes even further near the CTF zeros.
    wiener_filter : bool
        Correct CTF by applying a Wiener filter. This corrects phases and attempts to \
restore amplitudes to their original values based on an ad-hoc spectral signal-to-noise\
ratio (SSNR) value. That means, at frequencies where the SNR is low amplitudes will be\
restored conservatively, while where the SNR is high these frequencies will be boosted.
    C : float
        Wiener filter constant (per frequency). A scalar can be given to use the same \
constant for all frequencies, whereas an 1D array (radial average) can be given to \
restore each frequency using a different SNR value.
    return_ctf : bool
        Whether to return an array containing the CTF itself alongside the corrected \
image.

    Returns
    -------
    CTFcor : list
        A list containing the corrected image(s), the CTF (if return_ctf is True) and \
a string indicating the type of correction(s) applied.

    Raises
    ------
    ValueError
        If any entry of the Wiener filter constant is lower than or equal to zero.

    Notes
    -----
    More than one type of CTF correction can be performed with one call, in which case \
all of the corrected versions will be returned consecutively as the first entries of \
the returned list. The last entries of the returned list are strings indicating the \
type of correction applied: "pf" for phase-flipping, "cm" for CTF multiplication and \
"wf" for Wiener filtering.
    """
    if df2 is None:
        df2 = df1

    # Direct CTF correction would invert the image contrast. By default we don't do
    # that, hence the negative sign:
    CTFim = -CTF(img.shape, df1, df2, ast, ampcon, Cs, kV, apix, 0.0, rfft=True)

    CTFcor = []
    cortype = []

    if invert_contrast:

        CTFim = -CTFim

        pass

    FT = np.fft.rfftn(img)

    if phase_flip:  # Phase-flipping
        s = np.sign(CTFim)
        CTFcor.append(np.fft.irfftn(FT * s))
        cortype.append("pf")

    if ctf_multiply:  # CTF multiplication
        CTFcor.append(np.fft.irfftn(FT * CTFim))
        cortype.append("cm")

    if wiener_filter:  # Wiener filtering
        if np.any(C <= 0.0):
            raise ValueError(
                "Error: Wiener filter contain value(s) less than or equal to zero!"
            )

        CTFcor.append(np.fft.irfftn(FT * CTFim / (CTFim * CTFim + C)))
        cortype.append("wf")

    if return_ctf:
        CTFcor.append(CTFim)

    CTFcor.append(cortype)

    return CTFcor


def AdhocSSNR(
    imsize: tuple = (128, 128),
    apix: float = 1.0,
    df: float = 50000.0,
    ampcon: float = 0.07,
    Cs: float = 2.7,
    kV: float = 300.0,
    S: float = 1.0,
    F: float = 1.0,
    hp_frac: float = 0.02,
    lp: bool = True,
):
    """
    An ad hoc SSNR model for cryo-EM data as proposed by Dimitry Tegunov [1,2].

    Parameters
    ----------
    imsize : tuple
        The shape of the input array.
    apix : float
        Input pixel size in Angstroms.
    df : float
        Average defocus in Angstroms.
    ampcon : float
        Amplitude contrast fraction (between 0.0 and 1.0).
    Cs : float
        Spherical aberration (in mm).
    kV : float
        Acceleration voltage of the TEM (in kV).
    S : float
        Strength of the deconvolution to be applied.
    F : float
        Strength of the SSNR falloff.
    hp_frac : float
        fraction of Nyquist frequency to be cut off on the lower end (since it will \
be boosted the most).
    lp : bool
        Whether to low-pass all information beyond the first zero of the CTF.

    Returns
    -------
    ssnr : ndarray
        Array containing the radial ad hoc SSNR.

    Notes
    -----
    This SSNR model ignores astigmatism.

    References
    ----------
    [1] Tegunov & Cramer, Nat. Meth. (2019). https://doi.org/10.1038/s41592-019-0580-y
    [2] https://github.com/dtegunov/tom_deconv/blob/master/tom_deconv.m
    """
    rmesh = RadialIndices(imsize, rounding=False, normalize=True, rfft=True)[0] / apix

    # The ad hoc SSNR exponential falloff
    falloff = np.exp(-100 * rmesh * F) * 10 ** (3 * S)

    # The cosine-shaped high-pass filter. It starts at zero frequency and reaches 1.0
    # at hp_freq (fraction of the Nyquist frequency)
    a = np.minimum(1.0, rmesh * apix / hp_frac)
    highpass = 1.0 - np.cos(a * pi / 2)

    if lp:
        # Ensure the filter will reach zero at the first zero of the CTF
        first_zero_res = FirstZeroCTF(df=df, ampcon=ampcon, Cs=Cs, kV=kV)
        a = np.minimum(1.0, rmesh / first_zero_res)

        lowpass = np.cos(a * pi / 2)

        # Composite filter
        ssnr = highpass * falloff * lowpass

    else:
        ssnr = highpass * falloff  # Composite filter

    return np.abs(ssnr)


def ElectronWavelength(kV: float = 300.0):
    """
    Calculates electron wavelength given acceleration voltage.

    Parameters
    ----------
    kV : float
        Acceleration voltage of the TEM (in kV).

    Returns
    -------
    WL : float
        A scalar value containing the electron wavelength in Angstroms.
    """
    WL = 12.2639 / np.sqrt(kV * 1e3 + 0.97845 * kV * kV)

    return WL


def FirstZeroCTF(
    df: float = 50000.0, ampcon: float = 0.07, Cs: float = 2.7, kV: float = 300.0
):
    """
    The frequency at which the CTF first crosses zero.

    Parameters
    ----------
    df : float
        Average defocus in Angstroms. 
    ampcon : float
        Amplitude contrast fraction (between 0.0 and 1.0).
    Cs : float
        Spherical aberration (in mm).
    kV : float
        Acceleration voltage of the TEM (in kV).

    Returns
    -------
    g : float
        A scalar containing the resolution in Angstroms corresponding to the first \
zero crossing of the CTF.

    Notes
    -----
    Finds the resolution at the first zero of the CTF
    Wolfram Alpha, solving for -w1 * sinXr - w2 * cosXr = 0
    https://www.wolframalpha.com/input/?i=solve+%CF%80*L*(g%5E2)*(d-1%2F(\
    2*(L%5E2)*(g%5E2)*C))%3Dn+%CF%80+-+tan%5E(-1)(c%2Fa)+for+g
    """
    Cs *= 1e7  # Convert Cs to Angstroms

    w1 = np.sqrt(1 - ampcon * ampcon)
    w2 = ampcon

    WL = ElectronWavelength(kV)

    g = np.sqrt(-2 * Cs * WL * np.arctan2(w2, w1) + 2 * pi * Cs * WL + pi) / (
        np.sqrt(2 * pi * Cs * df) * WL
    )

    return g
