from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)
from membrain_seg.tomo_preprocessing.deconvolution.deconv_utils import (
    AdhocSSNR,
    CorrectCTF,
)


def deconvolve(
    mrcin: str,
    mrcout: str,
    df1: float = 50000.0,
    df2: float = None,
    ast: float = 0.0,
    ampcon: float = 0.07,
    Cs: float = 2.7,
    kV: float = 300.0,
    apix: float = None,
    strength: float = 1.0,
    falloff: float = 1.0,
    hp_frac: float = 0.02,
    skip_lowpass: bool = True,
) -> None:
    """
    Deconvolve the input tomogram using the Warp deconvolution filter. 

    Parameters
    ----------
    mrcin : str
        The file path to the input tomogram to be processed.
    mrcout : str
        The file path where the processed tomogram will be stored.
    df1: float
        Defocus 1 (or Defocus U in some notations) in Angstroms. Principal defocus \
        axis. Underfocus is positive.
    df2: float
        Defocus 2 (or Defocus V in some notations) in Angstroms. Defocus axis \
        orthogonal to the U axis. Only mandatory for astigmatic data.
    ast: float
        Angle for astigmatic data (in degrees).
    ampcon: float
        Amplitude contrast fraction (between 0.0 and 1.0).
    Cs: float
        Spherical aberration (in mm).
    kV: float
        Acceleration voltage of the TEM (in kV).
    apix: float
        Input pixel size (optional). If not specified, it will be read from the \
        tomogram's header. ATTENTION: This can lead to severe errors if the header \
        pixel size is not correct.
    strength: float
        Strength parameter for the denoising filter.
    falloff: float
        Falloff parameter for the denoising filter.
    hp_frac : float
        fraction of Nyquist frequency to be cut off on the lower end (since it will \
        be boosted the most).
    skip_lowpass: bool
        The denoising filter by default will have a smooth low-pass effect that \
        enforces filtering out any information beyond the first zero of the CTF. Use \
        this option to skip this filter (i.e. potentially include information beyond \
        the first CTF zero).

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the file specified in `mrcin` does not exist.

    Notes
    -----
    This function reads the input tomogram and applies the deconvolution filter on it \
    following the Warp implementation (https://doi.org/10.1038/s41592-019-0580-y), then\
    stores the processed tomogram to the specified output path. The deconvolution \
    process is controlled by several parameters including the tomogram defocus, \
    acceleration voltage, spherical aberration, strength and falloff. The \
    implementation here is based on that of the focustools package: \
    https://github.com/C-CINA/focustools/
    """
    tomo = load_tomogram(mrcin)

    if apix is None:
        apix = tomo.voxel_size.x

    if df2 is None:
        df2 = df1

    print(
        "\nDeconvolving input tomogram:\n",
        mrcin,
        "\noutput will be written as:\n",
        mrcout,
        "\nusing:",
        f"\npixel_size: {apix:.3f}",
        f"\ndf1: {df1:.1f}",
        f"\ndf2: {df2:.1f}",
        f"\nast: {ast:.1f}",
        f"\nkV: {kV:.1f}",
        f"\nCs: {Cs:.1f}",
        f"\nstrength: {strength:.3f}",
        f"\nfalloff: {falloff:.3f}",
        f"\nhp_fraction: {hp_frac:.3f}",
        f"\nskip_lowpass: {skip_lowpass}\n",
    )
    print("Deconvolution can take a few minutes, please wait...")

    ssnr = AdhocSSNR(
        imsize=tomo.data.shape,
        apix=apix,
        df=0.5 * (df1 + df2),
        ampcon=ampcon,
        Cs=Cs,
        kV=kV,
        S=strength,
        F=falloff,
        hp_frac=hp_frac,
        lp=not skip_lowpass,
    )

    wiener_constant = 1 / ssnr

    deconvtomo = CorrectCTF(
        tomo.data,
        df1=df1,
        df2=df2,
        ast=ast,
        ampcon=ampcon,
        invert_contrast=False,
        Cs=Cs,
        kV=kV,
        apix=apix,
        phase_flip=False,
        ctf_multiply=False,
        wiener_filter=True,
        C=wiener_constant,
        return_ctf=False,
    )

    store_tomogram(mrcout, deconvtomo[0], voxel_size=apix)

    print("\nDone!")
