from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)
from membrain_seg.tomo_preprocessing.deconvolution.deconv_utils import (
    CorrectCTF,
    AdhocSSNR,
)


def deconvolve(
    mrcin: str,
    mrcout: str,
    DF1: float = 50000.0,
    DF2: float = None,
    AST: float = 0.0,
    ampcon: float = 0.07,
    Cs: float = 2.7,
    kV: float = 300.0,
    apix: float = None,
    strength: float = 1.0,
    falloff: float = 1.0,
    skip_lowpass: bool = False
) -> None:
    """
    Deconvolve the input tomogram using the Warp deconvolution filter. For the definition of the filter please see Tegunov & Cramer, Nat. Meth. (2019), https://doi.org/10.1038/s41592-019-0580-y

    Parameters
    ----------
    mrcin : str
        The file path to the input tomogram to be processed.
    mrcout : str
        The file path where the processed tomogram will be stored.
    DF1: float
        Defocus 1 (or Defocus U in some notations) in Angstroms. Principal defocus axis. Underfocus is positive.
    DF2: float
        Defocus 2 (or Defocus V in some notations) in Angstroms. Defocus axis orthogonal to the U axis. Only mandatory for astigmatic data.
    AST: float
        Angle for astigmatic data (in degrees).
    ampcon: float
        Amplitude contrast fraction (between 0.0 and 1.0).
    Cs: float
        Spherical aberration (in mm).
    kV: float
        Acceleration voltage of the TEM (in kV).
    apix: float
        Input pixel size (optional). If not specified, it will be read from the tomogram's header. ATTENTION: This can lead to severe errors if the header pixel size is not correct.
    strength: float
        Strength parameter for the denoising filter.
    falloff: float
        Falloff parameter for the denoising filter.
    skip_lowpass: bool
        The denoising filter by default will have a smooth low-pass effect that enforces filtering out any information beyond the first zero of the CTF. Use this option to skip this filter (i.e. potentially include information beyond the first CTF zero).


    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the file specified in `mrcin` does not exist.

    Notes
    -----
    This function reads the input tomogram and applies the deconvolution filter on it following the Warp implementation (see reference above), then stores the processed tomogram to the specified output path. The deconvolution process is
    controlled by several parameters including the tomogram defocus, acceleration voltage, spherical aberration, strength and falloff. The implementation here is based on that of the focustools package: https://github.com/C-CINA/focustools/
    """

    tomo = load_tomogram( mrcin )

    if apix == None:

        apix = tomo.voxel_size.x

    if DF2 == None:

        DF2 = DF1

    ssnr = AdhocSSNR(imsize=tomo.data.shape, apix=apix, DF=0.5 * (DF1 + DF2),
                                 WGH=ampcon, Cs=Cs, kV=kV, S=strength, F=falloff, hp_frac=0.01, lp=not skip_lowpass)

    wiener_constant = 1 / ssnr

    deconvtomo = CorrectCTF(tomo.data, DF1=DF1, DF2=DF2, AST=AST, WGH=ampcon, invert_contrast=False, Cs=Cs, kV=kV,
                            apix=apix, phase_flip=False, ctf_multiply=False, wiener_filter=True, C=wiener_constant, return_ctf=False)

    store_tomogram(mrcout, deconvtomo[0], voxel_size=apix)
