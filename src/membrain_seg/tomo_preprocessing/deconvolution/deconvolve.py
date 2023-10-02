from membrain_seg.segmentation.dataloading.data_utils import (
    load_tomogram,
    store_tomogram,
)
from membrain_seg.tomo_preprocessing.deconvolution import (
    deconv_utils_cpu,
    deconv_utils_gpu,
)


def deconvolve(
    mrcin: str,
    mrcout: str,
    df: float = 50000.0,
    ampcon: float = 0.07,
    Cs: float = 2.7,
    kV: float = 300.0,
    apix: float = None,
    strength: float = 1.0,
    falloff: float = 1.0,
    hp_frac: float = 0.02,
    skip_lowpass: bool = True,
    cpu: bool = False,
) -> None:
    """
    Deconvolve the input tomogram using the Warp deconvolution filter. 

    Parameters
    ----------
    mrcin : str
        The file path to the input tomogram to be processed.
    mrcout : str
        The file path where the processed tomogram will be stored.
    df: float
        "The defocus value to be used for deconvolution, in Angstroms. This is \
typically the defocus of the zero tilt. Underfocus is positive."
    ampcon: float
        Amplitude contrast fraction (between 0.0 and 1.0).
    Cs: float
        Spherical aberration (in mm).
    kV: float
        Acceleration voltage of the TEM (in kV).
    apix: float
        Input pixel size (optional). If not specified, it will be read from the \
tomogram's header. ATTENTION: This can lead to severe errors if the header pixel \
size is not correct.
    strength: float
        Strength parameter for the denoising filter.
    falloff: float
        Falloff parameter for the denoising filter.
    hp_frac : float
        fraction of Nyquist frequency to be cut off on the lower end (since it will \
be boosted the most).
    skip_lowpass: bool
        The denoising filter by default will have a smooth low-pass effect that \
enforces filtering out any information beyond the first zero of the CTF. Use this \
option to skip this filter (i.e. potentially include information beyond the first CTF \
zero).
    cpu: bool
        Use CPU for computations. Helpful if running out of memory on the GPU.

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
following the Warp implementation (https://doi.org/10.1038/s41592-019-0580-y), then \
stores the processed tomogram to the specified output path. The deconvolution process \
is controlled by several parameters including the tomogram defocus, acceleration \
voltage, spherical aberration, strength and falloff. The implementation here is based \
on that of the focustools package: https://github.com/C-CINA/focustools/
    """
    tomo = load_tomogram(mrcin)

    if apix is None:
        apix = tomo.voxel_size.x

    # if df2 is None:
    #     df2 = df1

    print(
        "\nDeconvolving input tomogram:\n",
        mrcin,
        "\noutput will be written as:\n",
        mrcout,
        "\nusing:",
        f"\npixel_size: {apix:.3f}",
        f"\ndf: {df:.1f}",
        f"\nkV: {kV:.1f}",
        f"\nCs: {Cs:.1f}",
        f"\nstrength: {strength:.3f}",
        f"\nfalloff: {falloff:.3f}",
        f"\nhp_fraction: {hp_frac:.3f}",
        f"\nskip_lowpass: {skip_lowpass}",
        f"\ncpu: {cpu}\n",
    )
    print("Deconvolution can take a few minutes, please wait...")

    if cpu:
        ssnr = deconv_utils_cpu.AdhocSSNR(
            imsize=tomo.data.shape,
            apix=apix,
            df=df,
            ampcon=ampcon,
            Cs=Cs,
            kV=kV,
            S=strength,
            F=falloff,
            hp_frac=hp_frac,
            lp=not skip_lowpass,
        )

        wiener_constant = 1 / ssnr

        deconvtomo = deconv_utils_cpu.CorrectCTF(
            tomo.data,
            df1=df,
            ast=0.0,
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

    else:
        ssnr = deconv_utils_gpu.AdhocSSNR(
            imsize=tomo.data.shape,
            apix=apix,
            df=df,
            ampcon=ampcon,
            Cs=Cs,
            kV=kV,
            S=strength,
            F=falloff,
            hp_frac=hp_frac,
            lp=not skip_lowpass,
        )

        wiener_constant = 1 / ssnr

        deconvtomo = deconv_utils_gpu.CorrectCTF(
            tomo.data,
            df1=df,
            ast=0.0,
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

    store_tomogram(mrcout, deconvtomo, voxel_size=apix)

    print("\nDone!")
