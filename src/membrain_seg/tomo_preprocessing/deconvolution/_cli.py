from typer import Option

from ..cli import OPTION_PROMPT_KWARGS as PKWARGS
from ..cli import cli
from .deconvolve import deconvolve as run_deconvolve


@cli.command(name="deconvolve", no_args_is_help=True)
def deconvolve(
    input: str = Option(  # noqa: B008
        None, help="Tomogram to deconvolve (.mrc/.rec format)", **PKWARGS
    ),
    output: str = Option(  # noqa: B008
        None,
        help="Output location for deconvolved tomogram (.mrc/.rec format)",
        **PKWARGS,
    ),
    pixel_size: float = Option(  # noqa: B008
        None,
        help="Input pixel size (optional). If not specified, it will be read from the \
        tomogram's header. ATTENTION: This can lead to severe errors if the header \
        pixel size is not correct.",
    ),
    df1: float = Option(  # noqa: B008
        50000,
        help="Defocus 1 (or Defocus U in some notations) in Angstroms. Principal \
        defocus axis. Underfocus is positive.",
    ),
    df2: float = Option(  # noqa: B008
        None,
        help="Defocus 2 (or Defocus V in some notations) in Angstroms. Defocus axis \
        orthogonal to the U axis. Only mandatory for astigmatic data.",
    ),
    ast: float = Option(  # noqa: B008
        0.0,
        help="Angle for astigmatic data (in degrees). Astigmatism is currently not \
        used in deconvolution (only the axis of largest defocus is considered), but \
        maybe some better model in the future will use it?",
    ),
    ampcon: float = Option(  # noqa: B008
        0.07,
        help="Amplitude contrast fraction (between 0.0 and 1.0).",
    ),
    cs: float = Option(  # noqa: B008
        2.7,
        help="Spherical aberration (in mm).",
    ),
    kv: float = Option(  # noqa: B008
        300.0,
        help="Acceleration voltage of the TEM (in kV).",
    ),
    strength: float = Option(  # noqa: B008
        1.0,
        help="Strength parameter for the denoising filter.",
    ),
    falloff: float = Option(  # noqa: B008
        1.0,
        help="Falloff parameter for the denoising filter.",
    ),
    hp_fraction: float = Option(  # noqa: B008
        0.02,
        help="Fraction of Nyquist frequency to be cut off on the lower end (since it \
        will be boosted the most)",
    ),
    skip_lowpass: bool = Option(  # noqa: B008
        False,
        help="The denoising filter by default will have a smooth low-pass effect that \
        enforces filtering out any information beyond the first zero of the CTF. Use \
        this option to skip this filter i.e. potentially include information beyond \
        the first CTF zero (not recommended).",
    ),
):
    """Deconvolve the input tomogram using the Warp deconvolution filter."""
    run_deconvolve(
        input,
        output,
        df1,
        df2,
        ast,
        ampcon,
        cs,
        kv,
        pixel_size,
        strength,
        falloff,
        hp_fraction,
        skip_lowpass,
    )
