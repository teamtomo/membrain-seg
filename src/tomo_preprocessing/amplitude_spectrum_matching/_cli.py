from membrain_seg.dataloading.data_utils import load_tomogram
from typer import Option

from ..cli import OPTION_PROMPT_KWARGS as PKWARGS
from ..cli import cli
from .extract_spectrum import extract_spectrum_from_file as _extract_spectrum
from .match_spectrum import match_amplitude_spectrum_for_files


@cli.command(name="extract_spectrum", no_args_is_help=True)
def extract(
    input_path: str = Option(  # noqa: B008
        ..., help="Tomogram path to extract spectrum from (.mrc/.rec format)", **PKWARGS
    ),
    output_path: str = Option(  # noqa: B008
        ..., help="Output destination for extracted spectrum (.tsv format)", **PKWARGS
    ),
):
    """Extracts the radially averaged amplitude spectrum from the input tomogram."""
    # Call your function here
    input_tomo = load_tomogram(input_path)
    _extract_spectrum(input_tomo, output_path)


@cli.command(name="match_spectrum", no_args_is_help=True)
def match_spectrum(
    input: str = Option(  # noqa: B008
        None, help="Tomogram to match (.mrc/.rec)", **PKWARGS
    ),
    target: str = Option(  # noqa: B008
        None, help="Target spectrum to match the input tomogram to (.tsv)", **PKWARGS
    ),
    output: str = Option(  # noqa: B008
        None, help="Output location for matched tomogram", **PKWARGS
    ),
    cutoff: int = Option(  # noqa: B008
        False,
        help="Lowpass cutoff to apply. All frequencies above this value will be \
set to zero.",
    ),
    shrink_excessive_value: int = Option(  # noqa: B008
        5e1,
        help="Regularization for excessive values. All Fourier coefficients above \
this values will be set to the value.",
    ),
    almost_zero_cutoff: bool = Option(  # noqa: B008
        True,
        help='Pass "True" or "False". Should Fourier coefficients close to zero be \
ignored? Recommended particularly in combination with pixel size matching. \
Defaults to True.',
    ),
    smoothen: float = Option(  # noqa: B008
        10,
        help="Smoothening to apply to lowpass filter. Value roughly resembles sigmoid \
width in pixels",
    ),
):
    """Match the input tomogram's spectrum to the target spectrum."""
    # Call your function here
    match_amplitude_spectrum_for_files(
        input,
        target,
        output,
        cutoff,
        smoothen,
        almost_zero_cutoff,
        shrink_excessive_value,
    )
