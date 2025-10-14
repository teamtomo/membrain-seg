import logging
import os

from typer import Option
from .cli import cli

logging.basicConfig(level=logging.INFO)


@cli.command(name="process-uncertainty-map", no_args_is_help=True)
def process_uncertainty_map_cli(
    prediction_path: str = Option(  # noqa: B008
        ..., help="Path to the binary segmentation mask (.mrc) used for foreground detection."
    ),
    uncertainty_map_path: str = Option(  # noqa: B008
        ..., help="Path to the raw uncertainty map (.mrc) generated from TTA predictions."
    ),
    out_folder: str = Option(  # noqa: B008
        "./predictions", help="Directory to save the processed uncertainty map."
    ),
):
    """
    Process a TTA uncertainty map by filling eroded foreground regions and applying
    median smoothing to reduce noise. The processed uncertainty map is then saved as
    a new .mrc file with the suffix "_processed.mrc".

    Parameters
    ----------
    prediction_path : str
        Path to the binary segmentation mask (.mrc) used for determining foreground voxels.
    uncertainty_map_path : str
        Path to the raw uncertainty map (.mrc) produced by TTA inference.
    out_folder : str
        Output directory where the processed uncertainty map will be saved.

    Examples
    --------
    membrain process-uncertainty-map \
        --prediction-path <path-to-segmentation.mrc> \
        --uncertainty-map-path <path-to-uncertainty.mrc> \
        --out-folder <output-directory>
    """

    from membrain_seg.segmentation.process_uncertainty_map import process_uncertainty_map as _process_uncertainty_map

    print("Processing uncertainty map...")
    print("")
    print(
        "This may take a few minutes depending on tomogram size. "
        "If you are curious, learn more about test-time augmentation in the "
        "MemBrain v2 preprint:"
    )
    print("MemBrain v2: an end-to-end tool for the analysis of membranes in cryo-electron tomography")
    print("https://www.biorxiv.org/content/10.1101/2024.01.05.574336v1")
    print("")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    # Run the core processing
    _process_uncertainty_map(
        prediction_path=prediction_path,
        uncertainty_map_path=uncertainty_map_path,
    )

    logging.info("Processed uncertainty map saved in: " + os.path.dirname(uncertainty_map_path))
    print("✅ Done.")
