import logging
from typing import Optional

from typer import Option

from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli

logging.basicConfig(level=logging.INFO)


@cli.command(name="download_data", no_args_is_help=True)
def download_data(
    out_folder: str = Option(  # noqa: B008
        ...,
        help="Folder to store the downloaded MemBrain-seg dataset (14GB .zip file).",
        **PKWARGS,
    ),
    num_parallel_processes: int = Option(  # noqa: B008
        help='Number of parallel processes to use for downloading the dataset.\
            Since dataset is hosted on Zenodo with limited bandwidth,\
            using multiple parallel processes can speed up the download.\
            Pass an integer value greater than 0. For example, to use 4 parallel\
            processes, pass "4". Default is "1" (single process).',
        default=1,
    ),
):
    """
    CLI for downloading the MemBrain-seg dataset.

    Downloads the MemBrain-seg benchmark dataset from Zenodo
    and saves it to the specified output folder.

    The dataset can be used to re-train MemBrain-seg, but also to
    benchmark MemBrain-seg against other segmentation methods.

    Parameters
    ----------
    out_folder : str
        Folder to store the downloaded MemBrain-seg dataset (14GB .zip file).
    num_parallel_processes : str
        Number of parallel processes to use for downloading the dataset.
        Since dataset is hosted on Zenodo with limited bandwidth,
        using multiple parallel processes can speed up the download.
        Pass an integer value greater than 0. For example, to use 4 parallel
        processes, pass "4". Default is "1" (single process).
    """
    from membrain_seg.benchmark.download_dataset import download_data as _download_data

    _download_data(out_folder=out_folder, num_parallel_processes=num_parallel_processes)


@cli.command(name="benchmark", no_args_is_help=True)
def benchmark(
    pred_path: str = Option(  # noqa: B008
        ...,
        help="Folder containing the predicted segmentation results to benchmark.\
            Should contain .mrc files with predicted segmentations per membrane patch.\
            See documentation for details.",
        **PKWARGS,
    ),
    gt_path: str = Option(  # noqa: B008
        ...,
        help="Folder containing the ground truth segmentations.\
            If not done yet, download the MemBrain-seg dataset using the\
            'download_data' command. ",
        **PKWARGS,
    ),
    out_dir: str = Option(  # noqa: B008
        ...,
        help="Output directory to store the benchmarking results (CSV file and plots).",
        **PKWARGS,
    ),
    out_file_token: Optional[str] = Option(  # noqa: B008
        "stats",
        help="Optional token to append to the output filenames.",
    ),
    skeletonization_method: Optional[str] = Option(  # noqa: B008
        "3D-NMS",
        help='Skeletonization method to use. Supported: "3D-NMS", "2D-skimage".',
    ),
):
    """
    CLI for benchmarking with MemBrain-seg's annotated dataset.

    Compares predicted segmentations against ground truth segmentations
    using Dice score and Surface-Dice score metrics.
    Saves the benchmarking results as a CSV file and generates plots.


    Parameters
    ----------
    pred_path : str
        Folder containing the predicted segmentation results to benchmark.
        Should contain .mrc files with predicted segmentations per membrane patch.
        See documentation for details.
    gt_path : str
        Folder containing the ground truth segmentations.
        If not done yet, download the MemBrain-seg dataset using the
        'download_data' command.
    out_dir : str
        Output directory to store the benchmarking results (CSV file and plots).
    out_file_token : Optional[str]
        Optional token to append to the output filenames.
    skeletonization_method : str
        Skeletonization method to use. Supported: "3D-NMS", "2D-skimage".

    Examples
    --------
    membrain benchmark --pred-path <path-to-predictions>
        --gt-path <path-to-ground-truth> --out-dir <output-directory>
        --out-file-token <optional-token>

    """
    from membrain_seg.benchmark.compute_stats import (
        compute_stats,
    )

    compute_stats(
        dir_pred=pred_path,
        dir_gt=gt_path,
        out_dir=out_dir,
        out_file_token=out_file_token,
        skeletonization_method=skeletonization_method,
    )
