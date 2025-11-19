# Benchmarking

Our well-annotated dataset can not only be used to train a model, but also the benchmark other models trained on different datasets.
To do so, we provide a script that can be used to evaluate the performance of a model on our dataset.

## Usage
### Preparations
1. Have the MemBrain-seg dataset ready, i.e. the imagesTest and labelsTest folders. For 1-line download, see [below](#downloading-the-dataset).
2. Perform predictions on the imagesTest folder using your model. Store the predictions in a folder of your choice.

**Important**: To match the predictions and ground truth labels, the predictions should have the same name as the input patch, respectively, but without the "_0000"-suffix.
E.g. if the input patch is called "patch_0000.nii.gz", the prediction should be called "patch.nii.gz" (but also .mrc is possible; can sometimes lead to flipped tomograms, be careful!).

### Running the benchmarking script
The benchmarking script does not provide. Instead, it is a Python function that you can integrate into your own code:
    
    ```python
    from membrain_seg.benchmark.compute_stats import compute_stats

    dir_gt = "path/to/ground_truth"
    dir_pred = "path/to/predictions"
    out_dir = "path/to/output"
    out_file_token = "stats"

    compute_stats(dir_gt, dir_pred, out_dir, out_file_token)
    ```

This will compute the statistics for the segmentations on the entire dataset and store the results in the specified output directory.
As metrics, the script computes the surface dice and the dice score for each segmentation. To learn more about the surface dice, please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2024.01.05.574336v1).

Alternatively, you can also run the benchmarking script from the command line using the following command:

```shell
membrain benchmark --pred-path <path-to-your-predictions> --gt-path <path-to-ground-truth> --out-folder <path-to-output-folder> --out-file-token <output-file-token>
```
#### Arguments:
**--pred-path:** Path to the folder containing the predicted segmentations.
**--gt-path:** Path to the folder containing the ground truth segmentations. 
**--out-folder:** Path to the folder where the benchmarking results should be stored. 
**--out-file-token:** Token to be added to the output file name. [default: stats]
**--skeletonization-method:** Skeletonization method to use. Supported methods are "3D-NMS" (default) and "2D-skimage". 3D-NMS is supposed to be a bit more accurate, especially for horizontal densities, while 2D-skimage is much faster. [default: 3D-NMS]


### Downloading the dataset
To download the MemBrain-seg benchmarking dataset, you can use the following command:
```shell
membrain download_data --out-folder <path-to-download-folder> --num-parallel-processes <number-of-parallel-processes>
``` 
This will download the dataset to the specified folder. The `--num-parallel-processes` argument allows you to specify how many parallel processes should be used for downloading the dataset. Since the dataset is hosted on Zenodo with limited bandwidth, using multiple parallel processes can speed up the download.
The default value is 1, i.e. single-process download.
The dataset consists of four folders, but all data is contained in the two folders: `imagesTr` and `labelsTr`, which contain the training images and the corresponding ground truth labels, respectively. For the purpose of benchmarking, you can use the ìmagesTr` and `labelsTr` folders as `imagesTest` and `labelsTest`, respectively.







