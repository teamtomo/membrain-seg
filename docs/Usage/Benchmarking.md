# Benchmarking

Our well-annotated dataset can not only be used to train a model, but also the benchmark other models trained on different datasets.
To do so, we provide a script that can be used to evaluate the performance of a model on our dataset.

## Usage
### Preparations
1. Have the MemBrain-seg dataset ready, i.e. the imagesTest and labelsTest folders.
2. Perform predictions on the imagesTest folder using your model. Store the predictions in a folder of your choice.

**Important**: To match the predictions and ground truth labels, the predictions should have the same name as the input patch, respectively, but without the "_0000"-suffix.
E.g. if the input patch is called "patch_0000.nii.gz", the prediction should be called "patch.nii.gz" (but also .mrc is possible).

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






