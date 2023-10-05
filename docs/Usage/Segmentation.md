# Segmentation

Welcome to the central feature of MemBrain-seg: **The segmentation of your tomograms!**

This guide provides detailed instructions to perform segmentation on your tomograms using a pre-trained model.


## MemBrain-seg Workflow
1. (Optional) [Preprocessing](./Preprocessing.md)
    - pixel size matching
    - Fourier amplitude matching
2. **Predict segmentation**
3. (Optional) [Match pixel size](./Preprocessing.md#pixel-size-matching) of output segmentation

For the optional preprocessing steps, find more information [here](./Preprocessing.md)

## Preparations

For the prediction, you will basically need two files:

1. The **tomogram** you would like to segment.  
It may make sense to use a preprocessed tomogram.
2. A **pre-trained MemBrain segmentation model**


We recommend to use denoised (ideally Cryo-CARE<sup>1</sup>) tomograms for segmentation. However, our current best model is available for download [here](https://drive.google.com/file/d/1tSQIz_UCsQZNfyHg0RxD-4meFgolszo8/view?usp=sharing) and should also work on non-denoised data. Please let us know how it works for you.
If the given model does not work properly, you may want to try one of our previous versions:

Other (older) model versions:
- [v9 -- best model until 10th Aug 2023](https://drive.google.com/file/d/15ZL5Ao7EnPwMHa8yq5CIkanuNyENrDeK/view?usp=sharing)
- [v9b -- model for non-denoised data until 10th Aug 2023](https://drive.google.com/file/d/1TGpQ1WyLHgXQIdZ8w4KFZo_Kkoj0vIt7/view?usp=sharing)


Please note that our best model changes often, as we are still in the development phase. So you can check in from time to time and see whether the model improved.
If you have problems with the model, please write an email to lorenz.lamm@helmholtz-munich.de

```
[1] T. -O. Buchholz, M. Jordan, G. Pigino and F. Jug, "Cryo-CARE: Content-Aware Image Restoration for Cryo-Transmission Electron Microscopy Data," 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), Venice, Italy, 2019, pp. 502-506, doi: 10.1109/ISBI.2019.8759519.
```
## Prediction
Typing
```
membrain segment
```
will display the segmentation command line interface and show available options.

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/34575029/250504257-611ea864-48c5-4424-9bc8-7b30bcfbb39b.png">
</p>

For example, for the prediction, you only need to type

```shell
membrain segment --tomogram-path <path-to-your-tomo> --ckpt-path <path-to-your-model>
```

Running this will segment your tomogram, and store the resulting .mrc file into the ./predictions 
folder. If you would like to change this folder, you can simply specify another folder using
the `--out_folder` argument:

```shell
membrain segment --tomogram-path <path-to-your-tomo> --ckpt-path <path-to-your-model> --out-folder <your-preferred-folder>
```

It is now also possible to assign different labels to different membrane instances via computing connected components and also remove small connected components:
```shell
membrain segment --tomogram-path <path-to-your-tomo> --ckpt-path <path-to-your-model> --store-connected-components
```

You can also compute the connected components [after you have segmented your tomogram](#connected-components).


### more membrain segment arguments:
**--tomogram-path**: TEXT Path to the tomogram to be segmented [default: None] 

**--ckpt-path** TEXT Path to the pre-trained model checkpoint that should be used. [default: None] 

**--out-folder** TEXT Path to the folder where segmentations should be stored. [default: ./predictions]

**--store-probabilities / --no-store-probabilities**: Should probability maps be output in addition to segmentations? [default: no-store-probabilities]

**--store-connected-components / no-store-connected-components**: Should connected components of the segmentation be computed? [default: no-store-connected-components]  

**--connected-component-thres**: Threshold for connected components. Components smaller than this will be removed from the segmentation. [default: None]

**--test-time-augmentation / --no-test-time-augmentation**: Should 8-fold test time augmentation be used? If activated (default), segmentations tendo be slightly better, but runtime is increased.

**--segmentation-threshold**: Set a custom threshold for thresholding your membrane scoremap to increase / decrease segmented membranes (default: 0.0).

**--sliding-window-size** INTEGER Sliding window size used for inference. Smaller values than 160 consume less GPU, but also lead to worse segmentation results! [default: 160] 

**--help** Show this message and exit.     


### Note: 
MemBrain-seg automatically detects a CUDA-enabled GPU, if available, and will execute the segmentation on it. Using a GPU device is highly recommended to accelerate the segmentation process.

### Note#2: 
Running MemBrain-seg on a GPU requires at least roughly 8GB of GPU space.

### Emergency tip:
In case you don't have enough GPU space, you can also try adjusting the `--sliding-window-size` parameter. By default, it is set to 160. Smaller values will require less GPU space, but also lead to worse segmentation results!

## Connected components
If you have segmented your tomograms already, but would still like to extract the connected components of the segmentation, you don't need to re-do the segmentation, but can simply use the following command:
```shell
membrain components --segmentation-path <path-to-your-segmentation> --connected-component-thres 50 --out-folder <folder-to-store-components>
```
### Note: 
Computing the connected components, and particularly also removing the small components can be quite compute intensive and take a while.

## Custom thresholding
In some cases, the standard threshold ($0.0$) may not be the ideal value for segmenting your tomograms. In order to explore what threshold may be best, you can use the above segmentation command with the flag `--store-probabilities`. This will store a membrane scoremap that you can threshold using different values using the command:

```
membrain thresholds --scoremap-path <path-to-scoremap>
        --thresholds -1.5 --thresholds -0.5 --thresholds 0.0 --thresholds 0.5
```
In this way, you can pass as many thresholds as you would like and the function will output one segmentation for each.


## Post-Processing
If you have pre-processed your tomogram using pixel size matching, you may want to [rescale](./Preprocessing.md#pixel-size-matching) your 
segmentation back to the shape of the original tomogram.

