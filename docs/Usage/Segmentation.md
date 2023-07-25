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


We recommend to use denoised (ideally Cryo-CARE<sup>1</sup>) tomograms for segmentation. You can find a pre-trained segmentation model for denoised tomograms [here](https://drive.google.com/file/d/15ZL5Ao7EnPwMHa8yq5CIkanuNyENrDeK/view?usp=sharing). 

In case you don't have denoised tomograms available, you can also use [this model](https://drive.google.com/file/d/1TGpQ1WyLHgXQIdZ8w4KFZo_Kkoj0vIt7/view?usp=sharing). It performs much better on non-denoised data, and maintains a good performance on denoised tomograms.

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
membrain segment --tomogram-path <path-to-your-tomo> --ckpt-path <path-to-your-model> --store-connected-components --connected-component-thres 50
```



### more membrain segment arguments:
**--tomogram-path**: TEXT Path to the tomogram to be segmented [default: None] 

**--ckpt-path** TEXT Path to the pre-trained model checkpoint that should be used. [default: None] 

**--out-folder** TEXT Path to the folder where segmentations should be stored. [default: ./predictions]

**--store-probabilities / --no-store-probabilities**: Should probability maps be output in addition to segmentations? [default: no-store-probabilities]

**--store-connected-components / no-store-connected-components**: Should connected components of the segmentation be computed? [default: no-store-connected-components]  

**--connected-component-thres**: Threshold for connected components. Components smaller than this will be removed from the segmentation. [default: None]

**--sliding-window-size** INTEGER Sliding window size used for inference. Smaller values than 160 consume less GPU, but also lead to worse segmentation results! [default: 160] 

**--help**                                                                      Show this message and exit.     


### Note: 
MemBrain-seg automatically detects a CUDA-enabled GPU, if available, and will execute the segmentation on it. Using a GPU device is highly recommended to accelerate the segmentation process.

### Note#2: 
Running MemBrain-seg on a GPU requires at least roughly 8GB of GPU space.

### Emergency tip:
In case you don't have enough GPU space, you can also try adjusting the `--sliding-window-size` parameter. By default, it is set to 160. Smaller values will require less GPU space, but also lead to worse segmentation results!

## Post-Processing
If you have pre-processed your tomogram using pixel size matching, you may want to [rescale](./Preprocessing.md#pixel-size-matching) your 
segmentation back to the shape of the original tomogram.

