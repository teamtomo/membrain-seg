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

A pre-trained MemBrain segmentation model can be downloaded [here](https://drive.google.com/file/d/15ZL5Ao7EnPwMHa8yq5CIkanuNyENrDeK/view?usp=sharing). 
Please note that our best model changes often, as we are still in the development phase. So you can check in from time to time and see whether the model improved.
If you have problems with the model, please write an email to lorenz.lamm@helmholtz-munich.de

## Prediction
Typing
```
membrain segment
```
will display the segmentation command line interface and show available options.

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

Note: MemBrain-seg automatically detects a CUDA-enabled GPU, if available, and will execute the segmentation on it. Using a GPU device is highly recommended to accelerate the segmentation process.

## Post-Processing
If you have pre-processed your tomogram using pixel size matching, you may want to [rescale](./Preprocessing.md#pixel-size-matching) your 
segmentation back to the shape of the original tomogram.

