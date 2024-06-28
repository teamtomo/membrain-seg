# Preprocessing

## Introduction

This is a quick guide on how to use the tomo_preprocessing module for MemBrain-seg. 

In order to improve the segmentation performance of MemBrain-seg on your tomograms,
it may be beneficial to first perform some preprocessing to "normalize" the tomograms
to similar styles as the training data.

This module currently allows you to use the following preprocessing methods:

- **Pixel size matching**: Rescaling of your tomogram to a similar pixel size as the training data
- **Fourier amplitude matching**: Rescaling of Fourier components to pronounce different features in the tomograms (adapted from [DeePiCt](https://github.com/ZauggGroup/DeePiCt))
- **Deconvolution**: Deconvolution filter to enhance tomogram contrast (described in the [Warp publication](https://www.nature.com/articles/s41592-019-0580-y).)

## Table of Contents
- [When to use what?](#when-to-use-what)
- [Usage](#usage)
  - [Available Commands](#available-commands)
  - [Pixel Size Matching](#pixel-size-matching)
  - [Fourier Amplitude Matching](#fourier-amplitude-matching)


## When to use what?

We are still exploring when it makes sense to use which preprocessing technique. But here are 
already some rules of thumb:

1. Whenever your pixel sizes differs by a lot from around 10-12&Aring; / pixel, you should consider using pixel size matching. We recommend to match to a pixel size of 10&Aring;. <br> It is also possible to do this rescaling on-the-fly, see our [segmentation instructions](https://teamtomo.org/membrain-seg/Usage/Segmentation/#on-the-fly-rescaling).
2. The Fourier amplitude matching only works in some cases, depending on the CTFs of input 
and target tomograms. Our current recommendation is: If you're not satisfied with MemBrain's 
segmentation performance, why not give the amplitude matching a shot?
3. Deconvolution: This can make sense if your input tomogram has a very low signal-to-noise ratio. We still recommend [Cryo-CARE](https://github.com/juglab/cryoCARE_pip) as a denoising method, but this deconvolution can provide an easy-to-use alternative.


## Usage
You can control all commands of this preprocessing module by typing `tomo_preprocessing`+ some options.  
To view all available commands, use:

```shell
tomo_preprocessing --help
```

For help on a specific command, use:

```shell
tomo_preprocessing <command> --help
```

### **Available Commands**


- **match_pixel_size**: Tomogram rescaling to specified pixel size. Example:  
```shell
tomo_preprocessing match_pixel_size --input-tomogram <path-to-tomo> --output-path <path-to-output> --pixel-size-out 10.0 --pixel-size-in <your-px-size>`
```
- **match_seg_to_tomo**: Segmentation rescaling to fit to target tomogram's shape. Example:  
```shell
tomo_preprocessing match_seg_to_tomo --seg-path <path-to-seg> --orig-tomo-path <path-to-tomo> --output-path <path-to-output>`
```
- **extract_spectrum**: Extracts the radially averaged amplitude spectrum from the input tomogram. Example:  
```shell
tomo_preprocessing extract_spectrum --input-path <path-to-tomo> --output-path <path-to-output>
```
- **match_spectrum**: Match amplitude of Fourier spectrum from input tomogram to target spectrum. Example:  
```shell
tomo_preprocessing match_spectrum --input <path-to-tomo> --target <path-to-spectrum> --output <path-to-output>
```
- **deconvolve**: Perform the deconvolution filter to enhance tomogram contrast. Example:
```shell
tomo_preprocessing deconvolve --input <path-to-tomo> --output <path-to-output> --pixel-size <your-pixel-size>
```

### **Pixel Size Matching**
Pixel size matching is recommended when your tomogram pixel sizes differs strongly from the training pixel size range (roughly 10-14&Aring;). <br>
**IMPORTANT NOTE**: MemBrain-seg can now also perform the rescaling on-the-fly during segmentation, making the below worklow redundant if you are not interested in the rescaled tomograms. You can check the on-the-fly rescaling at our [segmentation instructions](https://teamtomo.org/membrain-seg/Usage/Segmentation/#on-the-fly-rescaling)

If you prefer to not do it on-the-fly, you can perform the pixel size matching using the command

```shell
tomo_preprocessing match_pixel_size --input-tomogram <path-to-tomo> --output-path <path-to-output> --pixel-size-out 10.0 --pixel-size-in <your-px-size>
```

after adjusting the paths to your respective tomograms.
Afterwards, you can perform MemBrain's segmentation on the rescaled tomogram (i.e. the one specified in `--output-path`).  
Now, this new segmentation does not have the same shape as the original non-pixel-size-matched tomogram. To rescale the new segmentation to the original tomogram again, you can use

```shell
tomo_preprocessing match_seg_to_tomo --seg_path <path-to-seg> --orig-tomo-path <path-to-tomo> --output-path <path-to-output>
```

where the `--seg-path`is the segmentation created by MemBrain and the `--orig-tomo-path`is the original tomogram before rescaling to the new pixel size.  
The output of this function will be MemBrain's segmentation, but matched to the pixel size of the original tomogram.


### **Fourier Amplitude Matching**
Fourier amplitude matching is performed in two steps:

1. Extraction of the target Fourier spectrum:  
```shell
tomo_preprocessing extract_spectrum --input-path <path-to-tomo> --output-path <path-to-output>
```  
This extracts the radially averaged Fourier spectrum and stores it into a .tsv file.
2. Matching of the input tomogram to the extracted spectrum:  
```shell
tomo_preprocessing match_spectrum --input <path-to-tomo> --target <path-to-spectrum> --output <path-to-output>
```  
Now, the input tomograms Fourier components are re-scaled based on the equalization kernel computed from the input tomogram's radially averaged Fourier intensities, and the previously extracted .tsv file.


### **Deconvolution**

Deconvolution can be applied in a single step as preprocesing before performing the segmentation using the command 
```shell
tomo_preprocessing deconvolve --input <path-to-tomo> --output <path-to-output> --pixel-size <your-pixel-size>
```