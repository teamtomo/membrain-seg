# How to create training annotations from your own dataset?

This is an example guide on how to create training data in order to improve MemBrain-seg's performance on your tomograms. 
The annotation strategy was developed as part of a [Helmholtz Imaging](https://helmholtz-imaging.de/) collaboration with [Fabian Isensee](https://helmholtz-imaging.de/person/dr-rer-nat-fabian-isensee/) & [Sebastian Ziegler](https://modalities.helmholtz-imaging.de/expert/46).


In parallel to this guide, please also check Simon Zufferey's [YouTube tutorial](https://www.youtube.com/playlist?list=PLV3O3yHyCjkXAi9MWgComzh6JuKcHUgNU) accompanying the instructions on this size. 

**Important note:** While this guide describes how to improve the performance for your own tomograms, we highly encourage you to also share your generated training patches with us and the community, so that everybody can benefit from a more generalized performance of the model.
We are currently working on a platform to easily share them. In the meantime, please reach out to us (Lorenz.Lamm@helmholtz-munich.de) to discuss how to best share your patches without giving away too much of your own data.

| | | |
|-|-|-|
| <img width="100%" src="https://user-images.githubusercontent.com/34575029/250343504-d8f11f76-7422-4085-854e-246ef9d90d89.gif"> | <img width="100%" src="https://user-images.githubusercontent.com/34575029/250345378-657063de-29b4-4f00-a11e-b3bc9f09a0d3.png"> | <img width="100%" src="https://user-images.githubusercontent.com/34575029/250343487-791513e9-0c01-4558-8c73-e58b8a9a0c7b.gif"> |


# General idea
Nobody would like (or has the time) to segment membranes a whole tomogram manually from scratch. Therefore, our approach is to extract small patches (160^3) from the tomogram an create manual annotations for these.

For this, we do not start from scratch, but use the prediction of the currently best MemBrain-seg model. Ideally, this will already segment most areas well and minimize the workload for corrections.

# Workflow
The steps described in this tutorial are:

1. [Which software to use](#software)
2. [Extraction of patches for correction](#patch-extraction)
3. [Performing the corrections](#corrections)
4. [Merging the corrections](#merging-of-corrections)

# Software
You will need software to inspect your tomograms and MemBrain-seg's segmentations, as well as to perform the corrections. For both of these tasks, we use [MITK Workbench](https://docs.mitk.org/nightly/MITKWorkbenchManualPage.html), but any software with these functionalities will do, e.g. Amira or Napari.

In our YouTube playlist, you can also find [a video on basic usage of MITK for patch correction](https://youtu.be/dhghgfO7Aoc).

# Patch extraction
In order to not have to correct the entire tomogram, we focus on small patches (160^3) where the segmentation performance is particularly bad. We crop these patches out of the tomogram and correct them manually.

In order to extract the patches from the tomogram, you can open the tomograms together with MemBrain's predicted segmentation, e.g. in MITK or IMOD. Now, you find regions where MemBrain's performance is not satisfying, but you can still see whether there should be a membrane or not. Use the center coordinates of these areas to extract patches (patches will be extracted centered around the x-, y-, and z-cooordintes given).

<p align="center" width="100%">
    <img width="35%" src="https://user-images.githubusercontent.com/34575029/250346977-cc7b9344-98d1-4845-9e60-942af3e75328.png">
</p>


Once you found all patches that you would like to extract (we recommend around 2-5 per tomogram), you can extract them using the following script

```
patch_corrections extract_patches
```
Running this command will open the help of this function and guide you through the required parameters.

Simon also describes this process of patch selection in his [first episode on YouTube](https://youtu.be/ilBYKQVGssQ).

# Corrections
The goal of the corrections is to assign every voxel in your extracted patch with the correct label (i.e. **"membrane"** or **"no membrane"**). However, each tomogram will probably have regions where it is very hard to tell where exactly the membrane is or if there is a membrane at all. In these cases, we want to use the **"ignore"** label. This label will not influence training of the U-Net in any direction, so whenever you are in doubt, it's best to assign the "ignore" label. **All voxels not assigned to the ignore label will contribute to the network training and should therefore be very reliable!**

## Correction workflow
The starting point for the generation of new training patches is the segmentation produced by the previous best MemBrain-seg segmentation.  
In order to use this segmentation for re-training the network, we need to make the annotations as good as we can. We do this by modifying the segmentation by creating different segmentation classes:

1. Starting point: MemBrain-seg segmentation
2. Subtract all "remove" annotations from the segmentation
3. Add all "add" annotations to the segmentation
4. Assign defined "ignore" labels

Note: Steps 1 to 4 are performed in the background when you merge your corrections with the command 
```
patch_corrections merge_corrections
```

## Folder structure
During the correction of the patches, you will generate different files ("add", "remove", and "ignore" labels). In order for them to be merged properly with the original segmentation, you should follow the following folder structure:

```
root_directory/
    ├── labels_dir/
    │   ├── label_file1
    │   ├── label_file2
    │   ├── label_file3
    │   └── ...
    ├── corrections_dir/
    │   ├── label_file1/
    │   │   ├── Add1.nrrd
    │   │   ├── Add2.nrrd
    │   │   ├── Remove1.nrrd
    │   │   ├── Ignore1.nrrd
    │   │   ├── Ignore2.nrrd
    │   │   ├── Ignore3.nrrd
    │   │   └── ...
    │   ├── label_file2/
    │   │   ├── Add1.nrrd
    │   │   ├── Add2.nrrd
    │   │   ├── Ignore1.nrrd
    │   │   └── ...
    │   ├── label_file3/
    │   │   ├── Add1.nrrd
    │   │   ├── Ignore1.nrrd
    │   │   └── ...
    │   └── ...
    └── out_dir/ (This directory will be filled with the corrected files
```

## Remove label
Whenever you find regions in your tomogram that MemBrain-seg segmented falsely (i.e., they actually are not a membrane), you should correct for this using the "remove" annotations.

For this, you create a segmentation layer in MITK and brush over all regions that are falsely annotated. (Can be coarse if no neighboring voxels belong to a membrane). Then, you can save the resulting segmentation as "Remove1.nrrd" (or replace 1 with the current number of your segmentation).

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/34575029/250078243-d49d27d1-0890-4837-a0a5-86aabba87d00.png">
</p>

Setting the "Remove" label is also explained in [Simon's YouTube video](https://youtu.be/diGTf4oSMMQ) about the remove label.

## Add label
For the "add" annotations, you look for areas in your patch where MemBrain-seg did not segment a membrane, even though the membrane is clearly visible.  
In these regions, you can now accurately delineate where the membrane is, i.e. you assign all voxels belonging to a membrane to the "add" annotation.

Similarly to the "remove" label, you should save the resulting segmentation as "Add1.nrrd".

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/34575029/250078228-6c6868fc-52d9-4885-89b4-b1958fc70a3f.png">
</p>

You can find visualizations of the "Add label" in this [YouTube video](https://youtu.be/F1ltG1DyZnA).

## Ignore label
The "ignore" annotation is used whenever you are not sure where exactly the membrane is or whether there is a membrane at all. In these cases, you can coarsely annotate these difficult regions. Thereby, you don't need to be very accurate and can coarsely capture the area.

Similarly to the "remove" and "add" label, you should save the resulting segmentation as "Ignore1.nrrd".

In the example below, one can see that the membrane should be closing somewhere, but it is not possible to exactly delineate where the membrane is going through. In these cases, it is best to assign the "ignore" label (purple)
<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/34575029/250078236-8f645295-0114-4dee-842f-14442c9961f2.png">
</p>

More details and examples of the "Ignore label" can be found in [this clip](https://youtu.be/3TEi8cubRyk).

# Merging of corrections
After [saving all your files with appropriate naming](https://youtu.be/6R-MqXUc8tA), you should check again that your saved corrections follow the [folder structure described above](#folder-structure).

Then, you can merge your corrections into training patches that can be used for re-training:

```
patch_corrections merge_corrections --labels-dir <path-to-your-labels-dir> --corrections-dir <path-to-your-corrections> --out-dir <out-directory>
```

Hereby, 
- "path-to-your-labels-dir" should be replaced with the folder that contains the labels of your extracted patches ("labels_dir" in above [folder structure](#folder-structure))
- "path-to-your-corrections" should be the folder containing all sub-folders for all patch corrections ("corrections_dir" in above [folder structure](#folder-structure))
- "out-directory" should be the folder where the merged corrections should be stored


# And now?
Unfortunately, we do not publicly provide our full training dataset yet, as it is still under development. But that should not stop you from having a model that works well on your tomograms.
Do not hesitate to reach out (Lorenz.Lamm@helmholtz-munich.de) and we will find a solution!

