# How to create training annotations from your own dataset?

This is an example guide on how to create training data in order to improve MemBrain-seg's performance on your tomograms. 
The annotation strategy was developed as part of a [Helmholtz Imaging](https://helmholtz-imaging.de/) Collaboration with [Fabian Isensee](https://helmholtz-imaging.de/person/dr-rer-nat-fabian-isensee/) & [Sebastian Ziegler](https://modalities.helmholtz-imaging.de/expert/46).

**Important note:** While this guide describes how to imporve the performance for your own tomograms, we highly encourage you to also share your generated training patches with us and the community, so that everybody can benefit from a more generalized performance of the model.

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

# Patch extraction
In order to not have to correct the entire tomogram, we focus on small patches (160^3) where the segmentation performance is particularly bad. We crop these patches out of the tomogram and correct them manually.

In order to extract the patches from the tomogram, you can open them, e.g. in MITK

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

## Add label
For the "add" annotations, you look for areas in your patch where MemBrain-seg did not segment a membrane, even though the membrane is clearly visible.  
In these regions, you can now accurately delineate where the membrane is, i.e. you assign all voxels belonging to a membrane to the "add" annotation.

Similarly to the "remove" label, you should save the resulting segmentation as "Add1.nrrd".

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/34575029/250078228-6c6868fc-52d9-4885-89b4-b1958fc70a3f.png">
</p>

## Ignore label
The "ignore" annotation is used whenever you are not sure where exactly the membrane is or whether there is a membrane at all. In these cases, you can coarsely annotate these difficult regions. Thereby, you don't need to be very accurate and can coarsely capture the area.

Similarly to the "remove" and "add" label, you should save the resulting segmentation as "Ignore1.nrrd".

In the example below, one can see that the membrane should be closing somewhere, but it is not possible to exactly delineate where the membrane is going through. In these cases, it is best to assign the "ignore" label (purple)
<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/34575029/250078236-8f645295-0114-4dee-842f-14442c9961f2.png">
</p>

# Merging of corrections
Once you are done with all your corrections, you should check again that your saved corrections follow the [folder structure described above](#folder-structure).

Then, you can merge your corrections into training patches that can be used for re-training:

```
patch_corrections merge_corrections --labels-dir <path-to-your-labels-dir> --corrections-dir <path-to-your-corrections> --out-dir <out-directory>
```

Hereby, 
- "path-to-your-labels-dir" should be replaced with the folder that contains the labels of your extracted patches ("labels_dir" in above [folder structure](#folder-structure))
- "path-to-your-corrections" should be the folder containing all sub-folders for all patch corrections ("corrections_dir" in above [folder structure](#folder-structure))
- "out-directory" should be the folder where the merged corrections should be stored


<div style="justify-content: space-around;" align=center>
    <img style="vertical-align: middle; width: 35%;" src="https://user-images.githubusercontent.com/34575029/250343504-d8f11f76-7422-4085-854e-246ef9d90d89.gif">
    <img style="vertical-align: middle; width: 5%;" src="https://user-images.githubusercontent.com/34575029/250343549-4cc1fbaa-a642-4af2-a0b9-c82ac538af6d.png">
    <img style="vertical-align: middle; width: 35%;" src="https://user-images.githubusercontent.com/34575029/250343487-791513e9-0c01-4558-8c73-e58b8a9a0c7b.gif">
</div>