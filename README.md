# Spinal-Rootlets Segmentation on T2w Images

## 1) Project Overview

The goal of this project is to develop a deep learning (DL)-based method to segment and locate the spinal rootlets.

## 2) Work Done

### A) Literature and Data Review

No articles were found on this topic.

After reviewing the available
datasets ([issue#1](https://github.com/ivadomed/model-spinal-rootlets/issues/1#issue-1706345176)), it was determined
that none of them were suitable.

### B) Dataset Creation

Initially, the plan was to test with a binary label (0: no rootlet, 1: rootlet). In the following text, the datasets
will be numbered as D1, D2, etc., and the models as M1, M2, etc.

Datasets summary:

| name | number images          | link                                                                   | labels         |
|------|------------------------|------------------------------------------------------------------------|----------------|
| D0a  | 30(10x3 sessions)      | [open neuro] ( https://openneuro.org/datasets/ds004507/versions/1.0.1) | No             |
| D0b  | 267                    | [spine-generic] ( https://github.com/spine-generic/data-multi-subject) | No             |
| D1a  | 12                     |                                                                        | Binary         |
| D1b  | 18                     |                                                                        | Binary         |
| D2   | 38 (20 from D0b + D1b) |                                                                        | Binary         |
| D3   | 31                     |                                                                        | Level specific |


#### D1a)
An initial labeling of 12 subjects (D1a) was performed.
Train one fold of the nnUNetV2 3d_fullres model (refer to [issue#5](https://github.com/ivadomed/model-spinal-rootlets/issues/5)). 
The resulting model (M1a) achieved a Dice score of approximately 0.52.

#### D1b)
After conducting inference on the entire dataset and performing manual corrections, the dataset was expanded to include
20 subjects. However, two subjects were subsequently excluded due to unsatisfactory label quality (
see [issue#7](https://github.com/ivadomed/model-spinal-rootlets/issues/7)), resulting in an 18-subject dataset (D1b). 
Full training on 5 folds with 250 epochs resulted in a Dice score of approximately 0.58 (M1b).

#### D2)
Inference was further conducted on T2w data from 267 subjects from
the [spinegeneric](https://github.com/spine-generic/data-multi-subject#spine-generic-public-database-multi-subject)
dataset (D0b) ([release r20230223](https://github.com/spine-generic/data-multi-subject/tree/r20230223), which had an
isotropic voxel size of 0.8mm. Some of the predictions (20) from this inference were
manually corrected, and the resulting dataset was merged with the previous dataset (D1b), resulting in a final dataset
containing 38 subjects (D2). 
Full training on 5 folds with 1000 epochs resulted in a Dice score of approximately 0.77 (M2) [issue#8](https://github.com/ivadomed/model-spinal-rootlets/issues/8).

#### D3)
The final dataset (D2) has been modified by changing voxel values according to the spinal level resulting in dataset D3.
An initial
training of 5 folds with 1000 epochs has been done. 
Reviewing the results and manually correcting the dataset (sct_maths -denoise 1 used to help labeling this time) improved the ground truth quality. 
Full training on 5 folds (2000 epochs) 

### C) Reproduce

In the next section, all the instructions to reproduce the label files used in the final dataset will be described.
However, label files are also available here (give to dataset with root_label).

nnUNet is used to train model but the dataset format is not BIDS. I have created/modified 4 script
available [here](https://github.com/ivadomed/utilities):

- From BIDS to nnUNet #new link after PR merged
- From nnUNet to BIDS #new link after PR merged
- Extract all image from bids to nnUNet inference #new link after PR
- Merge nnUNet dataset #add link after PR merged

On [this repo](https://github.com/ivadomed/utilities) there is also help to run nnUNet commands. 

#How to ewplain FLSEYES labeling ?

#### i) Reproduce D1a, M1a and D1b, M1b

Clone the original dataset D0a

```
git clone https://github.com/OpenNeuroDatasets/ds004507.git
```

This dataset is composed of 10 subject with 3 session per subject. Each session have a different neck position Up, Down,
Normal. We will not use Down position because nerve rootlets are really hard to see on this type of neck flexion.

Linked to [issue#5](https://github.com/ivadomed/model-spinal-rootlets/issues/5)

With FSLeyes, manually segment the following files:
<details>
<summary>12 first images to label</summary>
```
sub-002_ses-headNormal_T2w_root-manual.nii.gz	
sub-002_ses-headUp_T2w_root-manual.nii.gz	
sub-003_ses-headNormal_T2w_root-manual.nii.gz
sub-003_ses-headUp_T2w_root-manual.nii.gz
sub-004_ses-headNormal_T2w_root-manual.nii.gz	
sub-004_ses-headUp_T2w_root-manual.nii.gz
sub-005_ses-headNormal_T2w_root-manual.nii.gz
sub-005_ses-headUp_T2w_root-manual.nii.gz
sub-006_ses-headNormal_T2w_root-manual.nii.gz
sub-006_ses-headUp_T2w_root-manual.nii.gz
sub-007_ses-headNormal_T2w_root-manual.nii.gz
sub-007_ses-headUp_T2w_root-manual.nii.gz
```
</details>

> You can use the `json_write.py` script to add the json file according to the .nii.gz file created

Now convert this BIDS dataset to a nnUNet dataset `link to command script`.
This is the D1a dataset (100% train image no test image), composed of 12 images

Add the
<details>
<summary>dataset.json</summary>
```
{
    "channel_names": {
        "0": "T2w"
    },
    "labels": {
        "background": 0,
        "label": 1
    },
    "numTraining": 12,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"
}
```
</details>

Train model D1a with : `CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASETID 3d_fullres 0`

> You can stop when the progress.png reach a plateau (approx 250)

Out nnUNet Dice score from `progress.png` was around 0.52. 
Now extract all image from D0a with `command line extract`. 

Predict all the segmentation of D0a dataset with the model M1a
with `nnUNetv2_predict -i PATH_TO:imagesTs -o PATH_TO:Out_directory -d DATASETID -c 3d_fullres --save_probabilities -chk checkpoint_best.pth`

Manually review the predicted labels.
Note: subjects `sub-006-headNormal` and `sub-009-headNormal` have been dropped since they did not satisfy the quality.


Linked to [issue#7](https://github.com/ivadomed/model-spinal-rootlets/issues/7)
For training add
<details>
<summary>dataset.json</summary>
```
{
    "channel_names": {
        "0": "T2w"
    },
    "labels": {
        "background": 0,
        "label": 1
    },
    "numTraining": 18,
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"
}
```
</details>

Now you have a dataset with 18 subject we call this one D1b

Train nnUNet model M1b with `CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASETID -tr nnUNetTrainer_250epochs -f 0`, repeat
for fold 1, 2, 3, 4.

Out nnUNet Dice score from `progress.png` was between 0.52 and 0.6. 


#### ii) Reproduce D2, M2

Linked to [issue#8 part 2)](https://github.com/ivadomed/model-spinal-rootlets/issues/8)

Clone the original dataset D0b

```
git clone git@github.com:spine-generic/data-multi-subject.git
#specify version
```

Extract all T2w images with `extract command line`

Predict all the segmentation of D0b dataset with the model M1b
with `nnUNetv2_predict -i PATH_TO:imagesTs -o PATH_TO:Out_directory -d DATASETID -tr nnUNetTrainer_250epochs -c 3d_fullres --save_probabilities -f 0 1 2 3 4`

With FSLeyes, manually correct the following files:
<details>
<summary>12 first images to label</summary>
```
XXX
```
</details>

> I skipped some center because the quality was not good enough to ensure a good manual correction.

#convert to BIDS ? 

Merge with D1b to create D2, take mri `XX` and `XX`and put them into `imagesTs` and `labelsTs`

Train nnUNet model M2 with `CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASETID -f 0`, repeat
for fold 1, 2, 3, 4.

Out nnUNet Dice score from `progress.png` was between GET VALUE ON DUKE.

Since this point we introduce new metric to evaluate the model more detail in [issue#8](https://github.com/ivadomed/model-spinal-rootlets/issues/8): 
- Z-axis F1 score
- Mean common F1 score

#### iii) Reproduce D3, M3

Linked to [issue#8 part 3)](https://github.com/ivadomed/model-spinal-rootlets/issues/8)

Before we used a binary labeling. But some spinal level are overlapping. One of the solution is to label spinal rootlets depending on their spinal level (C2->2 .. T1->9). 
I have manually corrected and change the value of segmentation of the following files: 
<details>
<summary>31 spinal level specific value</summary>
```
XXX
```
</details>

This dataset D3 composed of 33 images with 31 for train . 
I have trained 4 folds of a nnUNet 3d_fullres model
Out nnUNet Dice score from `progress.png` was between GET VALUE ON DUKE.

#compare results with binary 
#Use PDF report and denoise to review manual correction 


#### iv) Get our dataset

#Link to dataset D1b, D2, D3 already done, make one release per dataset ?

## 3) Results

#How to use and results pdf 

## 4) Discussion

#### TODO

- [x] Explain new metrics
- [x] Choose the suffix
    - `sub-XXX_CONTRAST_label-rootlet.nii.gz`
    - `sub-XXX_ses-XXX_CONTRAST_label-rootlet.nii.gz`
- [ ] Push labeled files
- [ ] Explore softseg value
- [ ] Clean script on repo
- [ ] Create script to highlight spinal levels ?
- [ ] Improve thoracic level segmentation 