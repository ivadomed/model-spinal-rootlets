# Spinal-Rootlets Segmentation on T2w Images

## 1) Project Overview

The goal of this project is to develop a deep learning (DL)-based method to segment and locate the spinal rootlets.

[Google slides](https://docs.google.com/presentation/d/1ZHliup_Mtk0OcmI1qkwmOIY7Ml4mO6vewIwFQjMMMPo/edit?usp=sharing) to
summarize this project is also available.

## 2) Work Done

### A) Literature and Data Review

No articles were found on this topic.

After reviewing the available
datasets ([issue#1](https://github.com/ivadomed/model-spinal-rootlets/issues/1#issue-1706345176)), it was determined
that none of them were suitable.

### B) Dataset Creation

Initially, the plan was to test with a binary label (0: no rootlet, 1: rootlet). In the following text, the datasets
will be numbered as D1, D2, etc., and the models as M1, M2, etc.

**Active learning pathway:**
![pipeline](pipeline-graph.png)

Datasets summary:

| name | number images          | link                                                                 | labels         |
|------|------------------------|----------------------------------------------------------------------|----------------|
| D0a  | 30(10x3 sessions)      | [open neuro](https://openneuro.org/datasets/ds004507/versions/1.0.1) | No             |
| D0b  | 267                    | [spine-generic](https://github.com/spine-generic/data-multi-subject) | No             |
| D1a  | 12                     |                                                                      | Binary         |
| D1b  | 18                     |                                                                      | Binary         |
| D2   | 38 (20 from D0b + D1b) |                                                                      | Binary         |
| D3   | 31                     |                                                                      | Level specific |

#### D1a)

Dataset D1a was constructed with 12 subjects manually labeled (binary) from D0a. 12 MRIs from 6 subjects (3 female, 3
male), each subject participated in 2 sessions, one with normal neck flexion and another with neck extension. The mean
age is 22.5 y.o with a standard deviation of 0.52. Isotropic resolution of 0.6mm^3 (only one has a
resolution of 0.7mm^3).

The nnUNetV2 fold 3d_fullres model (M1a) was trained on D1a for 50 epochs, achieving a plateau with a dice
score of approximately 0.51. M1a was used to predict 20 subjects from D0a (all head-normal and head-up
images). After a manual review, two images were excluded because of unsatisfactory quality (sub-006_ses-headNormal and
sub-009_ses-headNormal)

> Refer to [issue#5](https://github.com/ivadomed/model-spinal-rootlets/issues/5).

#### D1b)

The resulting dataset, D1b consists of 18 MRIs from 10 subjects, 2 sessions (3 female, 7 male). The mean age is 22.9
years old, with a standard deviation of 1.21. Isotropic resolution of 0.6mm^3 (only one has a resolution of 0.7mm^3).

On this new dataset a five-fold training of nnUNetV2 3d_fullres model (M1b) has been conducted for 250 epochs, dice
scores were between 0.52 and 0.6. An attempt was made to enhance results using the post-processing
command of nnUNetV2, but no possible improvement was found so post-processing is useless in this case. Inference with
M1b has been conducted on the full D0b (spine-generic) dataset.

> Refer to [issue#7](https://github.com/ivadomed/model-spinal-rootlets/issues/7)

#### D2)

A manual review of the D0b prediction has led to a substantial number of images dropped. To facilitate the manual
labelin SCT was used to denoise images (`sct_image --denoise` ). Some centers have image specificity that made the
manual
reviewing hazardous and I preferred to only take images where I had a good confident level on my labels.

As a result, only 20 subjects from D0b were retained and combined with D1b to create a new dataset comprising 38
subjects (D2). Within this dataset, two subjects were transferred from the training dataset to the test dataset (
sub-008_ses-headUp, sub-brnoUhb01).

A five-fold training of nnUNetV2 3d_fullres model (M2) has been conducted for 1000 epochs, dice scores
were between 0.65 and 0.75. Notably, no post-processing techniques yielded an improvement in scores under these
circumstances.
Inference on the D2 dataset with the M2 model helped me to correct my label and improve the D2 ground truth quality.

> Refer to issue [issue#8 part 2)](https://github.com/ivadomed/model-spinal-rootlets/issues/8).

#### D3)

A new labeling of the D2 dataset with spinal level-depending values has been conducted. As a result of uncertainty, five
images were excluded. The resultant Dataset D3 comprises 33 images, including 31 for training and 2 for testing (same as
D2). This dataset features a subject mean age of XX and incorporates spinal level-specific spinal nerve segmentation.

A five-fold training of nnUNet 3d_fullres model has been conducted for 1000 epochs, dice scores were
between 0.4 and 0.6. No post-processing techniques led to an increase in scores under these
conditions. Upon reviewing the progress.png graph, a subsequent training was conducted with 2000 epochs. This decision
was based on the observation that the plateau had not been reached within the first 1000 epochs. The second training
yielded a dice score also ranging between 0.4 and 0.6. However, it exhibited more folds with scores
exceeding 0.5 compared to the first training conducted with 1000 epochs.

> Refer to [issue#8 part 3)](https://github.com/ivadomed/model-spinal-rootlets/issues/8).

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

#How to explain FLSEYES labeling ?

#### i) Reproduce D1a, M1a and D1b, M1b
<details>
<summary>Details</summary>
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
</details>

#### ii) Reproduce D2, M2

Linked to [issue#8 part 2)](https://github.com/ivadomed/model-spinal-rootlets/issues/8)

<details>
<summary>Details</summary>
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

Since this point we introduce new metric to evaluate the model more detail
in [issue#8](https://github.com/ivadomed/model-spinal-rootlets/issues/8):

- Z-axis F1 score
- Mean common F1 score
</details>

#### iii) Reproduce D3, M3

Linked to [issue#8 part 3)](https://github.com/ivadomed/model-spinal-rootlets/issues/8)

Before we used a binary labeling. But some spinal level are overlapping. One of the solution is to label spinal rootlets
depending on their spinal level (C2->2 .. T1->9).
<details>
<summary>Details</summary>

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
</details>

#### iv) Get our dataset

#Link to dataset D1b, D2, D3 already done, make one release per dataset ?

## 3) Results

### A) Segmentation results

### B) Spinal level prediction

## 4) Discussion

#### TODO

- [x] Explain new metrics
- [x] Choose the suffix
    - `sub-XXX_CONTRAST_label-rootlet.nii.gz`
    - `sub-XXX_ses-XXX_CONTRAST_label-rootlet.nii.gz`
- [ ] Push labeled files
- [ ] Explore softseg value
- [ ] Clean script on repo
- [ ] Repair broken link 
- [x] Create script to highlight spinal levels
    - `rootlet_to_level.py`
- [x] Compare results with Cadotte and frostel
    - `segment_to_csv.py`
    - `calc_all.py`
    - [issue#10](https://github.com/ivadomed/model-spinal-rootlets/issues/10)
- [ ] Improve thoracic level segmentation 