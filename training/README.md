# Spinal Nerve Rootlets Segmentation on T2w Images

## 1) Project Overview

The goal of this project is to develop a deep learning (DL)-based method to segment and locate the spinal nerve
rootlets.

A script to convert nerve segmentation to spinal level is also available.

[Google slides](https://docs.google.com/presentation/d/1ZHliup_Mtk0OcmI1qkwmOIY7Ml4mO6vewIwFQjMMMPo/edit?usp=sharing) to
summarize this project are also available.

## 2) Work Done

How to use the model

#### a) convert your image to nnUNet inference format

Your image have to be T2w and follow this name format.
For example `sub-amuAL05_T2w.nii.gz`
ORIGINAL-NAME = sub-amuAL05
`ORIGINAL-NAME_XXX_0000.nii.gz`
Replace XXX by integer value with 3 digit example 002

#### b) Reorient your images to LPI

`sct_image -i image.nii.gz -o reoriented.nii.gz -setorient LPI`

#### c) Run nnUNet prediction

To install nnUNetV2
follow : [Official documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
or [quick start guide](https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md)

Get the model `duke/temp/theo_root/Dataset010_M3.zip` (`unzip Dataset010_M3.zip`) and place it in nnUNet_results directory
`nnUNetv2_predict -i INPUT_PATH -o OUTPUT_PATH -d 010 -tr nnUNetTrainer_2000epochs -f 0 1 2 3 4 -c 3d_fullres`

> If you already have a model with ID 10 (example mymodel010_perso), change the ID for example `mv Dataset010_M3 Dataset011_M3`

### A) Literature and Data Review

No articles were found on this topic.

After reviewing the available
datasets ([issue#1](https://github.com/ivadomed/model-spinal-rootlets/issues/1#issue-1706345176)), it was determined
that no dataset had available suitable ground truth labels.

### B) Dataset Creation

Initially, the plan was to test with a binary label (0: no rootlet, 1: rootlet). In the following text, the datasets
will be numbered as D1, D2, etc., and the models as M1, M2, etc.

**Active learning pathway:**
![pipeline](pipeline-graph.png)

**Datasets summary**:

| name | number images                          | (mean age (y.o) | (std age | labels         | label origin               | nnUNetV2 <br/>Dice score | link                                                                                            |
|:----:|----------------------------------------|:---------------:|:--------:|----------------|----------------------------|:------------------------:|-------------------------------------------------------------------------------------------------|
| D0a  | 30(10x3 sessions)                      |                 |          | No             | ø                          |            ø             | [open neuro](https://openneuro.org/datasets/ds004507/versions/1.0.1)                            |
| D0b  | 267                                    |                 |          | No             | ø                          |            ø             | [spine-generic](https://github.com/spine-generic/data-multi-subject/tree/r20230223)                            |
| D1a  | 12                                     |      22.5       |   0.52   | Binary         | 100% Manual                |           0.51           | [D1a.tsv](dataset_creation/D1a.tsv) |
| D1b (nnUNet 006)  | 18                                     |      22.9       |   1.21   | Binary         | Prediction + Manual review |         0.52~0.6         | [D1b.tsv](dataset_creation/D1b.tsv) |
|  D2 (nnUNet 007)  | 36 (20 from D0b + D1b) + 2 test images |      26.2       |   5.2    | Binary         | Prediction + Manual review |        0.65~0.75         | [D2.tsv](dataset_creation/D2.tsv)   |
|  D3 (nnUNet 008-9) | 31 + 2 test images                     |      26.5       |   5.5    | Level specific | Value modification         |         0.4~0.6          | [D3.tsv](dataset_creation/D3.tsv)   |

<details>
<summary>Details</summary>

#### D1a)

Dataset D1a was constructed with 12 subjects manually labeled (binary) from D0a. 12 MRIs from 6 subjects (3 female, 3
male), each subject participated in 2 sessions, one with normal neck flexion and another with neck extension. The mean
age is 22.5 y.o with a standard deviation of 0.52. Isotropic resolution of 0.6mm^3 (only one has a
resolution of 0.7mm^3).

One nnUNetV2 fold 3d_fullres model (M1a) was trained on D1a for 50 epochs, achieving a plateau with a dice
score of approximately 0.51. M1a was used to predict 20 subjects from D0a (all head-normal and head-up
images). After a manual review, two images were excluded because of unsatisfactory quality (sub-006_ses-headNormal and
sub-009_ses-headNormal)

> Refer to [issue#5](https://github.com/ivadomed/model-spinal-rootlets/issues/5).

#### D1b)

The resulting dataset, D1b consists of 18 MRIs from 10 subjects, 2 sessions (3 female, 7 male). The mean age is 22.9
years old, with a standard deviation of 1.21. Isotropic resolution of 0.6mm^3 (only one has a resolution of 0.7mm^3).

On this new dataset a five-fold training of nnUNetV2 3d_fullres model (M1b) has been conducted for 250 epochs, dice
scores were between 0.52 and 0.6. An attempt was made to enhance results using the post-processing
command (`nnUNetv2_apply_postprocessing`) of nnUNetV2, but no possible improvement was found so post-processing is
useless in this case. Inference with
M1b has been conducted on the full D0b (spine-generic) dataset.

> Refer to [issue#7](https://github.com/ivadomed/model-spinal-rootlets/issues/7)

#### D2)

A manual review of the D0b prediction has led to a substantial number of images dropped. To facilitate the manual
labeling SCT was used to denoise images (`sct_image --denoise`). Some centers have image specificity that made the
manual
reviewing hazardous and I preferred to only take images where I had a good confident level on my labels.

As a result, only 20 subjects from D0b were retained and combined with D1b to create a new dataset comprising 38
subjects (D2). Within this dataset, two subjects were transferred from the training dataset to the test dataset (
sub-008_ses-headUp, sub-brnoUhb01). The mean age is 26.2 years old, with a standard deviation of 5.2.

A five-fold training of nnUNetV2 3d_fullres model (M2) has been conducted for 1000 epochs, dice scores
were between 0.65 and 0.75. Notably, no post-processing techniques yielded an improvement in scores under these
circumstances.
Inference on the D2 dataset with the M2 model helped me to correct my label and improve the D2 ground truth quality.

> Refer to issue [issue#8 part 2)](https://github.com/ivadomed/model-spinal-rootlets/issues/8).

#### D3)

A new labeling of the D2 dataset with spinal level-depending values has been conducted. As a result of uncertainty, five
images were excluded. The resultant Dataset D3 comprises 33 images, including 31 for training and 2 for testing (same as
D2). This dataset features a subject mean age of 26.5 y.o, standard deviation of 5.5 and incorporates spinal
level-specific spinal nerve segmentation.

A five-fold training of nnUNet 3d_fullres model has been conducted for 1000 epochs, dice scores were
between 0.4 and 0.6. No post-processing techniques led to an increase in scores under these
conditions. Upon reviewing the progress.png graph, a subsequent training was conducted with 2000 epochs. This decision
was based on the observation that the plateau had not been reached within the first 1000 epochs. The second training
yielded a dice score also ranging between 0.4 and 0.6. However, it exhibited more folds with scores
exceeding 0.5 compared to the first training conducted with 1000 epochs.

> Refer to [issue#8 part 3)](https://github.com/ivadomed/model-spinal-rootlets/issues/8).

</details>

### C) Reproduce

In the next section, all the instructions to reproduce the label files used in the final dataset will be described.

nnUNet is used to train model but the dataset format is not BIDS:

- From BIDS to
  nnUNet [convert_bids_to_nnUnetv2.py](https://github.com/ivadomed/utilities/blob/main/dataset_conversion/convert_bids_to_nnUnetv2.py)
- From nnUNet to BIDS [nnUNetV2_to_bids.py](https://github.com/ivadomed/utilities/blob/main/dataset_conversion/nnunetV2_to_bids.py)
- Extract all images from BIDS to nnUNet
  inference [extract_bids_subject.py](https://github.com/ivadomed/model-spinal-rootlets/blob/main/dataset_creation/extract_bids_subject.py)
- Merge several nnUNet
  datasets [concat_nnUnet_dataset.py](https://github.com/ivadomed/model-spinal-rootlets/blob/main/dataset_creation/concat_nnUnet_dataset.py)

#How to explain FLSEYES labeling ?

<details>
<summary>Details</summary>

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

Now convert this BIDS dataset to a nnUNet
dataset `python convert_bids_to_nnUNetv2.py --path-data ~/BIDS --path-out ~/data/dataset-nnunet
--dataset-name Dataset1a --dataset-number 001 --split 1 --seed 99 --copy False`.
This is the D1a dataset (100% train image no test image), composed of 12 images

<details>
<summary>Add dataset.json</summary>

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
Now extract all image from D0a
with `python extract_bids_subject.py --path-bids ~/BIDS --path-out ~/D0a --contrast T2w --suffix 0000`.

Predict all the segmentation of D0a dataset with the model M1a
with `nnUNetv2_predict -i PATH_TO:imagesTs -o PATH_TO:Out_directory -d 001 -c 3d_fullres --save_probabilities -chk checkpoint_best.pth`

Manually review the predicted labels.
Note: subjects `sub-006-headNormal` and `sub-009-headNormal` have been dropped since they did not satisfy the quality.

Linked to [issue#7](https://github.com/ivadomed/model-spinal-rootlets/issues/7)

<details>
<summary>For training dataset.json</summary>

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

Extract all T2w images
with `python extract_bids_subject.py --path-bids ~/spine-generic --path-out ~/D0b --contrast T2w --suffix 0000`

Predict all the segmentation of D0b dataset with the model M1b
with `nnUNetv2_predict -i PATH_TO:imagesTs -o PATH_TO:Out_directory -d DATASETID -tr nnUNetTrainer_250epochs -c 3d_fullres --save_probabilities -f 0 1 2 3 4`

With FSLeyes, manually correct the headUp and headNormal files

> I skipped some center because the quality was not good enough to ensure a good manual correction.


Merge with D1b to create D2, take mri `sub-008_ses-headUp` and `sub-brnoUhb01`and put them into `imagesTs`
and `labelsTs`

Train nnUNet model M2 with `CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASETID -f 0`, repeat
for fold 1, 2, 3, 4.

Out nnUNet Dice score from `progress.png` was between 0.65 and 0.75.

#### iii) Reproduce D3, M3

Linked to [issue#8 part 3)](https://github.com/ivadomed/model-spinal-rootlets/issues/8)

Before we used a binary labeling. But some spinal level are overlapping. One of the solution is to label spinal rootlets
depending on their spinal level (C2->2 .. T1->9).

I have manually corrected and change the value of segmentation of the following files:
<details>
<summary>31 spinal level specific value</summary>

```
sub-amu01_T2w.nii.gz
sub-amu02_T2w.nii.gz
sub-amu05_T2w.nii.gz
sub-balgrist01_T2w.nii.gz
sub-balgrist02_T2w.nii.gz
sub-balgrist03_T2w.nii.gz
sub-balgrist04_T2w.nii.gz
sub-balgrist06_T2w.nii.gz
sub-barcelona01_T2w.nii.gz
sub-barcelona02_T2w.nii.gz
sub-barcelona03_T2w.nii.gz
sub-barcelona06_T2w.nii.gz
sub-brnoUhb03_T2w.nii.gz
sub-cardiff02_T2w.nii.gz
sub-cardiff04_T2w.nii.gz
sub-cmrra02_T2w.nii.gz
sub-cmrra04_T2w.nii.gz
sub-geneva01_T2w.nii.gz
sub-002_ses-headNormal_T2w.nii.gz�
sub-003_ses-headNormal_T2w.nii.gz
sub-003_ses-headUp_T2w.nii.gz
sub-004_ses-headNormal_T2w.nii.gz�
sub-004_ses-headUp_T2w.nii.gz
sub-005_ses-headNormal_T2w.nii.gz
sub-005_ses-headUp_T2w.nii.gz
sub-007_ses-headNormal_T2w.nii.gz
sub-007_ses-headUp_T2w.nii.gz
sub-010_ses-headNormal_T2w.nii.gz
sub-010_ses-headUp_T2w.nii.gz
sub-011_ses-headNormal_T2w.nii.gz
sub-011_ses-headUp_T2w.nii.gz
```

</details>

This dataset D3 composed of 33 images with 31 for train .
I have trained 4 folds of a nnUNet 3d_fullres model for 2000
epochs `CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASETID -tr nnUNetTrainer_2000epochs -f 0`

nnUNet Dice score from `progress.png` was between 0.4 and 0.6.

#### iv) Get our dataset

#Link to dataset D1b, D2, D3 already done, make one release per dataset ?

</details>

## 3) Results

During training the nnUNet dice score was used to evaluate the model. We can observe
that the Dice score improved with each increase in the dataset size. However, there was
a drop in the score between the binary and the level-specific models. This drop can be
partially explained by the low Dice scores observed for thoracic levels

### A) Segmentation results

We introduce new metrics more adapted to the possible use of a spinal rootlet segmentation than the global dice score.
There can be 3 slices types of prediction results :

- True Positive (TP) and True Negative (TN): 100% similarity ground truth vs predicted.
- Slice Positive (SP): 1 or more voxel similarity (but not 100%) ground truth vs predicted.
- False Negative (FN) and False Positive (FP): only ground truth or predicted have voxel labeled.

We can define new evaluation metrics:

- Z-axis F1 score: ![z-axis](z-axis.png)
- Mean common F1 score: ![f1](f1.png)

[dice_y_value_soft.py](https://github.com/ivadomed/model-spinal-rootlets/blob/main/utilities/dice_y_value_soft.py)
produce a report with score for each level and a PDF associated to plot images and compare ground truth versus prediction.

**Comparison between M2 and M3:**

Global dice score on sub-brnoUhb01,
with `sct_dice_coefficient -i sub-brnoUhb01_ground-truth.nii.gz -d sub-brnoUhb01_prediction.nii.gz -bin 1`.

> with `-bin 1` image are binarized so M2 and M3 prediction are compatible

M2 prediction: 3D Dice coefficient = 0.823604

M3 prediction (binarized): 3D Dice coefficient = 0.909745

Beyond the score which is better, other points make the use of the M3 model more relevant. On some slices, we can
observe two distinct spinal levels.

![2level](levl8-9.png)

In these cases having a specific value depending on the spinal level is
mandatory to predict spinal level from spinal nerve segmentation.

Custom metrics for M3 prediction on sub-brnoUhb01:

`python dice_y_value_soft.py -gt sub-brnoUhb01_label-rootlet.nii.gz -pr ~/prediction/sub-brnoUhb01_01.nii.gz -im ~/spine-generic/sub-brnoUhb01_T2w.nii.gz -o PATH_custom_metric-v 2 3 4 5 6 7 8 9 10 11 `

| Spinal level    |  C2   |  C3   |  C4   |  C5   |  C6  |  C7   |  T1  |  T2   | T3 |
|:----------------|:-----:|:-----:|:-----:|:-----:|:----:|:-----:|:----:|:-----:|:--:|
| z-axis F1 score |   1   | 0.947 | 0.923 |   1   | 0.88 | 0.77  |  1   | 0.968 | ø  |
| mean common F1  | 0.813 |  0.9  | 0.913 | 0.908 | 0.92 | 0.925 | 0.89 | 0.961 | ø  |

**Test on other dataset:**

[issue#12](https://github.com/ivadomed/model-spinal-rootlets/issues/12) present results on different dataset.

### B) Spinal level prediction

One of the nerve segmentation usage is to predict spinal labels.
The script `rootlet_to_level.py` can convert rootlet segmentation to spinal level prediction.
In [issue#13](https://github.com/ivadomed/model-spinal-rootlets/issues/13) you will find more explanation on how to
convert from spinal nerve segmentation to spinal level.

#### TODO

- [x] Choose the suffix
    - `sub-XXX_CONTRAST_label-rootlet.nii.gz`
    - `sub-XXX_ses-XXX_CONTRAST_label-rootlet.nii.gz`
 -[ ] Use different suffix
    - `label-nerve` (entire nerve) and `label-rootlet` (only the entry point?)?
- [ ] Push labeled files (labels are located under `~/duke/temp/theo_root/results_img/Dataset010_M3`)
- [x] Individual result with metrics on other images ? 
- [x] Add video tuto
- [ ] Release the model ?
- [ ] Jan: save `Dataset010_M3.zip` and `results_img` to `~/duke/projects/`

#### 4) Future steps

The model M3 can be improved with some ideas:
- Increase the dataset size
- Use MRI from multiple studies, not only spine-generic
- Improve thoracic level segmentation --> Use images with labeled thoracic level (whole-spine)
- Explore softseg value (nnUNet can output soft outputs - one file per level - soft value represent the probability)
