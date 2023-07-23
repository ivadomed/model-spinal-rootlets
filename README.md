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

The original dataset (D0a) can be found [here](https://openneuro.org/datasets/ds004507/versions/1.0.1). An initial labeling of
12 subjects (D1a) was performed to train one fold of the nnUNetV2 3d_fullres model (refer
to [issue#5](https://github.com/ivadomed/model-spinal-rootlets/issues/5)). The resulting model (M1a) achieved a Dice
score of approximately 0.52.

After conducting inference on the entire dataset and performing manual corrections, the dataset was expanded to include
20 subjects. However, two subjects were subsequently excluded due to unsatisfactory label quality (
see [issue#7](https://github.com/ivadomed/model-spinal-rootlets/issues/7)), resulting in an 18-subject dataset (D1b). A
full training on 5 folds with 250 epochs resulted in a Dice score of approximately 0.58 (M1b).

Inference was further conducted on T2w data from 267 subjects from
the [spinegeneric](https://github.com/spine-generic/data-multi-subject#spine-generic-public-database-multi-subject)
dataset (D0b) (release r20230223), which had an isotropic voxel size of 0.8mm. Some of the predictions from this inference were
manually corrected, and the resulting dataset was merged with the previous dataset (D1b), resulting in a final dataset
containing 38 subjects (D2). A full training on 5 folds with 1000 epochs resulted in a Dice score of approximately
0.77 (M2) [issue#8](https://github.com/ivadomed/model-spinal-rootlets/issues/8).

The final dataset (D2) has been modified by changing voxel values according to the spinal level resulting in dataset D3. An initial
training of 5 folds with 1000 epochs has been done. After reviewing the results and manually correcting the dataset (
sct_maths -d 1 used to help labeling this time), a new training of 5 folds with 2000 epochs is in progress.

### C) Reproduce
- Find labeled images 
- Create different dataset 
- Run training 
- Run inference 

In the next section, all the instructions to reproduce the label files used in the final dataset will be described. However, label files are also available here (give to dataset with root_label).

#introduce script to convert BIDS->nnUNet and nnUNet->BIDS
#explain how to label on FSLeyes 

#### i) Reproduce D1a, M1a and D1b, M1b
Clone the original dataset D0a
```
git clone https://github.com/OpenNeuroDatasets/ds004507.git
```
#Explain session
#Explain why don't take HeadDown 
Linked to [issue#5](https://github.com/ivadomed/model-spinal-rootlets/issues/5)

With FSLeyes, manually segment the following file: 
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
This is the D1a dataset (100% train image no test image)

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

#Add create a dataset with all D0a in imagesTs

Predict all the segmentation of D0a dataset with the model M1a with `nnUNetv2_predict -i PATH_TO:imagesTs -o PATH_TO:Out_directory -d DATASETID -c 3d_fullres --save_probabilities -chk checkpoint_best.pth`

Manually review the predicted labels. 
Note: subjects `sub-006-Normal` and `sub-009-Normal` have been dropped since they did not satisfy the quality. 

#Convert to BIDS dataset and add json
#Convert to nnUNet

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


Train nnUNet model M1b with `CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASETID -tr nnUNetTrainer_250epochs -f 0`, repeat for fold 1, 2, 3, 4.

#### ii) Reproduce D2, M2 

Linked to [issue#8 part 2)](https://github.com/ivadomed/model-spinal-rootlets/issues/8)

Clone the original dataset D0b
```
git clone git@github.com:spine-generic/data-multi-subject.git
#specify version
```
#Convert Bids to nnUNet 100% imagesTs

#run prediction on full dataset with D1b 

#Manually correct images in this list ()

#convert to Bids dataset 

#Convert to nnUNet dataset with 1 images in Ts (image)

#Merge with D1b with 1 images in Ts (image)

#Create merge script

#Train full fold 1000epochs 

#### iii) Reproduce D3, M3

Linked to [issue#8 part 3)](https://github.com/ivadomed/model-spinal-rootlets/issues/8)
#denoise with sct_maths

#Mannually review and set voxel value C2:2, C3:3, T1: 9... --> D3

#Save to BIDS 

#train 2000 epochs -->M3

#### iv) Get our dataset 

#Link to dataset D1b, D2, D3 already done, make one release per dataset ? 

## 3) Results 
#Explain new metrics used 

## 4) Discussion
#### TODO
- [ ] Explain new metrics 
- [x] Choose the suffix 
  - `sub-XXX_CONTRAST_label-rootlet.nii.gz`
  - `sub-XXX_ses-XXX_CONTRAST_label-rootlet.nii.gz`
- [ ] Push labeled files 
- [ ] Explore softseg value 
- [ ] Clean script on repo
- [ ] Create script to highlight spinal levels ?
- [ ] Improve thoracic level segmentation 