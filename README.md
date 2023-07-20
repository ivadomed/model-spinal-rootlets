# Spinal-Rootlets Segmentation on T2w Images

## 1) Project Overview

The goal of this project is to develop a deep learning (DL) technique-based method to segment and locate the spinal
roots.

## 2) Work Done

### a) Literature and Data Review

No articles were found on this topic.

After reviewing the available
datasets ([issue#1](https://github.com/ivadomed/model-spinal-rootlets/issues/1#issue-1706345176)), it was determined
that none of them were suitable.

### b) Dataset Creation

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

The previous dataset (D2) has been modified by changing voxel values according to the spinal level (D3). An initial
training of 5 folds with 1000 epochs has been done. After reviewing the results and manually correcting the dataset (
sct_maths -d 1 used to help labeling this time), a new training of 5 folds with 2000 epochs is in progress.

### 3) Reproduce
- Find labeled images 
- Create different dataset 
- Run training 
- Run inference 

In the next section all the instruction to reproduce the label file used in the final dataset will be described. Hoever label file are also available here (give to dataset with root_label)

#### a) Reproduce D1a, M1a and D1b, M1b
Clone the original dataset D0a
```
git clone https://github.com/OpenNeuroDatasets/ds004507.git
```
With Fsleyes manually segment the following file: 
<details>
<summary>12 first images to label</summary>
```
#sub-002 --> sub-007
```
</details>

> You can use the `json_write.py` script to add the json file according to the .nii.gz file created

Now convert this Bids dataset to a nnUNet dataset `link to command script`. 
This is the D1a dataset

Add the 
<details>
<summary>dataset.json</summary>
```
# file
```
</details>

Train model D1a with : `CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASETID 3d_fullres 0`

> You can stop when the progress.png reach a plateau (approx 250)

#Add create a dataset full with all D0a in imagesTs

Predict all the segmentation of D0a dataset with the model M1a with `nnUNetv2_predict -i PATH_TO:imagesTs -o PATH_TO:Out_directory -d DATASETID -c 3d_fullres --save_probabilities -chk checkpoint_best.pth`

Manually review the predicted label. In my case I have dropped `sub-006-Normal` `sub-009-Normal` I was not satisfy with the quality. 

#Convert to BIDS dataset and add json
#Convert to nnUNet

Now you have a dataset with 18 subject we call this one D1b 

Train nnUNet model M1b with `CUDA_VISIBLE_DEVICES=XXX nnUNetv2_train DATASETID -tr nnUNetTrainer_250epochs -f 0`, repeat for fold 1, 2, 3, 4.

#### b) Reproduce D2, M2 

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

#### c) Reproduce D3, M3

#denoise with sct_maths

#Mannually review and set voxel value C2:2, C3:3, T1: 9... --> D3

#Save to BIDS 

#train 2000 epochs -->M3

#### d) Get our dataset 

#Link to dataset D1b, D2, D3 already done, make one release per dataset ? 



#### TODO
- [ ] Explain new metrics 
- [ ] Choose the suffix 
- [ ] Push labeled files 
- [ ] Explore softseg value 
- [ ] Improve thoracic level segmentation 