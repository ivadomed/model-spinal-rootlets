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
#### TODO
- [ ] Explain new metrics 
- [ ] Push labeled files 
- [ ] Explore softseg value 
- [ ] Improve thoracic level segmentation 