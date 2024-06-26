# Automatic Segmentation of Spinal Nerve Rootlets 

[![DOI](https://img.shields.io/badge/ImagingNeuroscience-10.1162/imag_a_00218-status.svg)](https://doi.org/10.1162/imag_a_00218)

![sub-barcelona01](https://github.com/ivadomed/model-spinal-rootlets/assets/39456460/0315228f-a3c5-4aca-80ce-c00fd13a5fc9)

This repository contains the code for deep learning-based segmentation of the spinal nerve rootlets. 
The code is based on the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).

## Citation Info

If you find this work and/or code useful for your research, please cite the [following paper](https://doi.org/10.1162/imag_a_00218):

```bibtex
@article{10.1162/imag_a_00218,
    author = {Valošek, Jan and Mathieu, Theo and Schlienger, Raphaëlle and Kowalczyk, Olivia S. and Cohen-Adad, Julien},
    title = "{Automatic Segmentation of the Spinal Cord Nerve Rootlets}",
    journal = {Imaging Neuroscience},
    year = {2024},
    month = {06},
    issn = {2837-6056},
    doi = {10.1162/imag_a_00218},
    url = {https://doi.org/10.1162/imag\_a\_00218},
}
```

## Model Overview

The model was trained on T2-weighted images and provides semantic (i.e., level-specific) segmentation of the dorsal 
spinal nerve rootlets.

## How to use the model

### Install dependencies

- [Spinal Cord Toolbox (SCT) v6.2](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/6.2) or higher -- follow the installation instructions [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox?tab=readme-ov-file#installation)
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- Python

Once the dependencies are installed, download the latest rootlets model:

```bash
sct_deepseg -install-task seg_spinal_rootlets_t2w
```

### Getting the rootlet segmentation

To segment a single image, run the following command: 

```bash
sct_deepseg -i <INPUT> -o <OUTPUT> -task seg_spinal_rootlets_t2w
```

For example:

```bash
sct_deepseg -i sub-001_T2w.nii.gz -o sub-001_T2w_label-rootlets_dseg.nii.gz -task seg_spinal_rootlets_t2w
```
