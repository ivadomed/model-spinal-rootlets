# Automatic Segmentation of Spinal Nerve Rootlets 

[![arXiv](https://img.shields.io/badge/arXiv-2310.15402-b31b1b.svg)](https://doi.org/10.48550/arXiv.2402.00724)

![sub-barcelona01](https://github.com/ivadomed/model-spinal-rootlets/assets/39456460/0315228f-a3c5-4aca-80ce-c00fd13a5fc9)

This repository contains the code for deep learning-based segmentation of the spinal nerve rootlets. 
The code is based on the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet).

## Citation Info

If you find this work and/or code useful for your research, please refer to the [following preprint](https://doi.org/10.48550/arXiv.2402.00724):

```bibtex
@misc{valosek2024automatic,
      title={Automatic Segmentation of the Spinal Cord Nerve Rootlets}, 
      author={Jan Valosek and Theo Mathieu and Raphaelle Schlienger and Olivia S. Kowalczyk and Julien Cohen-Adad},
      year={2024},
      eprint={2402.00724},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Model Overview

The model was trained on T2-weighted images and provides semantic (i.e., level-specific) segmentation of the dorsal 
spinal nerve rootlets.

## Getting started

### Dependencies

- [Spinal Cord Toolbox (SCT) v6.2](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/6.2) or higher -- follow the installation instructions [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox?tab=readme-ov-file#installation)
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- Python

Once the SCT v6.2 or higher is installed, download the latest rootlets model:

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
