# Try the dorsal/ventral rootlet classifier

This research pre-release divides an existing level-labelled RootletSeg mask
into dorsal and ventral voxels.

## Install

```console
conda create -n rootlets-dv python=3.10 -y
conda activate rootlets-dv
```

Install one PyTorch build:

```console
# CPU or macOS
python -m pip install torch==2.5.1

# Romane
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies and verify the model:

```console
python -m pip install nnunetv2==2.4.2 nibabel==5.4.2
shasum -a 256 -c SHA256SUMS
```

## Input

- One anatomical MRI NIfTI image.
- Its level-labelled RootletSeg NIfTI output on the same voxel grid.
- Rootlet values `2`–`8` match training. Other positive levels are accepted but
  were not represented in training.

The model was trained on HC-Leipzig UNIT1 images. Other contrasts have
qualitative QC only and no expert dorsal/ventral accuracy measurement.

## Run

Run from the extracted pre-release folder:

```console
python infer_dorsal_ventral.py \
  --image sub-001_UNIT1.nii.gz \
  --rootlets sub-001_UNIT1_label-rootlets_dseg.nii.gz \
  --output-class-map sub-001_UNIT1_desc-dvClassifier_dseg.nii.gz \
  --output-dorsal sub-001_UNIT1_desc-dorsal_label-rootlets_dseg.nii.gz \
  --output-ventral sub-001_UNIT1_desc-ventral_label-rootlets_dseg.nii.gz \
  --device cuda
```

Use `--device cpu` when CUDA is unavailable. The command reorients both inputs
to RPI, runs `fold_0/checkpoint_final.pth`, restores the original grid, and
guarantees that dorsal plus ventral equals the supplied RootletSeg mask.

The class map uses `0=background`, `1=dorsal`, and `2=ventral`. The separate
dorsal and ventral files retain the original spinal-level values.

## Scope

- Pilot: 10 train, 2 validation, and 3 held-out HC-Leipzig UNIT1 cases.
- Held-out accuracy is conditional on manual rootlet support.
- External RootletSeg predictions have no expert dorsal/ventral labels.
- Research use only; not intended for clinical decisions.
