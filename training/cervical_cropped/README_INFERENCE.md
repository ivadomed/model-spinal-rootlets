# Try the cropped C2–T1 rootlets model

This research pre-release segments dorsal and ventral C2–T1 rootlets from a
full 3-D T2w, MP2RAGE INV1, INV2, or UNIT1 NIfTI image.

## Install

```console
conda create -n rootlets-cropped python=3.10 -y
conda activate rootlets-cropped
```

Install one PyTorch build:

```console
# CPU or macOS
python -m pip install torch==2.5.1 torchvision==0.20.1

# Romane or Rosenberg
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# TASSAN (Blackwell)
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```console
python -m pip install \
  nnunetv2==2.5.2 \
  nibabel==5.4.2 \
  onnx==1.22.0 \
  onnxruntime==1.23.2 \
  ultralytics==8.4.92
python -m pip install \
  "sc-crop @ git+https://github.com/ivadomed/sc-crop.git@f5f952e5a0e6f4c63cf8cb391aad88fe66a72e68"

python install_inference_trainer.py
shasum -a 256 -c SHA256SUMS
```

## Run

Run from the extracted pre-release folder:

```console
python infer_new_image.py \
  -i sub-001_T2w.nii.gz \
  -o sub-001_T2w_label-rootlets_dseg.nii.gz \
  --device cuda \
  --crop-device cpu
```

Use `--device cpu` when CUDA is unavailable. Use `--crop-device cuda` only
when `onnxruntime-gpu` is installed.

### NeuroPoly GPU clusters

Reserve one GPU before running.

On Romane or TASSAN:

```console
ssh romane  # or: ssh tassan
tmux new -s rootlets-inference
set_slot <GPU_ID>
conda activate rootlets-cropped

CUDA_VISIBLE_DEVICES=<GPU_ID> python infer_new_image.py \
  -i /path/to/input.nii.gz \
  -o /path/to/input_label-rootlets_dseg.nii.gz \
  --device cuda \
  --crop-device cpu
```

On Rosenberg:

```console
ssh rosenberg
tmux new -s rootlets-inference
conda activate rootlets-cropped

CUDA_VISIBLE_DEVICES=<GPU_ID> python infer_new_image.py \
  -i /path/to/input.nii.gz \
  -o /path/to/input_label-rootlets_dseg.nii.gz \
  --device cuda \
  --crop-device cpu
```

The command:

- detects and crops the spinal cord with the training margins;
- reorients the crop to RPI;
- runs `fold_all/checkpoint_best.pth` without test-time augmentation;
- restores the labels to the input image grid.

Output values are `0` for background, `2`–`8` for C2–C8, and `9` for T1.

## Check

```console
fsleyes \
  sub-001_T2w.nii.gz \
  sub-001_T2w_label-rootlets_dseg.nii.gz \
  -cm subcortical -a 70
```

Check that each predicted level follows the rootlets on both sides of the cord.
This is a research model and is not intended for clinical decisions.
