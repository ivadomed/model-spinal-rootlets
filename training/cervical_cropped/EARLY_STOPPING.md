# Early-stopping policy for Dataset403

## Decision

Training retains the original 2,000-epoch maximum and adds a conservative
plateau stop. A run ends early only when:

1. at least 1,000 epochs have completed;
2. the exponential moving average (EMA) of mean foreground validation
   pseudo-Dice has not gained more than 0.002 over its material-best reference;
3. this condition has persisted for 400 consecutive epochs.

In compact form:

```text
stop =
  completed_epochs >= 1000
  and epochs_since_material_improvement >= 400

material_improvement =
  current_ema_pseudo_dice > material_best + 0.002
```

Gains smaller than 0.002 are not discarded. They accumulate against the fixed
material-best reference and reset patience once their total exceeds 0.002.
nnU-Net's normal, unthresholded best-checkpoint rule is unchanged, so any
strict EMA improvement can still update `checkpoint_best.pth`.

## Why this is not a 1,000-epoch cap

The RootletSeg paper's supplementary training protocol reports 2,000 epochs
for the uncropped five-fold experiment and the production model trained on all
76 non-test scans. It does not report an early-stopping experiment:

- [Krejci et al., 2026](https://www.nature.com/articles/s41598-026-49164-0)
- [Supplementary training protocol](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-026-49164-0/MediaObjects/41598_2026_49164_MOESM1_ESM.pdf)

The released uncropped `fold_all` checkpoint is stronger evidence about
convergence than the fixed epoch count in the paper. Its stored EMA
pseudo-Dice history continued to improve after epoch 1,000:

| Epoch | EMA foreground pseudo-Dice |
| ---: | ---: |
| 250 | 0.264 |
| 500 | 0.411 |
| 750 | 0.501 |
| 1,000 | 0.591 |
| 1,250 | 0.606 |
| 1,500 | 0.697 |
| 1,750 | 0.722 |
| 2,000 | 0.751 |

The absolute best was at epoch 1,999 in one-based reporting (zero-based index
1,998). A hard cap near 1,000 would therefore have discarded meaningful late
learning in the reference model. This is also consistent with the repository
history describing how training behavior depended on spatial context and patch
size in [issue #84](https://github.com/ivadomed/model-spinal-rootlets/issues/84).

## Retrospective policy check

The rule was replayed against two completed 2,000-epoch checkpoints:

| Historical run | Absolute best | Policy outcome |
| --- | ---: | --- |
| Released uncropped `fold_all` | epoch 1,999; EMA 0.7516 | no early stop; preserves all 2,000 epochs |
| Earlier cropped fold 0 | epoch 1,089; EMA 0.6839 | stop after epoch 1,445 |

For the cropped fold, the material-best reference was last reset near epoch
1,045. Patience of 400 would save about 555 epochs while still retaining the
actual best checkpoint. For the uncropped model, later material gains
repeatedly reset patience.

Replay any compatible nnU-Net checkpoint with:

```console
python training/cervical_cropped/analyze_early_stopping_checkpoint.py \
  /path/to/checkpoint_final.pth
```

## Metric interpretation

Validation pseudo-Dice is computed from validation patches sampled during
training. It is useful for detecting optimization plateaus, but it is not the
final full-image Dice reported on the held-out test set.

- Folds 0-4 use cases excluded from that fold's training subset, so their
  pseudo-Dice provides a validation signal.
- For `fold_all`, nnU-Net uses all cases for both training and
  pseudo-validation. Its curve is therefore a convergence signal on seen
  cases, not an estimate of generalization.
- Final scientific comparison with the released uncropped model must use the
  unchanged 17-case test cohort and full-image metrics.

## Safeguards and outputs

- Non-finite histories cannot trigger early stopping.
- Training cannot stop before epoch 1,000.
- The 2,000-epoch maximum and polynomial learning-rate schedule are unchanged.
- `checkpoint_best.pth` remains the true unthresholded nnU-Net best.
- `checkpoint_final.pth` is written normally when a plateau stop occurs.
- The stopping decision and its metric state are written to the training log.

The custom trainer is kept in this repository and installed into the pinned
Romane environment with
`install_early_stopping_trainer_romane.sh`. Keeping installation explicit
avoids silently modifying a shared Python environment.
