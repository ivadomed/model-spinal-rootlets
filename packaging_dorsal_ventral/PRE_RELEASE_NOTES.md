## Dorsal/ventral rootlet classifier

- Experimental Dataset906 `fold_0/checkpoint_final` model.
- Input: an MRI and its level-labelled RootletSeg mask on one grid.
- Output: dorsal and ventral masks that exactly partition the supplied support.
- Best observed held-out result: 99.54% dorsal Dice and 98.74% ventral Dice.
- Usage: see `README.md` in the ZIP.
- Results and limitations: PR #112.
