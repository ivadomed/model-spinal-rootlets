## Test Dice by contrast

Macro level Dice is computed per image by first calculating a 3D Dice for each level present in the union of prediction and reference, then averaging those per-level Dice values. Values are mean ± sample SD and median [IQR]. Binary Dice collapses all rootlet levels to foreground.

| Contrast | n | Macro level Dice | Macro median [IQR] | Binary Dice | Binary median [IQR] |
| --- | --- | --- | --- | --- | --- |
| All | 17 | 0.614 ± 0.064 | 0.636 [0.578–0.662] | 0.625 ± 0.062 | 0.647 [0.600–0.670] |
| T2w | 5 | 0.618 ± 0.047 | 0.606 [0.596–0.664] | 0.638 ± 0.042 | 0.647 [0.628–0.670] |
| INV1 | 4 | 0.587 ± 0.085 | 0.607 [0.551–0.642] | 0.595 ± 0.083 | 0.616 [0.569–0.641] |
| INV2 | 4 | 0.636 ± 0.057 | 0.651 [0.625–0.661] | 0.642 ± 0.055 | 0.657 [0.631–0.667] |
| UNIT1 | 4 | 0.616 ± 0.082 | 0.640 [0.598–0.658] | 0.624 ± 0.081 | 0.647 [0.609–0.661] |

## Test Dice by spinal level

| Level | GT present | Pred present | n valid | Dice | Median [IQR] |
| --- | --- | --- | --- | --- | --- |
| C2 | 17 | 17 | 17 | 0.597 ± 0.128 | 0.653 [0.500–0.715] |
| C3 | 17 | 17 | 17 | 0.598 ± 0.171 | 0.654 [0.619–0.686] |
| C4 | 17 | 17 | 17 | 0.556 ± 0.081 | 0.569 [0.493–0.592] |
| C5 | 17 | 17 | 17 | 0.608 ± 0.090 | 0.649 [0.571–0.658] |
| C6 | 17 | 17 | 17 | 0.658 ± 0.055 | 0.674 [0.607–0.682] |
| C7 | 17 | 17 | 17 | 0.664 ± 0.053 | 0.659 [0.643–0.705] |
| C8 | 17 | 17 | 17 | 0.642 ± 0.113 | 0.688 [0.596–0.693] |
| T1 | 17 | 17 | 17 | 0.589 ± 0.077 | 0.589 [0.567–0.632] |

## Per-image test Dice

| Case | Contrast | Macro level Dice | Binary Dice | GT levels | Predicted levels |
| --- | --- | --- | --- | --- | --- |
| sub-007_ses-headNormal_009 | T2w | 0.557 | 0.571 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-010_ses-headUp_015 | T2w | 0.596 | 0.647 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-amu02_215 | T2w | 0.606 | 0.628 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-barcelona01_212 | T2w | 0.664 | 0.674 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-brnoUhb03_209 | T2w | 0.666 | 0.670 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-17_inv-1_part-mag_MP2RAGE | INV1 | 0.636 | 0.632 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-17_inv-2_part-mag_MP2RAGE | INV2 | 0.649 | 0.654 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-17_UNIT1 | UNIT1 | 0.648 | 0.643 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-24_inv-1_part-mag_MP2RAGE | INV1 | 0.471 | 0.477 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-24_inv-2_part-mag_MP2RAGE | INV2 | 0.555 | 0.563 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-24_UNIT1 | UNIT1 | 0.498 | 0.508 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-31_inv-1_part-mag_MP2RAGE | INV1 | 0.578 | 0.600 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-31_inv-2_part-mag_MP2RAGE | INV2 | 0.652 | 0.659 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-31_UNIT1 | UNIT1 | 0.631 | 0.650 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-37_inv-1_part-mag_MP2RAGE | INV1 | 0.662 | 0.670 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-37_inv-2_part-mag_MP2RAGE | INV2 | 0.687 | 0.691 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-37_UNIT1 | UNIT1 | 0.686 | 0.696 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |

Notes:

- Dice is computed on the supplied prediction/reference grids with no post-processing.
- A level absent from both prediction and reference is excluded (NA); a missed reference level scores 0.
- The official `nnUNetv2_evaluate_folder` output may report a NaN aggregate when classes are absent from some images; these tables aggregate finite per-image values explicitly.
