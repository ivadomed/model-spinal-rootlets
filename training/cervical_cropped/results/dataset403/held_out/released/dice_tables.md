## Test Dice by contrast

Macro level Dice is computed per image by first calculating a 3D Dice for each level present in the union of prediction and reference, then averaging those per-level Dice values. Values are mean ± sample SD and median [IQR]. Binary Dice collapses all rootlet levels to foreground.

| Contrast | n | Macro level Dice | Macro median [IQR] | Binary Dice | Binary median [IQR] |
| --- | --- | --- | --- | --- | --- |
| All | 17 | 0.635 ± 0.061 | 0.645 [0.606–0.683] | 0.644 ± 0.059 | 0.655 [0.628–0.685] |
| T2w | 5 | 0.639 ± 0.046 | 0.627 [0.606–0.683] | 0.652 ± 0.035 | 0.643 [0.628–0.685] |
| INV1 | 4 | 0.603 ± 0.073 | 0.622 [0.584–0.642] | 0.611 ± 0.072 | 0.630 [0.598–0.643] |
| INV2 | 4 | 0.661 ± 0.056 | 0.670 [0.631–0.700] | 0.666 ± 0.057 | 0.677 [0.639–0.704] |
| UNIT1 | 4 | 0.638 ± 0.079 | 0.657 [0.621–0.674] | 0.647 ± 0.079 | 0.667 [0.632–0.682] |

## Test Dice by spinal level

| Level | GT present | Pred present | n valid | Dice | Median [IQR] |
| --- | --- | --- | --- | --- | --- |
| C2 | 17 | 17 | 17 | 0.614 ± 0.103 | 0.626 [0.546–0.703] |
| C3 | 17 | 17 | 17 | 0.608 ± 0.147 | 0.647 [0.600–0.710] |
| C4 | 17 | 17 | 17 | 0.609 ± 0.075 | 0.619 [0.552–0.662] |
| C5 | 17 | 17 | 17 | 0.637 ± 0.086 | 0.653 [0.570–0.704] |
| C6 | 17 | 17 | 17 | 0.670 ± 0.044 | 0.660 [0.644–0.693] |
| C7 | 17 | 17 | 17 | 0.667 ± 0.057 | 0.654 [0.636–0.724] |
| C8 | 17 | 17 | 17 | 0.665 ± 0.121 | 0.712 [0.587–0.750] |
| T1 | 17 | 17 | 17 | 0.612 ± 0.078 | 0.594 [0.581–0.681] |

## Per-image test Dice

| Case | Contrast | Macro level Dice | Binary Dice | GT levels | Predicted levels |
| --- | --- | --- | --- | --- | --- |
| sub-007_ses-headNormal_009 | T2w | 0.606 | 0.611 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-010_ses-headUp_015 | T2w | 0.588 | 0.628 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-amu02_215 | T2w | 0.627 | 0.643 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-barcelona01_212 | T2w | 0.683 | 0.685 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-brnoUhb03_209 | T2w | 0.690 | 0.691 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-17_inv-1_part-mag_MP2RAGE | INV1 | 0.633 | 0.632 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-17_inv-2_part-mag_MP2RAGE | INV2 | 0.645 | 0.655 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-17_UNIT1 | UNIT1 | 0.661 | 0.665 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-24_inv-1_part-mag_MP2RAGE | INV1 | 0.500 | 0.507 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-24_inv-2_part-mag_MP2RAGE | INV2 | 0.588 | 0.591 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-24_UNIT1 | UNIT1 | 0.527 | 0.534 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-31_inv-1_part-mag_MP2RAGE | INV1 | 0.611 | 0.628 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-31_inv-2_part-mag_MP2RAGE | INV2 | 0.695 | 0.699 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-31_UNIT1 | UNIT1 | 0.653 | 0.670 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-37_inv-1_part-mag_MP2RAGE | INV1 | 0.669 | 0.676 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-37_inv-2_part-mag_MP2RAGE | INV2 | 0.714 | 0.719 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-37_UNIT1 | UNIT1 | 0.712 | 0.719 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |

Notes:

- Dice is computed on the supplied prediction/reference grids with no post-processing.
- A level absent from both prediction and reference is excluded (NA); a missed reference level scores 0.
- The official `nnUNetv2_evaluate_folder` output may report a NaN aggregate when classes are absent from some images; these tables aggregate finite per-image values explicitly.
