## Test Dice by contrast

Macro level Dice is computed per image by first calculating a 3D Dice for each level present in the union of prediction and reference, then averaging those per-level Dice values. Values are mean ± sample SD and median [IQR]. Binary Dice collapses all rootlet levels to foreground.

| Contrast | n | Macro level Dice | Macro median [IQR] | Binary Dice | Binary median [IQR] |
| --- | --- | --- | --- | --- | --- |
| All | 76 | 0.797 ± 0.051 | 0.806 [0.781–0.834] | 0.802 ± 0.046 | 0.813 [0.787–0.834] |
| T2w | 31 | 0.755 ± 0.050 | 0.769 [0.727–0.792] | 0.769 ± 0.051 | 0.782 [0.746–0.803] |
| INV1 | 15 | 0.820 ± 0.023 | 0.827 [0.804–0.833] | 0.820 ± 0.024 | 0.821 [0.811–0.835] |
| INV2 | 15 | 0.832 ± 0.021 | 0.834 [0.812–0.850] | 0.829 ± 0.022 | 0.838 [0.812–0.845] |
| UNIT1 | 15 | 0.828 ± 0.022 | 0.835 [0.807–0.842] | 0.827 ± 0.023 | 0.832 [0.813–0.843] |

## Test Dice by spinal level

| Level | GT present | Pred present | n valid | Dice | Median [IQR] |
| --- | --- | --- | --- | --- | --- |
| C2 | 76 | 76 | 76 | 0.792 ± 0.068 | 0.787 [0.763–0.832] |
| C3 | 76 | 76 | 76 | 0.821 ± 0.045 | 0.829 [0.804–0.848] |
| C4 | 76 | 76 | 76 | 0.799 ± 0.073 | 0.813 [0.778–0.842] |
| C5 | 76 | 76 | 76 | 0.793 ± 0.070 | 0.803 [0.774–0.838] |
| C6 | 76 | 76 | 76 | 0.811 ± 0.063 | 0.819 [0.789–0.855] |
| C7 | 76 | 76 | 76 | 0.823 ± 0.061 | 0.842 [0.784–0.867] |
| C8 | 76 | 76 | 76 | 0.798 ± 0.069 | 0.814 [0.766–0.847] |
| T1 | 75 | 76 | 76 | 0.742 ± 0.109 | 0.754 [0.704–0.799] |

## Per-image test Dice

| Case | Contrast | Macro level Dice | Binary Dice | GT levels | Predicted levels |
| --- | --- | --- | --- | --- | --- |
| sub-002_ses-headNormal_000 | T2w | 0.725 | 0.738 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-003_ses-headNormal_002 | T2w | 0.679 | 0.689 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-003_ses-headUp_003 | T2w | 0.648 | 0.654 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-004_ses-headNormal_004 | T2w | 0.682 | 0.686 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-005_ses-headNormal_006 | T2w | 0.671 | 0.676 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-005_ses-headUp_007 | T2w | 0.638 | 0.652 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-007_ses-headUp_010 | T2w | 0.711 | 0.720 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-010_ses-headNormal_014 | T2w | 0.717 | 0.746 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-011_ses-headNormal_016 | T2w | 0.739 | 0.746 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-011_ses-headUp_017 | T2w | 0.758 | 0.772 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-amu01_064 | T2w | 0.742 | 0.762 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-amu05_216 | T2w | 0.760 | 0.771 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-balgrist01_218 | T2w | 0.812 | 0.827 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-balgrist02_084 | T2w | 0.790 | 0.800 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-balgrist03_066 | T2w | 0.791 | 0.795 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-balgrist04_071 | T2w | 0.798 | 0.802 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-balgrist06_222 | T2w | 0.769 | 0.775 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-barcelona02_063 | T2w | 0.783 | 0.788 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-barcelona03_074 | T2w | 0.765 | 0.782 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-barcelona06_214 | T2w | 0.794 | 0.808 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-brnoUhb01_085 | T2w | 0.788 | 0.798 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-cardiff02_182 | T2w | 0.775 | 0.793 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-cardiff04_172 | T2w | 0.787 | 0.805 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-cmrra02_140 | T2w | 0.803 | 0.818 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-cmrra04_158 | T2w | 0.743 | 0.774 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-geneva01_253 | T2w | 0.797 | 0.813 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-mgh01_301 | T2w | 0.729 | 0.815 | C2;C3;C4;C5;C6;C7;C8 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-mgh02_302 | T2w | 0.776 | 0.780 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-stanford02_303 | T2w | 0.794 | 0.797 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-stanford05_304 | T2w | 0.814 | 0.819 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-ucdavis03_305 | T2w | 0.828 | 0.829 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-18_inv-1_part-mag_MP2RAGE | INV1 | 0.846 | 0.847 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-18_inv-2_part-mag_MP2RAGE | INV2 | 0.839 | 0.839 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-18_UNIT1 | UNIT1 | 0.848 | 0.848 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-19_inv-1_part-mag_MP2RAGE | INV1 | 0.828 | 0.833 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-19_inv-2_part-mag_MP2RAGE | INV2 | 0.834 | 0.839 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-19_UNIT1 | UNIT1 | 0.835 | 0.843 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-20_inv-1_part-mag_MP2RAGE | INV1 | 0.773 | 0.773 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-20_inv-2_part-mag_MP2RAGE | INV2 | 0.794 | 0.794 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-20_UNIT1 | UNIT1 | 0.783 | 0.784 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-22_inv-1_part-mag_MP2RAGE | INV1 | 0.848 | 0.844 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-22_inv-2_part-mag_MP2RAGE | INV2 | 0.850 | 0.838 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-22_UNIT1 | UNIT1 | 0.850 | 0.846 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-23_inv-1_part-mag_MP2RAGE | INV1 | 0.828 | 0.824 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-23_inv-2_part-mag_MP2RAGE | INV2 | 0.853 | 0.850 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-23_UNIT1 | UNIT1 | 0.839 | 0.839 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-25_inv-1_part-mag_MP2RAGE | INV1 | 0.832 | 0.837 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-25_inv-2_part-mag_MP2RAGE | INV2 | 0.850 | 0.853 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-25_UNIT1 | UNIT1 | 0.840 | 0.842 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-27_inv-1_part-mag_MP2RAGE | INV1 | 0.827 | 0.830 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-27_inv-2_part-mag_MP2RAGE | INV2 | 0.841 | 0.844 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-27_UNIT1 | UNIT1 | 0.834 | 0.838 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-28_inv-1_part-mag_MP2RAGE | INV1 | 0.806 | 0.819 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-28_inv-2_part-mag_MP2RAGE | INV2 | 0.811 | 0.822 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-28_UNIT1 | UNIT1 | 0.807 | 0.821 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-29_inv-1_part-mag_MP2RAGE | INV1 | 0.851 | 0.857 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-29_inv-2_part-mag_MP2RAGE | INV2 | 0.859 | 0.866 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-29_UNIT1 | UNIT1 | 0.860 | 0.867 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-30_inv-1_part-mag_MP2RAGE | INV1 | 0.802 | 0.804 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-30_inv-2_part-mag_MP2RAGE | INV2 | 0.811 | 0.812 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-30_UNIT1 | UNIT1 | 0.804 | 0.807 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-36_inv-1_part-mag_MP2RAGE | INV1 | 0.786 | 0.780 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-36_inv-2_part-mag_MP2RAGE | INV2 | 0.812 | 0.805 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-36_UNIT1 | UNIT1 | 0.799 | 0.791 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-38_inv-1_part-mag_MP2RAGE | INV1 | 0.825 | 0.817 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-38_inv-2_part-mag_MP2RAGE | INV2 | 0.857 | 0.846 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-38_UNIT1 | UNIT1 | 0.838 | 0.830 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-39_inv-1_part-mag_MP2RAGE | INV1 | 0.801 | 0.792 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-39_inv-2_part-mag_MP2RAGE | INV2 | 0.806 | 0.793 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-39_UNIT1 | UNIT1 | 0.807 | 0.799 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-40_inv-1_part-mag_MP2RAGE | INV1 | 0.834 | 0.821 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-40_inv-2_part-mag_MP2RAGE | INV2 | 0.831 | 0.811 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-40_UNIT1 | UNIT1 | 0.844 | 0.832 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-41_inv-1_part-mag_MP2RAGE | INV1 | 0.820 | 0.817 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-41_inv-2_part-mag_MP2RAGE | INV2 | 0.833 | 0.827 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |
| sub-41_UNIT1 | UNIT1 | 0.825 | 0.819 | C2;C3;C4;C5;C6;C7;C8;T1 | C2;C3;C4;C5;C6;C7;C8;T1 |

Notes:

- Dice is computed on the supplied prediction/reference grids with no post-processing.
- A level absent from both prediction and reference is excluded (NA); a missed reference level scores 0.
- The official `nnUNetv2_evaluate_folder` output may report a NaN aggregate when classes are absent from some images; these tables aggregate finite per-image values explicitly.
