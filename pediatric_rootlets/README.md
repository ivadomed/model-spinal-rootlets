## Pediatric rootlets

This README provides instructions for running the model for **_ventral_ and dorsal rootlets** on pediatric data 
and for generating a figure showing the rootlets and vertebral levels. 

### Description of 'pediatric_rootlets.sh' script
Script `pediatric_rootlets.sh` consists of the following steps:
 1. segmentation of rootlets from T2w data (model model-spinal-rootlets_ventral_D106_r20240523 - https://github.com/ivadomed/model-spinal-rootlets/releases/tag/r20240523),
 2. segmentation of spinal cord from T2w data (`sct_deepseg_sc`)
 3. labeling of intervertebral discs from T2w data (`sct_label_vertebrae`)
 4. detection of PMJ from T2w data (`sct_detect_pmj`)
 5. run `inter-rater_variability/02a_rootlets_to_spinal_levels.py` for rootlets to spinal levels on pediatric data
 6. run `pediatric_rootlets/discs_to_vertebral_levels.py` for disc to vertebral levels on pediatric data 

The `pediatric_rootlets.sh` can be run across multiple subjects using the following command:
``````commandline
sct_run_batch -path-data /path/to/data/ -path-output /path/to/output -script pediatric_rootlets.sh
``````

### Description of 'generate_figure_rootlets_and_vertevral_spinal_levels.py' script
Script `generate_figure_rootlets_and_vertevral_spinal_levels.py` generates a figure showing the rootlets and 
vertebral levels on the spinal cord. 

This script can be run the following command:
``````commandline
python generate_figure_rootlets_and_vertevral_spinal_levels.py -i /path/to/data/ -participants /path/to/participants.tsv
``````



