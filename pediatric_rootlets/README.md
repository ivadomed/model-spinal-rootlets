## Pediatric rootlets

This README provides instructions for running the model for **_ventral_ and dorsal rootlets** on pediatric data 
and for generating a figure showing the rootlets and vertebral levels on the spinal cord. 

### Description of 'pediatric_rootlets.sh' script
Script `pediatric_rootlets.sh` consists of the following steps:
 1. segmentation of rootlets from T2w data (model model-spinal-rootlets_ventral_D106_r20240523),
 2. segmentation of spinal cord from T2w data (contrast agnostic model)
 3. segmentation of vertebral levels from T2w data (`sct_label_vertebrae`)
 4. detection of PMJ from T2w data (`sct_detect_pmj`)
 5. run `02a_rootlets_to_spinal_levels.py` for rootlets to spinal levels on pediatric data
 6. run `02a_rootlets_to_spinal_levels.py` for vertebrae to spinal levels on pediatric data 

This script can be run by executing the following command:
``````commandline
sct_run_batch -path-data /path/to/data/ -path-output /path/to/output -script pediatric_rootlets.sh
``````

### Description of 'generate_figure_rootlets_and_vertevral_spinal_levels.py' script
Script `generate_figure_rootlets_and_vertevral_spinal_levels.py` generates a figure showing the rootlets and 
vertebral levels on the spinal cord. 

This script can be run by executing the following command:
``````commandline
python generate_figure_rootlets_and_vertevral_spinal_levels.py -i /path/to/data/
``````


