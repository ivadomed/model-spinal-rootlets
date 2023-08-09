import os
from segment_to_csv import main as segment_to_csv
import pandas as pd
from tqdm import tqdm
import numpy as np

path_out = '/Users/theomathieu/Downloads/cadotte_15/'
df_dict = {"level": [], "sub_name": [], "spinal_start": [], "spinal_end": [], "height": [],
           "PMJ_start": [], "PMJ_end": []}
for im in tqdm(os.listdir('/Users/theomathieu/Downloads/cadotte_15/pred/')):
    if im.startswith('.'):
        pass
    else:
        path_image = '/Users/theomathieu/Downloads/cadotte_15/extracted/' + im.split('.')[0] + '_0000.nii.gz'
        path_rootlet = '/Users/theomathieu/Downloads/cadotte_15/pred/' + im
        df_dict, im_name = segment_to_csv(path_image=path_image, path_temp='/tmp/seg_to_csv', path_out=path_out,
                                          df_dict=df_dict, path_rootlet=path_rootlet, rm=True)
df = pd.DataFrame(df_dict)
with open(f'{path_out}_height.log', 'w') as file:
    file.write(f"Spinal level\tMean (mm)\tStd (mm)\n")
    for lvl in range(2, 12):
        mean = np.mean(df["height"][df["level"] == lvl])
        std = np.std(df["height"][df["level"] == lvl])
        print(f"Height level {lvl} (mm): {mean:.2f} +/- {std:.2f}")
        file.write(f"{lvl}\t{mean:.2f}\t{std:.2f}\n")

df.to_csv(path_out + 'dist_pred.csv', index=False)
