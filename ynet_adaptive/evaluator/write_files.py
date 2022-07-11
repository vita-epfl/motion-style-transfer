import os
import csv
import pathlib
import numpy as np
from typing import List


def write_csv(out_dir, out_name, ade, fde, ade_final=None, fde_final=None):
    # file name
    out_path = os.path.join(out_dir, out_name)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # format data 
    out_data = [round_val(min(ade)), round_val(np.mean(ade))] + \
        [round_val(ade_val) for ade_val in ade]
    if ade_final is not None:
        out_data = [round_val(ade_final)] + out_data
    out_data = convert_to_str(out_data)
    with open(out_path, 'w') as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(out_data)


def get_out_dir(out_dir, dataset_path, seed, train_net, val_files, train_files=None):
    val_name = f"{'_'.join(['_'+f.split('.pkl')[0]+'_' for f in val_files])}"
    if train_files:
        train_name = f"{'_'.join(['_'+f.split('.pkl')[0]+'_' for f in train_files])}"
    else:
        train_name = "None"
    out_dir = os.path.join(out_dir, dataset_path, train_name.strip(
        "_"), val_name.strip("_"), train_net, str(seed))
    return out_dir 


def round_val(num: float, ndig: int = 4):
    if num is None:
        return 0.0
    else:
        return str(round(num, ndig))


def convert_to_str(in_list: List):
    out_list = []
    for i in in_list:
        if not isinstance(i, str):
            i = str(i)
        out_list.append([i])
    return out_list
