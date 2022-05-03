"""This script is supposed to evaluate a
GLM feature model snapshot
"""
import sys
import subprocess
import json
import os
from time import localtime
import numpy as np
from pystoms.aligned_feature_data import AlignedFeatureData
from logging import warning
from pyproteolizard.data import PyTimsDataHandle

# user set meta data, please edit:
folder_output_path ="/home/tim/Master/prototyping/TutorialsAndArchive/evaluation_1_0/old_features/"# pylint: disable=line-too-long
description = "Initial Test with hardcoded features"
data_path = "/home/tim/Master/MassSpecDaten/M210115_001_Slot1-1_1_850.d/"
random_seed = 29042022
number_of_features = 20
num_data_points = 10
random_features = False
user_selected_feature_ids = [200,20,2011,2016,506,302,120]


# METADATA OF RUN, do not edit below
time_stamp = localtime()
meta_data = {

    "file_name" : __file__,
    "description" : description,
    "year" : time_stamp.tm_year,
    "month" : time_stamp.tm_mon,
    "day" : time_stamp.tm_mday,
    "Time" : str(time_stamp.tm_hour)\
             +":"\
             +str(time_stamp.tm_min)\
             +":"\
             +str(time_stamp.tm_sec),
    "python_exec" : sys.executable,
}

important_modules = ["pystoms","pyproteolizard","pymc","arviz","aesara"]

for module in important_modules:
    meta_data[module] = sys.modules[module].__version__
    meta_data[module+"_current_commit"] = subprocess.run(
                                            ["git","log","-n","1"],
                                            text=True,
                                            cwd=sys.modules[module]\
                                                   .__path__[0],
                                            check=True,
                                            capture_output=True).stdout

if not os.path.exists(folder_output_path):
    os.makedirs(folder_output_path)

with open(folder_output_path+"/metadata.json","w",encoding="utf_8") as jsonfile:
    json.dump(meta_data,jsonfile,indent=2)

# set random seed numpy
rng = np.random.default_rng(random_seed)

if __name__ == "__main__":

    # get raw data via proteolizard
    data_handle = PyTimsDataHandle(data_path)
    # precursors are listed in precursor table
    precursor_table = data_handle.get_selected_precursors()

    # extract some features with random ids
    if random_features:
        feature_ids = rng.integers(low = 0,
                                   high = precursor_table.Id.max(),
                                   size = number_of_features)
    else:
        feature_ids = user_selected_feature_ids
    # parallel
    parallel_features = AlignedFeatureData(data_handle,
                                           ids=feature_ids,
                                           num_data_points=num_data_points)
    parallel_features\
        .feature_data\
        .to_netcdf(folder_output_path+"/parallel/input_data.nc")
    model_1 = parallel_features.generate_model()
    model_1.evaluation(
                prior_pred_in=True,
                prior_pred_out=True,
                posterior_pred_in=True,
                posterior_pred_out=True,
                plots = ["prior_pred_in",
                         "prior_pred_out",
                         "posterior_pred_in",
                         "posterior_pred_out"],
                pred_name_list=["mu","obs"],
                write_to_file=True,
                folder_path=folder_output_path+"/parallel/")
    model_1.arviz_plots(save_fig=True,path=folder_output_path+"/parallel/")
    # not parallel
    for feature in feature_ids:
        try:
            single_feature = AlignedFeatureData(
                                data_handle,
                                ids=[feature],
                                num_data_points=num_data_points)
        except ValueError:
            warning(f"Skipping feature {feature}.")
            continue
        single_feature\
            .feature_data\
            .to_netcdf(folder_output_path+f"/single/input_data_{feature}.nc")
        model_2 = single_feature.generate_model()
        model_2.evaluation(
                prior_pred_in=True,
                prior_pred_out=True,
                posterior_pred_in=True,
                posterior_pred_out=True,
                plots = ["prior_pred_in",
                         "prior_pred_out",
                         "posterior_pred_in",
                         "posterior_pred_out"],
                pred_name_list=["mu","obs"],
                write_to_file=True,
                folder_path=folder_output_path+"/single/")
        model_2.arviz_plots(save_fig=True,path=folder_output_path+"/single/")
