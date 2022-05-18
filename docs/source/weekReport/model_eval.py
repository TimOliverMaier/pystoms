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

# load user defined variables:
from evaluation_variables import * # pylint: disable=wildcard-import

# test for clean working directory
class DirtyGitWorkingDirectory(Exception):
    pass
wd_status = subprocess.run(
                        ["git","diff","HEAD"],
                        text=True,
                        cwd=sys.modules["pystoms"].__path__[0],
                        check=True,
                        capture_output=True).stdout
if wd_status != "":
    raise DirtyGitWorkingDirectory("Git working directory seems dirty. Abort.")

# gather meta data of run
time_stamp = localtime()
meta_data = {

    "evaluation version" : evaluation_version,
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
                                            ["git","rev-parse","HEAD"],
                                            text=True,
                                            cwd=sys.modules[module]\
                                                   .__path__[0],
                                            check=True,
                                            capture_output=True).stdout

folder_output_parallel = folder_output_path+"/parallel/"
folder_output_single = folder_output_path+"/single/"

if not os.path.exists(folder_output_parallel):
    os.makedirs(folder_output_parallel)
if not os.path.exists(f"{folder_output_parallel}/inference_data/"):
    os.makedirs(f"{folder_output_parallel}/inference_data/")
if not os.path.exists(folder_output_single):
    os.makedirs(folder_output_single)
if not os.path.exists(f"{folder_output_single}/inference_data/"):
    os.makedirs(f"{folder_output_single}/inference_data/")

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
        .to_netcdf(folder_output_parallel+"input_data.nc")
    model_1 = parallel_features.generate_model()
    parallel_feature_trace = model_1.evaluation(
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
                folder_path=folder_output_parallel)
    model_1.arviz_plots(save_fig=True,path=folder_output_parallel)
    parallel_feature_trace.to_netcdf(folder_output_parallel+"/inferencedata.nc")
    # save inference data
    accepted_feature_ids = parallel_features.accepted_feature_ids
    for feature in accepted_feature_ids:
        trace = parallel_feature_trace.sel(feature=feature)
        trace.to_netcdf(f"{folder_output_parallel}/inference_data/inference_data_{feature}.nc")
    # not parallel
    for feature in feature_ids:
        try:
            single_feature = AlignedFeatureData(
                                data_handle,
                                ids=[feature],
                                is_parallel=False)
        except ValueError:
            warning(f"Skipping feature {feature}.")
            continue
        single_feature\
            .feature_data\
            .to_netcdf(folder_output_single+f"input_data_{feature}.nc")
        model_2 = single_feature.generate_model()
        trace = model_2.evaluation(
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
                folder_path=folder_output_single)
        model_2.arviz_plots(save_fig=True,path=folder_output_single)

        # save inference data
        trace.to_netcdf(
            f"{folder_output_single}/inference_data/inference_data_{feature}.nc"
            )
