"""This script is supposed to evaluate a
GLM feature model snapshot (commit 3f05ba86418357b7d15b02d37ed59f3d6eadf077)
"""
import sys
import subprocess
import json
import os
from time import localtime
import numpy as np
import xarray as xa
import pandas as pd
from typing import List
from pyproteolizard.data import PyTimsDataHandle
from pystoms.feature_loader_dda import FeatureLoaderDDA
from pystoms.models_glm import ModelGLM3D
from logging import warning

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

# Class holding data of one or more features
class AlignedFeatureData():

    def __init__(self,ids:List[int],
                 is_parallel:bool=True,
                 num_data_points = 10):
        accepted_feature_ids = []
        charges = []
        feature_data = []
        for feature_id in ids:
            feature = FeatureLoaderDDA(data_handle,feature_id)
            if np.isnan(feature.monoisotopic_mz):
                continue
            try:
                # estimate feature hull boundaries with
                # averagine model for isotopic pattern
                # and gaussian model for IMS
                data_tmp = feature.load_hull_data_3d(intensity_min=0,
                                                     ims_model="gaussian",
                                                     plot_feature=False,)
            except RuntimeError:
                warning(f"RuntimeError with feature {feature_id}")
                continue
            if is_parallel:
                if len(data_tmp)<num_data_points:
                    continue
                feature_data.append(data_tmp.nlargest(num_data_points,
                                                  columns="Intensity"))
            else:
                feature_data.append(data_tmp)
            accepted_feature_ids.append(feature_id)
            charges.append(feature.charge)

        if len(accepted_feature_ids) == 0:
            raise ValueError("No accepted features. Check chosen features.")
        s,mz,i = self._set_data_parallel(feature_data)
        data_dict = {
            "Charge" : ("feature",charges),
            "Scan" : (("data_point","feature"),s),
            "Mz" : (("data_point","feature"),mz),
            "Intensity" : (("data_point","feature"),i),

        }
        coord_dict = {"feature":accepted_feature_ids}
        self.feature_data = xa.Dataset(data_vars = data_dict,
                                         coords = coord_dict)

    def _set_data_parallel(self,feature_data_list:List[pd.DataFrame]):
        """

        Args:
            feature_data_list (List[pd.DataFrame]): _description_
        """
        s = np.stack([fd["Scan"] for fd in feature_data_list],axis=1)
        mz = np.stack([fd["Mz"] for fd in feature_data_list],axis=1)
        i = np.stack([fd["Intensity"] for fd in feature_data_list],axis=1)\
                    .astype("float")

        return(s,mz,i)

def standardize(array,axis=0, mean = 0, std = 1):
    std_zero = (array-array.mean(axis=axis))/np.sqrt(array.var(axis=axis))
    return std_zero*std+mean

def generate_model(dataset:xa.Dataset,
                   num_isotopic_peaks:int = 6,
                   standardize:bool=False):
    # get dimensions
    num_features = dataset.dims["feature"]
    num_datapoints = dataset.dims["data_point"]
    # get observed data
    scans = dataset.Scan.values
    intensities = dataset.Intensity.values
    mzs = dataset.Mz.values.reshape((num_datapoints,
                              num_features,
                              1))
    charges = dataset.Charge.values
    feature_ids = dataset.feature.values
    # hyperpriors
    if standardize:
        pass
    else:
        # reshape is necessary here, because average deletes first
        # dimension
        ims_mu = np.average(scans,axis=0,weights=intensities)\
                    .reshape((1,num_features))
        ims_sigma_max = np.max(scans,axis=0)-np.min(scans,axis=0)\
                    .reshape((1,num_features))
        mz_mu = np.average(mzs.reshape((num_data_points,num_features)),
                           axis=0,
                           weights=intensities
                           ).reshape((1,
                                    num_features,
                                    1))
        mz_sigma = np.ones((1,num_features,1),dtype="float")*10
        alpha_lam = np.ones((1,num_features),dtype="float")\
                    *1/intensities.max(axis=0)
        model_error = np.ones((1,num_features),dtype="float")*10
        z = np.array(charges).reshape((1,num_features,1))
        mzs_tile = np.tile(mzs,(1,1,num_isotopic_peaks))
        peaks = np.arange(num_isotopic_peaks)
        peaks = peaks.reshape((1,1,num_isotopic_peaks))
        peaks_tile = np.tile(peaks,(num_datapoints,num_features,1))

    return ModelGLM3D(
                                    z,
                                    feature_ids,
                                    intensities,
                                    scans,
                                    mzs_tile,
                                    peaks_tile,
                                    ims_mu,
                                    ims_sigma_max,
                                    mz_mu,
                                    mz_sigma,
                                    alpha_lam,
                                    model_error
                                    )


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
    dataset_parallel = AlignedFeatureData(ids=feature_ids,
                                          num_data_points=num_data_points)\
                                        .feature_data
    dataset_parallel.to_netcdf(folder_output_path+"/parallel/input_data.nc")
    model_1 = generate_model(dataset_parallel)
    model_1.evaluation(True,True,True,True,["prior_pred_in","prior_pred_out", "posterior_pred_in","posterior_pred_out"],pred_name_list=["mu","obs"],write_to_file=True,folder_path=folder_output_path+"/parallel/")
    model_1.arviz_plots(save_fig=True,path=folder_output_path+"/parallel/")
    # not parallel
    for feature in feature_ids:
        try:
            dataset_single = AlignedFeatureData(
                                ids=[feature],
                                num_data_points=num_data_points)\
                                .feature_data
        except ValueError:
            warning(f"Skipping feature {feature}.")
            continue
        dataset_single.to_netcdf(folder_output_path+f"/single/input_data_{feature}.nc")
        model_2 = generate_model(dataset_single)
        model_2.evaluation(True,True,True,True,["prior_pred_in","prior_pred_out", "posterior_pred_in","posterior_pred_out"],pred_name_list=["mu","obs"],write_to_file=True,folder_path=folder_output_path+"/single/")
        model_2.arviz_plots(save_fig=True,path=folder_output_path+"/single/")
