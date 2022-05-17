"""
This module is supposed to aid with comparison of
two or more models.
"""
import arviz as az
from logging import warning
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Dict

def read_idata_directory(directory_path:str) -> Dict[str,az.InferenceData]:
    """Reads all nc files in given directory as InferenceData.

    Args:
        directory_path (str): Path to folder with inference data.

    Returns:
        Dict[str,az.InferenceData]: Dictionary with file paths as keys
            and InferenceData as values.
    """
    inference_data_dic = {}
    for idata_file in os.listdir(directory_path):
        file_path = f"{directory_path}{idata_file}"
        suffix = os.path.splitext(file_path)[1]
        if os.path.isfile(file_path) and suffix == ".nc":
            inference_data_dic[idata_file] = az.from_netcdf(file_path)
    return inference_data_dic

def compare_feature_model(models_to_compare:Dict[str,str])->pd.DataFrame:
    """Compares the performance of two
    or more peptide feature models

    Args:
        models_to_compare (Dict[str,str]): Dictionary with model names as
            keys and paths to inference data directory as values.
    Returns:
        pd.DataFrame: Dataframe with comparison stats.
    """
    warning(f"This function is currently under construction, "
            f"it is only comparing the first features in the directories")
    first_features = [models_to_compare[key][0] for key in models_to_compare]
    comparison = az.compare(first_features)
    return comparison

