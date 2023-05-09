"""
This module is supposed to aid with comparison of
two or more models.
"""
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from typing import Dict


def read_idata_directory(directory_path: str) -> Dict[str, az.InferenceData]:
    """Reads all nc files in given directory as InferenceData.

    Args:
        directory_path (str): Path to folder with inference data.

    Returns:
        Dict[str,az.InferenceData]: Dictionary with file paths as keys
            and InferenceData as values.
    """
    inference_data_dic = {}
    for idata_file in os.listdir(directory_path):
        file_path = f"{directory_path}/{idata_file}"
        suffix = os.path.splitext(file_path)[1]
        if os.path.isfile(file_path) and suffix == ".nc":
            idata = az.from_netcdf(file_path)
            feature_id = idata.posterior.feature.values[0]
            inference_data_dic[feature_id] = idata
    return inference_data_dic


def compare_feature_model(
    models_to_compare: Dict[str, str], plot: bool = True
) -> pd.DataFrame:
    """Compares the performance of two
    or more peptide feature models

    Args:
        models_to_compare (Dict[str,str]): Dictionary with model names as
            keys and paths to inference data directory as values.
        plot (bool, optional): Wether to plot comparison.
    Returns:
        pd.DataFrame: Dataframe with comparison stats.
    """
    idata_of_models = {}
    # read in data
    for model_name, dir_path in models_to_compare.items():
        idata_of_models[model_name] = read_idata_directory(dir_path)
    # from {"model_name":{"feature_id":idata, …}, …}
    # to {"feature_id":{"model_name":idata, …}, …}
    comparisons = {}
    for model_name, idata_dict in idata_of_models.items():
        for feature_id in idata_dict:
            if feature_id not in comparisons:
                comparisons[feature_id] = {}
            comparisons[feature_id][model_name] = idata_dict[feature_id]
    # run comparisons per feature and concat dataframes
    comp_data = []
    for feature_id, idata_dict in comparisons.items():
        comp = az.compare(idata_dict)
        comp["feature_id"] = feature_id
        comp_data.append(comp)
    comp_df = pd.concat(comp_data)
    comp_df.reset_index(inplace=True)
    comp_df.rename(columns={"index": "model_name"}, inplace=True)
    if plot:
        sns.violinplot(x="model_name", y="loo", data=comp_df)
        plt.show()
        sns.boxplot(x="model_name", y="loo", data=comp_df, showfliers=False)
        plt.show()
    return comp_df
