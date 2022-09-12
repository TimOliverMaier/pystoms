import os
import yaml
from proteolizarddata.data import PyTimsDataHandleDDA
from pystoms.aligned_feature_data import AlignedFeatureData

# construct output path
file_path = os.path.dirname(__file__)
relative_output_path = "../output_data/"
output_path = os.path.join(file_path,relative_output_path)
# construct path to params.yaml file
relative_param_path = "../params.yaml"
param_path = os.path.join(file_path,relative_param_path)
params = yaml.safe_load(open(param_path))["run_model"]
# load parameters
dp = params["data_path"]
feature_ids = params["feature_ids"]
model_name = params["model_name"]
# load data
dh = PyTimsDataHandleDDA(dp)

for feature_id in feature_ids:
    aligned_features = AlignedFeatureData(
                                dh,
                                ids=[feature_id],
                                is_parallel=False)
    if model_name == "M1":
        from pystoms.models_3d.model_3d_m1 import ModelM1
        model = ModelM1(aligned_features)
    elif model_name == "M2":
        from pystoms.models_3d.model_3d_m2 import ModelM2
        model = ModelM2(aligned_features)
    else:
        raise ValueError("Unknown model name.")
    model.evaluation(prior_pred_in=True,
                     prior_pred_out=True,
                     posterior_pred_in=True,
                     posterior_pred_out=True,
                     plots =  ["prior_pred_in","prior_pred_out", "posterior_pred_in","posterior_pred_out"],
                     write_to_file=True,
                     folder_path=output_path)
    model.arviz_plots(path=output_path)


