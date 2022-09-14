import os
import yaml
import json
from proteolizarddata.data import PyTimsDataHandleDDA
from pystoms.aligned_feature_data import AlignedFeatureData
from arviz import InferenceData
# construct output path
file_path = os.path.dirname(__file__)
relative_output_path = "../output_data/"
output_path = os.path.join(file_path,relative_output_path)
plot_path = os.path.join(output_path,"plots")
metrics_path = os.path.join(output_path,"metrics")
idata_path = os.path.join(output_path,"inference_data")
plot_metrics_path = os.path.join(output_path,"plot_metrics")
plotly_path = os.path.join(output_path,"plotly")
if not os.path.exists(plot_path):
    os.mkdir(plot_path)
if not os.path.exists(metrics_path):
    os.mkdir(metrics_path)
if not os.path.exists(idata_path):
    os.mkdir(idata_path)
if not os.path.exists(plot_metrics_path):
    os.mkdir(plot_metrics_path)
if not os.path.exists(plotly_path):
    os.mkdir(plotly_path)
# construct path to params.yaml file
relative_param_path = "../params.yaml"
param_path = os.path.join(file_path,relative_param_path)
params = yaml.safe_load(open(param_path))["run_model"]
# load parameters
dp = params["data_path"]
feature_ids = params["feature_ids"]
model_name = params["model_name"]
save_traces = params["save_traces"]
# load data
dh = PyTimsDataHandleDDA(dp)
metrics_dictionary = {}
metrics_plot_list = []
divergencies_list =  []
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
    model_trace: InferenceData = model.evaluation(prior_pred_in=True,
                     prior_pred_out=True,
                     posterior_pred_in=True,
                     posterior_pred_out=True,
                     plots =  ["prior_pred_in","prior_pred_out", "posterior_pred_in","posterior_pred_out"],
                     write_to_file=True,
                     folder_path=plotly_path)
    model.arviz_plots(path=plot_path)
    if save_traces:
        model_trace.to_netcdf(f"{idata_path}/{feature_id}_idata.nc")
    stats = model_trace.sample_stats
    feature_dictionary = {}
    feature_dictionary["divergences"] = int(stats.diverging.values.sum())
    feature_dictionary["acceptance_rate"] = float(stats.acceptance_rate.values.mean())
    metrics_dictionary[feature_id] = feature_dictionary

    acc_rate = stats.acceptance_rate.values.flatten().tolist()
    tree_depth = stats.tree_depth.values.flatten().tolist()
    n_steps = stats.n_steps.values.flatten().tolist()
    step_size = stats.step_size.values.flatten().tolist()
    metrics_plot_list += [{"feature_id":str(feature_id),"tree_depth":td,"acceptance_rate":ar,"n_steps":ns,"step_size":ss} for (td,ar,ns,ss) in zip(tree_depth,acc_rate,n_steps,step_size) ]
    divergencies_list.append({"feature_id":str(feature_id),"divergencies":feature_dictionary["divergences"]})
    
    
with open(f"{metrics_path}/metrics.json", "w") as json_file:
    jf = json.dumps(metrics_dictionary,indent=4)
    json_file.write(jf)
with open(f"{plot_metrics_path}/metrics_plot_dict.json","w") as json_file:
    jf = json.dumps({
        "plot_metrics": metrics_plot_list
    },indent=4)
    json_file.write(jf)
with open(f"{plot_metrics_path}/divergencies.json","w") as json_file:
    jf = json.dumps({
        "divs": divergencies_list
    },indent=4)
    json_file.write(jf)
