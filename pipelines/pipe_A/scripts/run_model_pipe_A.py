import os
import sys
import yaml
import json
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from numpy.random import default_rng
import gitinfo
from proteolizarddata.data import PyTimsDataHandleDDA
from pystoms.aligned_feature_data import AlignedFeatureData
from pystoms.plotting import plot_marginals
file_path = os.path.dirname(__file__)
plt.style.use("ggplot")
# dependency paths for metadata metrics
prot_data_path = os.path.join(file_path,"../../../../proteolizard-data/")
prot_algo_path = os.path.join(file_path,"../../../../proteolizard-algorithm")
pymc_path = os.path.join(file_path,"../../../../pymc")

# construct output paths
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
model_parameters = params["model_parameters"]
random_seed = params["random_seed"]
rng = default_rng(random_seed)
# load data
dh = PyTimsDataHandleDDA(dp)
metrics_dictionary = {}
metrics_plot_list = []
divergencies_list =  []
sampling_times = []
loo_list = []
for feature_id in feature_ids:
    try:
        aligned_features = AlignedFeatureData(
                                dh,
                                ids=[feature_id],
                                is_parallel=False)
    except ValueError:
        continue
    if model_name == "M1":
        from pystoms.models_3d.model_3d_m1 import ModelM1
        model = ModelM1(aligned_features,model_parameters, random_number_generator = rng)
    elif model_name == "M2":
        from pystoms.models_3d.model_3d_m2 import ModelM2
        model = ModelM2(aligned_features,model_parameters, random_number_generator = rng)
    else:
        raise ValueError("Unknown model name.")
    # sample posterior
    model_trace, sampling_time = model.evaluation(prior_pred_in=True,
                     prior_pred_out=True,
                     posterior_pred_in=True,
                     posterior_pred_out=True,
                     plots =  ["prior_pred_in","prior_pred_out", "posterior_pred_in","posterior_pred_out"],
                     write_to_file=True,
                     folder_path=plotly_path)
    # generate png plot files
    model.arviz_plots(path=plot_path)
    # generate feature plot
    plot_marginals(aligned_features,plot_path=plot_path)
    # store recorded traces to files, if desired
    if save_traces:
        model_trace.to_netcdf(f"{idata_path}/{feature_id}_idata.nc")
    # record plot metrics for dvc plots
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
    sampling_times.append({"feature_id":str(feature_id),"sampling_time":sampling_time})

    elpd_data = az.loo(model_trace).to_dict()
    loo_list.append({"feature_id":str(feature_id),
                     "elpd":elpd_data["loo"],
                     "se": elpd_data["loo_se"],
                     "shape_warn": elpd_data["warning"]})

with open(f"{metrics_path}/metrics.json", "w") as json_file:
    jf = json.dumps(metrics_dictionary,indent=4)
    json_file.write(jf)
with open(f"{plot_metrics_path}/metrics_plot_dict.json","w") as json_file:
    jf = json.dumps({
        "plot_metrics": metrics_plot_list
    }, indent=4)
    json_file.write(jf)
with open(f"{plot_metrics_path}/divergencies.json","w") as json_file:
    jf = json.dumps({
        "divs": divergencies_list
    }, indent=4)
    json_file.write(jf)
with open(f"{plot_metrics_path}/sampling_times.json","w") as json_file:
    jf = json.dumps({
        "times": sampling_times
    }, indent=4)
    json_file.write(jf)
with open(f"{plot_metrics_path}/loo.json","w") as json_file:
    jf = json.dumps({
        "loo": loo_list
    }, indent=4)
    json_file.write(jf)
with open(f"{metrics_path}/metadata.json","w") as json_file:
    # get pystoms git info
    pystoms_git = gitinfo.get_git_info()
    prot_data_git = gitinfo.get_git_info(prot_data_path)
    prot_algo_git = gitinfo.get_git_info(prot_algo_path)
    pymc_git = gitinfo.get_git_info(pymc_path)
    python_interpreter = sys.executable
    pymc_stable_version = pm.__version__
    jf = json.dumps({
        "pystoms": {
                "message":pystoms_git["message"],
                "commit":pystoms_git["commit"]
        },
        "proteolizard-data":{
                "message":prot_data_git["message"],
                "commit":prot_data_git["commit"]
        },
        "proteolizard-algo":{
                "message":prot_algo_git["message"],
                "commit":prot_algo_git["commit"]
        },
        "pymc":{
                "message":pymc_git["message"],
                "commit":pymc_git["commit"]
        },
        "python_env_path":python_interpreter,
        "pymc_stable_version": pymc_stable_version
    }, indent=4)
    json_file.write(jf)