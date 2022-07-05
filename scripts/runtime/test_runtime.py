import pymc as pm
from pymc.sampling_jax import sample_numpyro_nuts
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from time import time
from pystoms.aligned_feature_data import AlignedFeatureData
from pyproteolizard.data import PyTimsDataHandle
from logging import warning

# PARAMETERS
data_path = "/home/tim/Master/MassSpecDaten/M210115_001_Slot1-1_1_850.d/"
data_handle = PyTimsDataHandle(data_path)
total_features = 100
feature_ids_flat = np.random.random_integers(1000,4000,size=total_features)
output_path = "test_runtime_output"

# test if output folder exists

if not os.path.exists(output_path):
    os.makedirs(output_path)

batch_n = [1,2,5,10,100]
times_per_feature_gpu = []
times_per_feature_cpu = []
num_eval_features = []

for n in batch_n:
    # distribute 100 features onto {n} batches
    feature_batchs = feature_ids_flat.reshape((n,total_features//n))

    total_time_cpu = 0
    total_time_gpu = 0
    evaluated_features = 0
    for batch_num,batch in enumerate(feature_batchs):
        try:
            aligned_features = AlignedFeatureData(
                                data_handle,
                                ids=batch,
                                is_parallel=True,
                                num_data_points=20)
        except ValueError:
            warning(f"Skipping feature {batch}.")
            continue
        evaluated_features += len(aligned_features.accepted_feature_ids)
        model = aligned_features.generate_model()
        time_start_gpu = time()
        trace_gpu = sample_numpyro_nuts(model=model,
                                            chain_method="vectorized")
        trace_gpu.to_netcdf(output_path+"/"+f"trace_batch_{n}_{batch_num}_gpu.nc")
        time_end_gpu = time()
        total_time_gpu  += time_end_gpu-time_start_gpu

        time_start_cpu = time()
        trace_cpu = pm.sample(model=model)
        trace_cpu.to_netcdf(output_path+"/"+f"trace_batch_{n}_{batch_num}_cpu.nc")
        time_end_cpu = time()
        total_time_cpu += time_end_cpu-time_start_cpu
    times_per_feature_gpu.append(total_time_gpu/evaluated_features)
    times_per_feature_cpu.append(total_time_cpu/evaluated_features)
    num_eval_features.append(evaluated_features)
# plot
plt.style.use("ggplot")


plot_df = pd.DataFrame({"batch_number":batch_n,
                        "GPU":times_per_feature_gpu,
                        "CPU":times_per_feature_cpu})\
                    .melt(id_vars="batch_size",
                          var_name="Label",
                          value_name="Time")

Fig,ax = plt.subplots()
Fig.set_dpi(300)
sns.pointplot(data=plot_df,
              x="batch_number",
              y="Time",
              ax=ax,
              hue="Label",
              markers=["^","o"])

ax.set_xlabel("Batch Number")
ax.set_ylabel("Runtime per Feature [s]")
ax.set_ylim(0,max(plot_df.Time)+5)
plt.savefig(output_path+"/"+"runtime_comp.png")
