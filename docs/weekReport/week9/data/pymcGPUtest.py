import pymc as pm
from scipy.stats import norm
import pymc.sampling_jax
import numpy as np
from time import time
import pandas as pd

np.random.seed(2022)

n = 640
n_shapes = 8

D2 = [1, 10, 20, 40, 80, 160, 320, 640]
D3 = 10
D1 = [640, 64, 32, 16, 8, 4, 2, 1]
obs_data_list = []

data = np.zeros((n, 1, D3))

for i in range(n):
    m = (np.random.random() - 0.5) * 20
    data_i = norm(m, 1).rvs(D3)
    data[i] = data_i

dataShapes = []

for d1, d2 in zip(D1, D2):
    dataShapes.append(data.reshape((d1, d2, D3)))


times_GPU = []
times_GPUv = []
times_model = []
traces_GPU = []
traces_GPUv = []


for d1, d2, obs_data in zip(D1, D2, dataShapes):
    tccs = time()  # time cpu model compile start
    with pm.Model() as model:
        data = pm.Data("data", obs_data[0])
        mus = pm.Normal("mus", 0, 20, shape=d2)
        normals = [
            pm.Normal(f"obs_{i}", mu=mus[i], sigma=1, observed=data.get_value()[i])
            for i in range(d2)
        ]
    tcce = time()
    times_model.append(tcce - tccs)

    with model as model:

        tgss = time()
        for i in range(d1):
            data_i = obs_data[i]
            pm.set_data({"data": data_i})
            traces_GPU.append(pm.sampling_jax.sample_numpyro_nuts(random_seed=2022))
        tgse = time()
        times_GPU.append(tgse - tgss)
        tgvss = time()
        for i in range(d1):
            data_i = obs_data[i]
            pm.set_data({"data": data_i})
            traces_GPUv.append(
                pm.sampling_jax.sample_numpyro_nuts(
                    chain_method="vectorized", random_seed=2022
                )
            )
        tgvse = time()
        times_GPUv.append(tgvse - tgvss)


df = pd.DataFrame(
    {
        "D1": D1,
        "D2": D2,
        "D3": np.repeat(D3, n_shapes),
        "GPU_times": times_GPU,
        "GPUv_times": times_GPUv,
        "Model_build_times": times_model,
    }
)
df.to_csv(f"dataGPUD3{D3}.csv")
