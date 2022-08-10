import pymc3 as pm
from scipy.stats import norm
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


times_CPU = []
times_model = []
traces_CPU = []


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

        tcss = time()
        for i in range(d1):
            data_i = obs_data[i]
            pm.set_data({"data": data_i})

            traces_CPU.append(pm.sample(return_inferencedata=True, random_seed=2022))
        tcse = time()
        times_CPU.append(tcse - tcss)


df = pd.DataFrame(
    {
        "D1": D1,
        "D2": D2,
        "D3": np.repeat(D3, n_shapes),
        "CPU_times": times_CPU,
        "Model_build_times": times_model,
    }
)
df.to_csv(f"dataCPUD3{D3}.csv")
