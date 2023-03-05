# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from math import ceil, floor, log
import plotly.express as px


# %%
def closest_power_of_2(x):
    return 1 if x == 0 else 2 ** round(log(x, 2))


# %%
full_results_dict = dict(
    loss=[],
    n_filters=[],
    dataset_fraction=[],
    compute=[],
)
# %%
for p in range(0, 5):
    for n in [11, 16, 23, 32, 45, 64, 90, 128, 180]:
        try:
            print(f"Trying n_filters={n}, dataset_fraction={(2**-p):3f}")
            results = torch.load(
                f"results/mnist_model_final_n_filters_{n}_dataset_frac_{(2**-p):3f}_results.pt"
            )
            full_results_dict["loss"].append(results["loss"])
            full_results_dict["n_filters"].append(results["n_filters"])
            full_results_dict["dataset_fraction"].append(results["dataset_fraction"])
            full_results_dict["compute"].append(closest_power_of_2(n * n * (2**-p)))
        except:
            print(f"Failed to load n_filters={n}, dataset_fraction={(2**-p):3f}")
            continue

# %%
df = pd.DataFrame(full_results_dict)
# %%
px.scatter(df, x="compute", y="loss", color="n_filters", log_x=True, log_y=True)
# %%
px.scatter(df, x="loss", y="n_filters", color="compute", log_x=True, log_y=True)
# %%
