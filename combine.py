# %%
import pandas as pd
import os


dir = "results_poster"
files = os.listdir(dir)
tables = []
cols = ["nruns","J", "T", "N", "level_bounds", "partial_effect_ratio", "staircase", "RegX"]
#%%
for fname in files:
    if "nruns=70" in fname:
        tb = pd.read_csv(f"{dir}/{fname}", index_col=[0, 1])
        for col in cols:
            if col in ["staircase", "RegX"]:
                # for booleans
                tb[col] = fname.split(col + "=")[1].split("_")[0].strip(".csv")
            else:
                # for floats
                tb[col] = float(fname.split(col + "=")[1].split("_")[0].strip(".csv"))
        tables.append(tb)


pd.concat(tables).to_csv(f"{dir}/full_result.csv")
tmp = (
    pd.concat([t.loc["mean"] for t in tables])
    .drop(["ins_mse", "ins_mae", "oos_mse", "oos_mae"], axis=1)
    .sort_values(by=cols, ascending=True)
)
import numpy as np
# tmp['SNR'] = np.sqrt(tmp['SNR'] * 2 / 300)
tmp.to_csv(f"{dir}/mean_result_70.csv")

# %%

frame30 = pd.read_csv((f"{dir}/mean_result_30.csv")).set_index(['method', 'partial_effect_ratio', 'level_bounds'])
frame70 = pd.read_csv((f"{dir}/mean_result_70.csv"))
# %%
frame30
# %%
(frame70[['method', 'partial_effect_ratio', 'level_bounds']] != frame30[['method', 'partial_effect_ratio', 'level_bounds']]).sum()
# %%
frame30
# %%
