from core import parallel_experiments
import numpy as np
import pandas as pd
import os


def main():
    for partial_effect_ratio in [0.5, 1]:
        for level_bounds in [1, 2, 5, 10]:
            for n in np.array([2e2]).astype(int):
                parallel_experiments(
                    n_runs=100,
                    n_jobs=2,
                    N=n,
                    T=300,
                    n_jumps=10,
                    level_bounds=level_bounds,
                    min_gaps=0,
                    partial_effect_ratio=partial_effect_ratio,
                    heavy_tail=False,
                    poission_corruption=False,
                    J=0.8,
                    staircase=False,
                    RegX=True,
                )

    dir = "results"
    files = os.listdir(dir)
    tables = []
    cols = [
        "nruns",
        "J",
        "T",
        "N",
        "level_bounds",
        "partial_effect_ratio",
        "staircase",
        "RegX",
    ]

    for fname in files:
        if "nruns=" in fname:
            tb = pd.read_csv(f"{dir}/{fname}", index_col=[0, 1])
            for col in cols:
                if col in ["staircase", "RegX"]:
                    # for booleans
                    tb[col] = fname.split(col + "=")[1].split("_")[0].strip(".csv")
                else:
                    # for floats
                    tb[col] = float(
                        fname.split(col + "=")[1].split("_")[0].strip(".csv")
                    )
            tables.append(tb)

    pd.concat(tables).to_csv(f"{dir}/full_result.csv")
    tmp = (
        pd.concat([t.loc["mean"] for t in tables])
        .drop(["ins_mse", "ins_mae", "oos_mse", "oos_mae"], axis=1)
        .sort_values(by=cols, ascending=True)
    )
    tmp.to_csv(f"{dir}/main_result.csv")


if __name__ == "__main__":
    main()
