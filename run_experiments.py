from core import experiment, generate_data, parallel_experiments

# import os
# os.environ["OMP_NUM_THREADS"] = "30" # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = "30" # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = "30" # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = "30" # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = "30"

import numpy as np

for partial_effect_ratio in [0.5,1]:
    for level_bounds in [1,2,5,10]:
        for n in np.array([2e2]).astype(int):
            parallel_experiments(
                n_runs=30,
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
                RegX=False
            )


# quick test
# parallel_experiments(
#                 n_runs=1,
#                 n_jobs=2,
#                 N=20,
#                 T=100,
#                 n_jumps=10,
#                 level_bounds=1,
#                 min_gaps=0,
#                 partial_effect_ratio=1,
#                 heavy_tail=False,
#                 poission_corruption=False,
#                 J=0.8
#             )
