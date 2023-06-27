## Panel Multiple Testing for Panel Change-Point Detection

### Usage

To cite this code, please use the following two citations:
1. The core functionalities of this repo replicates a working paper at non-archival workshop SPIGM @ ICML 2023,
```
@article{zouyangpelger2023cpd,
  title={Large Dimensional Change Point Detection with \\FWER Control as Automatic Stopping},
  author={Zou, Jiacheng and Fan, Yang and Pelger, Markus},
  journal={ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling},
  year={2023}
}
```

2. The underlying statistical method is introduced in,
```
@article{pelger2022inference,
  title={Inference for Large Panel Data with Many Covariates},
  author={Pelger, Markus and Zou, Jiacheng},
  journal={arXiv preprint arXiv:2301.00292},
  year={2022}
}
```

### Repository Files
- `run_experiments.py`: Generates and compiles experiment results.
- `core.py`: Contains the implementation loop of the change-point detection experiments.
- `lasso.py`: Contains corrections to the `selectinf` package.

### Installation
1. Install Anaconda.
2. Run the command `bash install.sh` in your terminal. This will create an environment named `panel_CPD` with all the necessary dependencies for you.

### Instructions to Run the Experiment
1. Open your terminal.
2. Activate the `panel_CPD` environment using the command `conda activate panel_CPD`.
3. Run the command `python run_experiments.py`.
4. The results of the experiment will be saved in the file `results/main_result.csv`.
