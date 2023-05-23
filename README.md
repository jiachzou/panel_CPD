## POSI for Panel Change-Point Detection

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
