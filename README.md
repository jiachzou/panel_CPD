## POSI for panel change-point detection
Files in this repo:
- ```run_experiments.py``` generates and compiles experiment results
- ```core.py``` contains the implementation loop of the change-point detection experiments
- ```dp.py``` contains the implementation of rDP.
- ```lasso.py``` contains corrections to ```selectinf``` package

Installation:
- Install anaconda
- Run ```bash install.sh```

Instructions to run the experiment:
- Run ```python run_experiments.py```
- Results can be found in ```results/main_result.csv```