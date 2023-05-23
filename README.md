## POSI for change-point detection
Files in this repo:
- ```main.ipynb``` contains the cleaned experiment results, the go-to file for our model performance.
- ```core.y``` contains the implementation loop of the change-point detection experiments, primarily the ```experiment()``` function.
- ```demo.ipynb``` contains development code. You should ignore this.
- ```dp.y``` contains the implementation of rDP. ```dp.m``` contains the original matlab implementation.
- ```multiple_testing.y``` contains the implementation of POSI.

To implement
 - find alternative to fuzzy rdp
 - add rmse, in sample out-of sample refit. out of sample, crosssection choose from N

Installation on soal
- install anaconda
- create and activate your conda env, this may take a while, run ```conda create -n env_name python=3.9``` to ensure python version is 3.9
- run ```pip install -r requirements.txt```
- run ```conda install -c conda-forge regreg```, this may take a while
- run ```git clone https://github.com/selective-inference/Python-software.git```
- before installation, replace ```selectinf/algorithms/lasso.py``` with my version
    ```
- enter ```Python-software``` directory, run 

    ```
    git submodule init 
    git submodule update
    pip install -r requirements.txt
    ```


- run ```python setup.py install```, ignore warnings
- run ```pip install ipykernel matplotlib```
- run ```conda install libgcc```
- run ```conda install statsmodels```
- run ```conda install -c conda-forge gcc```
- run ```python install_rpy_package.py```, select yes if necessary
- run ```conda install -c conda-forge matplotlib```
- you should be able to run ```tests_for_selectinf/testlasso.py``` to generate ```test_done_lasso.png``` for a correct installation
- run ```python -m ipykernel install --user --name env_name``` to add your env to jupyterlab on soal if you are using it.


