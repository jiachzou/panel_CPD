#!/bin/bash

# Activate conda environment
conda create -n env_name python=3.9
conda activate env_name

# Install Python dependencies using pip
pip install -r requirements.txt

# Install regreg using conda-forge
conda install -c conda-forge regreg

# Clone the repository
git clone https://github.com/selective-inference/Python-software.git

# Replace lasso.py with your version
cp lasso.py Python-software/selectinf/algorithms/lasso.py

# Enter Python-software directory
cd Python-software

# Initialize and update submodules
git submodule init
git submodule update

cd ..
# Install Python dependencies in Python-software directory
pip install -r requirements.txt

# Run python setup.py install, ignoring warnings
python setup.py install -W ignore

# Install ipykernel and matplotlib using pip
pip install ipykernel matplotlib

# Install libgcc using conda
conda install libgcc

# Install statsmodels using conda
conda install statsmodels

# Install gcc using conda-forge
conda install -c conda-forge gcc

# Run python install_rpy_package.py and select yes if necessary
python install_rpy_package.py

# Install matplotlib using conda-forge
conda install -c conda-forge matplotlib
