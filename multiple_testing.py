# %%
import numpy as np
import numpy.random as rnd
from numpy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LassoCV, Lasso, LassoLarsCV
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
from scipy.stats import truncnorm, norm
from scipy.linalg import toeplitz
from sklearn.metrics import r2_score
from scipy.stats import norm


np.seterr(divide="ignore")
rnd.seed(1)
# %%


def generate_fake_data(
    N, T_obs, s=10, strong_factor_uniform_bound=0.5, noise_std=1, xsec_cov=0.5
):
    d = T_obs
    full_beta = np.zeros((d, N))

    splitted = np.split(np.arange(N), s)

    for i_factor in range(s):
        curr_units = np.concatenate(splitted[-(i_factor):])

        # active_units = curr_units[rnd.uniform(0,1,len(curr_units))>0.5]
        full_beta[i_factor, curr_units] = rnd.uniform(
            low=-strong_factor_uniform_bound,
            high=strong_factor_uniform_bound,
            size=len(curr_units),
        )

    noise_cov = np.eye(N) * noise_std

    for i_unit in range(N):
        other_factor = np.delete(np.arange(N), i_unit)
        for j_factor in other_factor:
            noise_cov[i_unit, j_factor] = xsec_cov
    noises = rnd.multivariate_normal(mean=np.zeros(N), cov=noise_cov, size=T_obs)

    # all_covariates = rnd.uniform(low=0,high=strong_factor_uniform_bound,size=(T_obs,d))
    # all_covariates = rnd.normal(size=(T_obs,d))
    all_covariates = np.tril(np.ones((T_obs, T_obs)), k=0)

    all_response = np.matmul(all_covariates, full_beta) + noises
    return all_response, all_covariates


def generate_fake_log_pvals(response_, covariates_, s=10):
    J = covariates_.shape[1]
    N = response_.shape[1]
    pval_matrix = rnd.uniform(size=(J, N))  # J by N
    for j in range(J):
        if j > s - 1:
            masking_to_nan = np.sort(
                rnd.choice(a=range(N), size=int(0.8 * N), replace=False)
            )
            pval_matrix[j, masking_to_nan] = np.nan
        else:
            pval_matrix[j, :] = 1 - norm.cdf(rnd.uniform(2, 5, N))
            masking_to_nan = np.sort(
                rnd.choice(a=range(N), size=int(0.5 * N), replace=False)
            )
            pval_matrix[j, masking_to_nan] = np.nan

    return np.log(pval_matrix)


def panel_posi_unordered(log_pval_matrix, gamma):
    log_pval_matrix = log_pval_matrix.copy()
    M_set = (~np.isnan(log_pval_matrix)).sum(axis=0)
    K_set = (~np.isnan(log_pval_matrix)).sum(axis=1)
    simultaneity_count_array = np.zeros(shape=log_pval_matrix.shape[0])
    for i in range(log_pval_matrix.shape[0]):
        simultaneity_count_array[i] = np.sum(
            M_set[np.where(~np.isnan(log_pval_matrix)[i, :])[0]]
        )

    log_pval_matrix[np.isnan(log_pval_matrix)] = np.inf
    smallest_log_pval_array = np.nanmin(log_pval_matrix, axis=1)
    rho_inv = np.sum(
        K_set[simultaneity_count_array > 0]
        / simultaneity_count_array[simultaneity_count_array > 0]
    )
    rho = 1 / rho_inv

    thresholds = np.log(gamma) - np.log(simultaneity_count_array) + np.log(rho)
    bonf_thresholds = (
        np.log(gamma)
        - np.log(log_pval_matrix.shape[0])
        - np.log(log_pval_matrix.shape[1])
    )

    selection_result = np.where(
        (smallest_log_pval_array <= thresholds) & (simultaneity_count_array > 0)
    )[0]
    bonf_selection_result = np.where(
        (smallest_log_pval_array <= bonf_thresholds) & (simultaneity_count_array > 0)
    )[0]

    return selection_result, rho, bonf_selection_result


# %%
# n_true_covariate=10
# response_,covariates_=generate_fake_data(d=2000,N=100,T_obs=300,s=n_true_covariate)

# logp_val_mat = generate_fake_log_pvals(response_,covariates_,s=n_true_covariate)

# selection_result, rho, bonf_selection_result=panel_posi_unordered(logp_val_mat,0.05)
# print('Truth',list(range(n_true_covariate)))
# print('Theorem 2',selection_result)
# print('rho',rho)
# print('Naive Bonf',bonf_selection_result)
# %%
