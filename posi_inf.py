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


def get_log_p(covariates, response):
    """
    covariates: design matrix X: T by d
    responses: outcome matrix y: T by N
    """
    d = covariates.shape[1]
    N = response.shape[1]
    T_obs_train_ = covariates.shape[0]
    posi_log_pval_matrix = np.nan * np.ones((d, N))
    # t_log_pval_matrix = np.nan * np.ones((d, N))
    # no prior
    omega_inv_vec = np.ones(d)
    for i_unit in range(N):
        lars = LassoLarsCV(cv=5, fit_intercept=False, normalize=False, max_n_alphas=d)
        lars_fitted = lars.fit(X=covariates, y=response[:, i_unit])
        mse_CVed = lars_fitted.mse_path_.mean(axis=1)
        picked_alpha = lars_fitted.cv_alphas_[
            np.max(np.where(mse_CVed <= 2 * min(mse_CVed)))
        ]

        lasso = Lasso(alpha=picked_alpha)
        lasso_fitted = lasso.fit(
            X=covariates,
            y=response[:, i_unit],
        )
        lasso_lambda = picked_alpha
        y = response[:, i_unit]
        active_set = lasso_fitted.coef_ != 0
        X_M = covariates[:, active_set]
        omega_inv_M = omega_inv_vec[active_set]
        omega_inv_notM = omega_inv_vec[~active_set]
        X_notM = covariates[:, ~active_set]
        X_M_card = X_M.shape[1]
        if X_M_card == 0:
            continue
        ols_post_lasso = OLS(endog=y, exog=X_M)
        ols_post_lasso_fitted = ols_post_lasso.fit()
        beta_bar = ols_post_lasso_fitted.params
        X_M_gram = np.matmul(X_M.transpose(), X_M)
        X_M_gram_inv = pinv(X_M_gram)
        X_M_pseudo_inv = np.matmul(X_M_gram_inv, X_M.transpose())
        this_df = max(1, T_obs_train_ - X_M_card)
        estimated_var = (
            np.sum(np.power(y - lasso_fitted.predict(covariates), 2)) / this_df
        )
        Sigma = np.eye(T_obs_train_) * estimated_var
        p_raw_vec = np.zeros(X_M_card)
        studentized_posi_vec, trunc_a_vec, trunc_b_vec = (
            np.zeros(X_M_card),
            np.zeros(X_M_card),
            np.zeros(X_M_card),
        )

        for i_covariate in range(X_M_card):
            eta = np.reshape(X_M_pseudo_inv[i_covariate, :], newshape=(T_obs_train_, 1))

            var_beta_bar = estimated_var * (X_M_gram_inv.diagonal())[i_covariate]
            std_beta_bar = np.sqrt(var_beta_bar)

            # print('Estimated sigma(beta)',std_beta_bar)
            xi = np.reshape(
                np.matmul(Sigma, eta) / var_beta_bar, newshape=(T_obs_train_, 1)
            )
            z = np.matmul(np.eye(T_obs_train_) - np.matmul(xi, eta.transpose()), y)
            s_vec = np.sign(lasso_fitted.coef_[active_set])
            P_M = np.matmul(X_M, X_M_pseudo_inv)
            # print("Dim of P_M is ", P_M.shape)
            reuseable_part1 = np.matmul(X_notM.transpose(), np.eye(T_obs_train_) - P_M)

            A_matrix = np.concatenate(
                [
                    reuseable_part1 / lasso_lambda,
                    -reuseable_part1 / lasso_lambda,
                    -np.matmul(np.diag(s_vec), X_M_pseudo_inv),
                ],
                axis=0,
            )

            reuseable_part2 = np.matmul(
                np.matmul(X_notM.transpose(), X_M_pseudo_inv.transpose()),
                s_vec / omega_inv_M,
            )

            b_vec = np.concatenate(
                [
                    omega_inv_notM - reuseable_part2,
                    omega_inv_notM + reuseable_part2,
                    -np.matmul(
                        np.matmul(np.diag(s_vec), X_M_gram_inv), s_vec / omega_inv_M
                    )
                    * lasso_lambda,
                ],
                axis=0,
            )

            numerator = b_vec - np.matmul(A_matrix, z)

            denominator = np.reshape(np.matmul(A_matrix, xi), numerator.shape[0])

            V_minus_bool = (b_vec - np.matmul(A_matrix, y) > 1e-16) & (denominator < 0)
            V_plus_bool = (b_vec - np.matmul(A_matrix, y) > 1e-16) & (denominator > 0)

            if (len(numerator[V_minus_bool]) == 0) | (
                len(denominator[V_minus_bool]) == 0
            ):
                V_minus = -np.inf
            else:
                V_minus = np.max(numerator[V_minus_bool] / denominator[V_minus_bool])

            if (len(numerator[V_plus_bool]) == 0) | (
                len(denominator[V_plus_bool]) == 0
            ):
                V_plus = np.inf
            else:
                V_plus = np.max(numerator[V_plus_bool] / denominator[V_plus_bool])

            # if True:
            #     print("V_minus is ", V_minus)
            #     print("V_plus is ", V_plus)

            a, b = V_minus / std_beta_bar, V_plus / std_beta_bar
            # if verbose:
            #     print("Truncation bounds: (%.2f, %.2f)" % (a, b))
            studentized_posi = beta_bar[i_covariate] / std_beta_bar

            studentized_posi_vec[i_covariate] = studentized_posi
            trunc_a_vec[i_covariate] = a
            trunc_b_vec[i_covariate] = b
            # p_raw_vec[i_covariate]=p_raw
            if beta_bar[i_covariate] > 0:
                right_tail = truncnorm.logsf(studentized_posi, a=a, b=b)
                left_tail = truncnorm.logcdf(-studentized_posi, a=a, b=b)
            else:
                right_tail = truncnorm.logsf(-studentized_posi, a=a, b=b)
                left_tail = truncnorm.logcdf(studentized_posi, a=a, b=b)

            if (np.isnan(right_tail)) | (np.isnan(left_tail)):
                p_raw = np.nan
                continue

            if (np.isinf(-right_tail)) & (np.isinf(-left_tail)):
                p_raw = -np.inf
            elif np.abs(right_tail - left_tail) > 16:
                p_raw = np.max([right_tail, left_tail])
            else:
                p_raw = np.log(np.exp(right_tail) + np.exp(left_tail))
            if p_raw < np.log(1e-16):
                p_raw = np.log(1e-16)
            p_raw_vec[i_covariate] = p_raw

        posi_log_pval_matrix[active_set, i_unit] = p_raw_vec

    return posi_log_pval_matrix


def get_log_p_no_cv(covariates, response, alphas):
    """
    covariates: design matrix X: T by d
    responses: outcome matrix y: T by N
    """
    d = covariates.shape[1]
    N = response.shape[1]
    T_obs_train_ = covariates.shape[0]
    posi_log_pval_matrix = np.nan * np.ones((d, N))
    # t_log_pval_matrix = np.nan * np.ones((d, N))
    # no prior
    omega_inv_vec = np.ones(d)
    for i_unit in range(N):
        picked_alpha = alphas[i_unit]

        lasso = Lasso(alpha=picked_alpha)
        lasso_fitted = lasso.fit(
            X=covariates,
            y=response[:, i_unit],
        )
        lasso_lambda = picked_alpha
        y = response[:, i_unit]
        active_set = lasso_fitted.coef_ != 0
        X_M = covariates[:, active_set]
        omega_inv_M = omega_inv_vec[active_set]
        omega_inv_notM = omega_inv_vec[~active_set]
        X_notM = covariates[:, ~active_set]
        X_M_card = X_M.shape[1]
        if X_M_card == 0:
            continue
        ols_post_lasso = OLS(endog=y, exog=X_M)
        ols_post_lasso_fitted = ols_post_lasso.fit()
        beta_bar = ols_post_lasso_fitted.params
        X_M_gram = np.matmul(X_M.transpose(), X_M)
        X_M_gram_inv = pinv(X_M_gram)
        X_M_pseudo_inv = np.matmul(X_M_gram_inv, X_M.transpose())
        this_df = max(1, T_obs_train_ - X_M_card)
        estimated_var = (
            np.sum(np.power(y - lasso_fitted.predict(covariates), 2)) / this_df
        )
        Sigma = np.eye(T_obs_train_) * estimated_var
        p_raw_vec = np.zeros(X_M_card)
        studentized_posi_vec, trunc_a_vec, trunc_b_vec = (
            np.zeros(X_M_card),
            np.zeros(X_M_card),
            np.zeros(X_M_card),
        )

        for i_covariate in range(X_M_card):
            eta = np.reshape(X_M_pseudo_inv[i_covariate, :], newshape=(T_obs_train_, 1))

            var_beta_bar = estimated_var * (X_M_gram_inv.diagonal())[i_covariate]
            std_beta_bar = np.sqrt(var_beta_bar)

            # print('Estimated sigma(beta)',std_beta_bar)
            xi = np.reshape(
                np.matmul(Sigma, eta) / var_beta_bar, newshape=(T_obs_train_, 1)
            )
            z = np.matmul(np.eye(T_obs_train_) - np.matmul(xi, eta.transpose()), y)
            s_vec = np.sign(lasso_fitted.coef_[active_set])
            P_M = np.matmul(X_M, X_M_pseudo_inv)
            # print("Dim of P_M is ", P_M.shape)
            reuseable_part1 = np.matmul(X_notM.transpose(), np.eye(T_obs_train_) - P_M)

            A_matrix = np.concatenate(
                [
                    reuseable_part1 / lasso_lambda,
                    -reuseable_part1 / lasso_lambda,
                    -np.matmul(np.diag(s_vec), X_M_pseudo_inv),
                ],
                axis=0,
            )

            reuseable_part2 = np.matmul(
                np.matmul(X_notM.transpose(), X_M_pseudo_inv.transpose()),
                s_vec / omega_inv_M,
            )

            b_vec = np.concatenate(
                [
                    omega_inv_notM - reuseable_part2,
                    omega_inv_notM + reuseable_part2,
                    -np.matmul(
                        np.matmul(np.diag(s_vec), X_M_gram_inv), s_vec / omega_inv_M
                    )
                    * lasso_lambda,
                ],
                axis=0,
            )

            numerator = b_vec - np.matmul(A_matrix, z)

            denominator = np.reshape(np.matmul(A_matrix, xi), numerator.shape[0])

            V_minus_bool = (b_vec - np.matmul(A_matrix, y) > 1e-16) & (denominator < 0)
            V_plus_bool = (b_vec - np.matmul(A_matrix, y) > 1e-16) & (denominator > 0)

            if (len(numerator[V_minus_bool]) == 0) | (
                len(denominator[V_minus_bool]) == 0
            ):
                V_minus = -np.inf
            else:
                V_minus = np.max(numerator[V_minus_bool] / denominator[V_minus_bool])

            if (len(numerator[V_plus_bool]) == 0) | (
                len(denominator[V_plus_bool]) == 0
            ):
                V_plus = np.inf
            else:
                V_plus = np.max(numerator[V_plus_bool] / denominator[V_plus_bool])

            # if True:
            #     print("V_minus is ", V_minus)
            #     print("V_plus is ", V_plus)

            a, b = V_minus / std_beta_bar, V_plus / std_beta_bar
            # if verbose:
            #     print("Truncation bounds: (%.2f, %.2f)" % (a, b))
            studentized_posi = beta_bar[i_covariate] / std_beta_bar

            studentized_posi_vec[i_covariate] = studentized_posi
            trunc_a_vec[i_covariate] = a
            trunc_b_vec[i_covariate] = b
            # p_raw_vec[i_covariate]=p_raw
            if beta_bar[i_covariate] > 0:
                right_tail = truncnorm.logsf(studentized_posi, a=a, b=b)
                left_tail = truncnorm.logcdf(-studentized_posi, a=a, b=b)
            else:
                right_tail = truncnorm.logsf(-studentized_posi, a=a, b=b)
                left_tail = truncnorm.logcdf(studentized_posi, a=a, b=b)

            if (np.isnan(right_tail)) | (np.isnan(left_tail)):
                p_raw = np.nan
                continue

            if (np.isinf(-right_tail)) & (np.isinf(-left_tail)):
                p_raw = -np.inf
            elif np.abs(right_tail - left_tail) > 16:
                p_raw = np.max([right_tail, left_tail])
            else:
                p_raw = np.log(np.exp(right_tail) + np.exp(left_tail))
            if p_raw < np.log(1e-16):
                p_raw = np.log(1e-16)
            p_raw_vec[i_covariate] = p_raw

        posi_log_pval_matrix[active_set, i_unit] = p_raw_vec

    return posi_log_pval_matrix
