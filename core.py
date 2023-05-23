from sklearn.linear_model import LassoCV
from scipy.sparse import vstack
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from selectinf.algorithms.api import lasso

from collections import Counter
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import warnings
import os
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
import pickle

from sklearn.linear_model import LinearRegression

rpy2.robjects.numpy2ri.activate()
hdbinseg = rpackages.importr("hdbinseg")

def rDP(x, Tau, ratio, Kmax):
    """
    Perform rDP algorithm on input data.

    Parameters:
    -----------
    x : array_like
        Input data. Can be either a 1D array of length T or a 2D matrix of shape T x N.
    Tau : list
        Potential changepoint locations. It is the output from the lasso algorithm.
    ratio : float
        Ratio threshold used to determine the stopping condition for the algorithm.
    Kmax : int
        Maximum number of selected points.

    Returns:
    --------
    list
        List of indices representing the detected changepoints.


    """
    Tau = list(Tau) + [len(x)]
    N_choice = len(Tau)
    mem_arr = np.zeros((Kmax + 1, N_choice))
    jumps = np.zeros((Kmax + 1, N_choice))

    J = None

    for k in range(Kmax + 1):
        if k == 0:
            for i in range(N_choice):
                mem_arr[k, i] = np.sum(
                    np.power(x[: Tau[i]] - np.mean(x[: Tau[i]], axis=0), 2)
                )
        else:
            for i in range(k + 1, N_choice):
                comps = []
                for prev_i in range(k, i):
                    comps.append(
                        mem_arr[k - 1, prev_i]
                        + np.sum(
                            np.power(
                                x[Tau[prev_i] : Tau[i]]
                                - np.mean(x[Tau[prev_i] : Tau[i]], axis=0),
                                2,
                            )
                        )
                    )
                mem_arr[k, i] = min(comps)
                jumps[k, i] = np.argmin(comps) + k

            if mem_arr[k, -1] / J > ratio or k == Kmax:
                jumps_idces = [jumps[k, -1]]
                for i in range(k - 1):
                    jumps_idces.append(jumps[k - i - 1, int(jumps_idces[-1])])

                actual_jumps = [Tau[int(i)] for i in jumps_idces]

                return actual_jumps

        J = mem_arr[k, -1]

def panel_posi_unordered(log_pval_matrix, gamma):
    """
    Perform panel post inference selection.

    Parameters:
    -----------
    log_pval_matrix : array_like
        Matrix of log p-values. Should be a 2D array of shape (N, T), where N is the number of tests and T is the number of observations.
    gamma : float
        Threshold parameter for selection.

    Returns:
    --------
    tuple
        Tuple containing three elements:
        - selection_result: Array of indices representing the results from the panel posi algorithm.
        - rho: Parameter used for threshold calculation.
        - bonf_selection_result: Array of indices representing the selected tests using Bonferroni correction.

    """
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


def cho_chpt(Y):
    """
    Perform changepoint detection using the SBS and DCBS algorithms.

    Parameters:
    -----------
    Y : array_like
        Input data. Should be a 2D array of shape (N, T), where N is the number of variables and T is the number of time points.

    Returns:
    --------
    tuple
        Tuple containing two arrays:
        - chpt_sbs: Array of indices representing the changepoints detected by the SBS algorithm.
        - dcbs_alg: Array of indices representing the changepoints detected by the DCBS algorithm.
    """
    
    Y_r = Y
    nr, nc = Y_r.T.shape
    Yr = ro.r.matrix(Y_r.T, nrow=nr, ncol=nc)
    result_sbs = hdbinseg.sbs_alg(Yr, cp_type=1, temporal=True)
    result_dcbs = hdbinseg.dcbs_alg(
        Yr,
        cp_type=1,
    )
    chpt_sbs = np.array(result_sbs[2]).astype(int)
    dcbs_alg = np.array(result_dcbs[2]).astype(int)
    return chpt_sbs, dcbs_alg


def regression_mse_mae(Y, beta_idces):
    T, N = Y.shape
    X = np.tril(np.ones((T, T)), k=0)[:, beta_idces]
    betas = []
    for i in range(N):
        betas.append(np.linalg.solve(X.T @ X, X.T @ Y[:, i]))
    Y_hat = X @ np.array(betas).T
    assert Y.shape == Y_hat.shape
    mse = np.mean((Y - Y_hat) ** 2)
    mae = np.mean(np.abs(Y - Y_hat))
    return mse, mae


def generate_data(
    T,
    N,
    n_jumps,
    level_bounds,
    min_gaps=0,
    partial_effect_ratio=1,
    show_plot=True,
    heavy_tail=False,
    poission_corruption=False,
    staircase=False,
    AR=False,
    RegX=False,
):
    """
    Generate synthetic data with jumps and noise.

    Args:
        T : int
            Length of each time series.
        N : int
            Number of time series.
        n_jumps : int
            Number of desired jumps.
        level_bounds : float
            Bounds on the jump level magnitude.
        min_gaps : int, optional
            Minimum time between two jumps. Defaults to 0.
        partial_effect_ratio : float, optional
            Percentage of series affected by each individual jump. Defaults to 1.
        show_plot : bool, optional
            Whether to show plots of the generated data. Defaults to True.
        heavy_tail : bool, optional
            Whether to use a heavy-tailed distribution for noises. Defaults to False.
        poission_corruption : bool, optional
            Whether to introduce Poisson-corrupted noises. Defaults to False.
        staircase : bool, optional
            Whether to create staircase effects in jumps. Defaults to False.
        AR : bool, optional
            Whether to add autoregressive (AR) effects to the generated data. Defaults to False.
        RegX : bool, optional
            Whether to introduce external regressors (RegX) to the generated data. Defaults to False.

    Returns:
        tuple
            Tuple containing the following elements:
            - full_beta: Matrix of jump magnitudes. Shape: (T, N).
            - noises: Matrix of noises. Shape: (T, N).
            - jump_idces: Array of indices representing the jump positions.
            - Y: Matrix of generated data. Shape: (T, N).
            - X: Matrix representing the structure of jumps.

    """
    assert min_gaps * n_jumps < T

    X = np.tril(np.ones((T, T)), k=0)
    noises = np.random.multivariate_normal(mean=np.zeros(N), cov=np.eye(N), size=T)

    # heavy tail -- t distribution for noises
    if heavy_tail:
        noises = np.random.standard_t(3, size=(T, N))
    full_beta = np.zeros((T, N))
    # ensure minimum gap
    jump_idces_no_gap = np.sort(
        np.random.choice(np.arange(T - n_jumps * min_gaps), size=n_jumps, replace=False)
    )
    jump_idces = jump_idces_no_gap + np.arange(n_jumps) * min_gaps

    levels = np.random.uniform(low=-level_bounds, high=level_bounds, size=(n_jumps, N))
    jumps = levels - np.concatenate([np.zeros((1, N)), levels[:-1, :]])

    # poisson
    if poission_corruption:
        noises += np.random.poisson(lam=0.2, size=(T, N)) * 5 * level_bounds

    # partial effect
    if partial_effect_ratio < 1:
        for i in range(n_jumps - 1):
            unaffected_idces = np.random.choice(
                np.arange(N), size=int(N * (1 - partial_effect_ratio)), replace=False
            ).astype(int)
            # absorbed to next jump to maintain constant variance
            if i < n_jumps - 1:
                jumps[i + 1, unaffected_idces] += jumps[i, unaffected_idces]
            jumps[i, unaffected_idces] = 0

    if staircase:
        nth_jumps = np.random.permutation(n_jumps)
        cur_unaffected_idces = np.arange(N)
        # the last one affects all of the series
        for i, idx in enumerate(nth_jumps[:-1]):
            cur_unaffected_idces = np.random.choice(
                cur_unaffected_idces,
                size=int(N * (1 - (i + 1) / n_jumps)),
                replace=False,
            ).astype(int)
            if idx < n_jumps - 1:
                jumps[idx + 1, cur_unaffected_idces] += jumps[idx, cur_unaffected_idces]
            jumps[idx, cur_unaffected_idces] = 0

    full_beta[jump_idces] = jumps

    Y = X @ full_beta + noises
    plot_indices = None
    if show_plot:
        # sample some series to visualize
        plot_indices = np.random.choice(np.arange(N), size=20, replace=False)
        plt.plot(Y[:, plot_indices])
        for i in range(len(jump_idces)):
            plt.axvline(jump_idces[i], linestyle="--")
        plt.title("Sampled 20 series for visualization")
        plt.show()

    if AR:
        for idx_series in range(N):
            Y_AR = np.zeros(T)
            Y_AR[0] = Y[0, idx_series]
            coef = np.random.uniform(0.5, 0.8)
            for idx_time in range(1, T):
                Y_AR[idx_time] = Y_AR[idx_time - 1] * coef + Y[idx_time, idx_series]

            # generation complete, now regress out AR coef
            reg = LinearRegression(fit_intercept=False).fit(
                X=Y_AR[:-1].reshape(-1, 1), y=Y_AR[1:]
            )

            fitted_ar_coef = reg.coef_
            res = Y_AR[1:] - fitted_ar_coef * Y_AR[:-1]
            Y[1:, idx_series] = res

        if show_plot:
            plt.plot(Y[:, plot_indices])
            for i in range(len(jump_idces)):
                plt.axvline(jump_idces[i], linestyle="--")
            plt.title("Sampled 20 series residuals for visualization")
            plt.show()

    if RegX:
        for idx_series in range(N):
            X_exo = np.random.normal(size=(T, 5))

            coef = np.random.uniform(0.5, 0.8, size=(5,))
            Y_reg = X_exo @ coef + Y[:, idx_series]

            # generation complete, now regress out AR coef
            reg = LinearRegression(fit_intercept=False).fit(X=X_exo, y=Y_reg)

            fitted_ar_coef = reg.coef_
            res = Y_reg - X_exo @ fitted_ar_coef
            Y[:, idx_series] = res

        if show_plot:
            plt.plot(Y[:, plot_indices])
            for i in range(len(jump_idces)):
                plt.axvline(jump_idces[i], linestyle="--")
            plt.title("Sampled 20 series residuals for visualization")
            plt.show()

    return full_beta, noises, jump_idces, Y, X


def fuzzy_join(A, N):
    """
    Perform fuzzy join operation on multiple lists.

    Parameters:
    -----------
    A : list
        List of input lists.
    N : int
        Number of lists to be considered for the join.

    Returns:
    --------
    list
        List of selected elements after the fuzzy join operation.

    """
    maxlen = max(len(r) for r in A)
    selected = []
    for i in range(maxlen):
        comp = []
        for j in range(N):
            if i < len(A[j]):
                comp.append(A[j][i])

        ct = Counter(comp)

        selected.append(ct.most_common(1)[0][0])
    return list(set(selected))


def cv_lasso(y, X):
    """
    Perform cross-validated Lasso regression and return the optimal alpha value.

    Args:
        y : array_like
            Single column of Y.
        X : array_like
            Whole X matrix.

    Returns:
        float
            Optimal alpha value determined by cross-validated Lasso regression.

    """

    kf = KFold(n_splits=5, shuffle=True)
    lcv = LassoCV(fit_intercept=False, cv=kf, n_jobs=-1)
    fitted = lcv.fit(X, y)

    return fitted.alpha_


def experiment(beta_mat, epsilon_mat, real_jump_idces, dp_param_J, Y=None, X=None):
    """
    Perform one set of experiment for N series of length T and return confusion matrices.

    Args:
        beta_mat : np.ndarray
            T x N matrix representing the true jump magnitudes.
        epsilon_mat : np.ndarray
            T x N matrix representing the noise.
        real_jump_idces : np.ndarray
            Array containing the indices of the real jump locations.
        dp_param_J : float
            Parameter for the rDP algorithm.
        Y : np.ndarray, optional
            Matrix of generated data. Defaults to None.
        X : np.ndarray, optional
            Matrix representing the structure of jumps. Defaults to None.

    Returns:
        pd.DataFrame
            DataFrame containing the confusion matrix results for different methods.
        list
            List containing stored selections, Y matrix, and real jump indices.

    """

    assert beta_mat.shape == epsilon_mat.shape

    T, N = beta_mat.shape
    num_train = int(0.8 * N)
    shuffled_index = np.arange(N).astype(int)
    # inplace
    np.random.shuffle(shuffled_index)
    train_index = shuffled_index[:num_train]
    test_index = shuffled_index[num_train:]
    assert max(real_jump_idces) < T
    assert dp_param_J < 1
    assert len(beta_mat.shape) == 2

    # generate data based on beta and epsilon, only in the None AR case
    if X is None:
        X = np.tril(np.ones((T, T)), k=0)
    if Y is None:
        Y = X @ beta_mat + epsilon_mat
    # SNR = np.mean(beta_mat**2)/2 * T
    SNR = np.sqrt(np.mean(beta_mat**2))
    # fit lasso column-wise and generate p_values
    rdp_selections = []
    pvals_list = []
    alphas = []
    for col in train_index:
        lam = cv_lasso(Y[:, col], X)
        alphas.append(lam)
        # adjust the lambdas between sklearn and selectinf
        lam_adjusted = lam * T

        L = lasso.gaussian(X, Y[:, col], lam_adjusted)
        L.fit()
        pval = np.array([np.nan] * T)
        significant_index = []
        if len(L.summary()) > 0:
            significant_index = L.summary()["variable"].values
            pval[significant_index] = L.summary()["pvalue"].values

        pvals_list.append(pval)
        if len(significant_index) > 0:
            rst = rDP(
                Y[:, col], significant_index, dp_param_J, Kmax=len(significant_index)
            )
            rdp_selections.append(rst)
        else:
            rdp_selections.append([])
        # print(rst, significant_index)

    pvals = np.array(pvals_list).T

    pvals[(pvals < 1e-32)] = 1e-32

    P_value_log = np.log(pvals)

    rst_data = []
    stored_selections = {}
    for posi_gamma in [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05]:
        # posi
        posi_selection, rho, bonf_selection_result = panel_posi_unordered(
            P_value_log, posi_gamma
        )

        # rdp with union and join

        rdp_union = np.array(list(set.union(*(set(s) for s in rdp_selections))))
        rdp_intersection = np.array(
            list(set.intersection(*(set(s) for s in rdp_selections)))
        )

        # rdp with fuzzy join -- majority voting
        rdp_fuzzy_selection = fuzzy_join(rdp_selections, num_train)

        panel_rdp_selection = rDP(
            Y[:, train_index], rdp_union, dp_param_J, Kmax=len(rdp_union)
        )
        sbs_selection, dcbs_selection = cho_chpt(Y[:, train_index])

        all_selections = [
            posi_selection,
            rdp_union,
            rdp_intersection,
            rdp_fuzzy_selection,
            panel_rdp_selection,
            bonf_selection_result,
            sbs_selection,
            dcbs_selection,
        ]
        stored_selections[posi_gamma] = all_selections
        for name, selection in zip(
            [
                "posi",
                "rdp_union",
                "rdp_intersection",
                "rdp_majority_voting",
                "panel_rdp",
                "bonf_selection",
                "sbs",
                "dcbs",
            ],
            all_selections,
        ):
            jump_ind_real = np.zeros(T)
            jump_ind_real[real_jump_idces] = 1
            jump_ind_pred = np.zeros(T)
            selection = np.array(selection, dtype="int")
            jump_ind_pred[selection] = 1
            tn, fp, fn, tp = confusion_matrix(jump_ind_real, jump_ind_pred).ravel()

            # mse mae
            ins_mse, ins_mae = regression_mse_mae(Y[:, train_index], selection)
            oos_mse, oos_mae = regression_mse_mae(Y[:, test_index], selection)
            rst_row = [
                name,
                tp / (tp + fp + 0.0),
                tp / (tp + fn + 0.0),
                (tp + tn) / (tp + tn + fp + fn + 0.0),
                fn / (tp + fn + 0.0),
                fp / (tp + fp + 0.0),
                tp + fp,
                fp,
                tp,
                fn,
            ]
            if name == "posi":
                rst_row.append(rho)

            else:
                rst_row.append(np.nan)
            f1 = 2 * tp / (2 * tp + fp + fn)
            rst_row.extend([ins_mse, ins_mae, oos_mse, oos_mae, f1, SNR, posi_gamma])
            rst_data.append(rst_row)

        colnames = [
            "method",
            "precision",
            "recall",
            "accuracy",
            "Type II error",
            "Type I error",
            "# selected",
            "# false selection",
            "# correct selection",
            "# ommited",
            "rho",
            "ins_mse",
            "ins_mae",
            "oos_mse",
            "oos_mae",
            "f1",
            "SNR",
            "posi_gamma",
        ]

    return pd.DataFrame(data=rst_data, columns=colnames).set_index("method"), [
        stored_selections,
        Y,
        real_jump_idces,
    ]


def parallel_wrapper(
    T,
    N,
    n_jumps,
    level_bounds,
    min_gaps,
    partial_effect_ratio,
    heavy_tail=False,
    poission_corruption=False,
    J=0.8,
    staircase=False,
    RegX=False,
):
    """
    Perform one set of experiment for N series of length T and return confusion matrices.

    Args:
        beta_mat (np.ndarray): T x N matrix representing the true jump magnitudes.
        epsilon_mat (np.ndarray): T x N matrix representing the noise.
        real_jump_idces (np.ndarray): Array containing the indices of the real jump locations.
        dp_param_J (float): Parameter for the rDP algorithm.
        Y (np.ndarray, optional): Matrix of generated data. Defaults to None.
        X (np.ndarray, optional): Matrix representing the structure of jumps. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the confusion matrix results for different methods.
        list: List containing stored selections, Y matrix, and real jump indices.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        beta, noises, jump_idces, Y, X = generate_data(
            T,
            N,
            n_jumps,
            level_bounds,
            min_gaps,
            partial_effect_ratio,
            show_plot=False,
            heavy_tail=heavy_tail,
            poission_corruption=poission_corruption,
            staircase=staircase,
            RegX=RegX,
        )
        rst, state = experiment(beta, noises, jump_idces, J, Y, X)
    return rst, state


def parallel_experiments(
    n_runs,
    n_jobs,
    T=100,
    N=10,
    n_jumps=1,
    level_bounds=10,
    min_gaps=0,
    partial_effect_ratio=1,
    heavy_tail=False,
    poission_corruption=False,
    J=0.8,
    staircase=False,
    RegX=False,
):
    
    """
    Perform parallel experiments for a given number of runs and return summary statistics.

    Args:
        n_runs (int): Number of experiment runs to perform.
        n_jobs (int): Number of parallel jobs to run.
        T (int, optional): Length of the series. Defaults to 100.
        N (int, optional): Number of series. Defaults to 10.
        n_jumps (int, optional): Number of jumps. Defaults to 1.
        level_bounds (int, optional): Level bounds for jumps. Defaults to 10.
        min_gaps (int, optional): Minimum gap between jumps. Defaults to 0.
        partial_effect_ratio (float, optional): Partial effect ratio. Defaults to 1.
        heavy_tail (bool, optional): Flag indicating heavy-tailed noise. Defaults to False.
        poission_corruption (bool, optional): Flag indicating Poisson corruption. Defaults to False.
        J (float, optional): Parameter for the rDP algorithm. Defaults to 0.8.
        staircase (bool, optional): Flag indicating staircase jumps. Defaults to False.
        RegX (bool, optional): Flag indicating RegX structure. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the summary statistics of the confusion matrix results.
    """
    
    identifier = f"nruns={n_runs}_T={T}_N={N}_n_jumps={n_jumps}_level_bounds={level_bounds}_min_gaps={min_gaps}_partial_effect_ratio={partial_effect_ratio}_heavy_tail={heavy_tail}_poission_corruption={poission_corruption}_J={J}_staircase={staircase}_RegX={RegX}"
    fname = f"{identifier}.csv"

    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("states"):
        os.mkdir("states")
    done_files = os.listdir("results")
    if fname in done_files:
        return
    tups = Parallel(n_jobs=n_jobs)(
        delayed(parallel_wrapper)(
            T,
            N,
            n_jumps,
            level_bounds,
            min_gaps,
            partial_effect_ratio,
            heavy_tail,
            poission_corruption,
            J,
            staircase,
            RegX,
        )
        for i in tqdm(range(n_runs))
    )
    dfs = [t[0] for t in tups]
    data = np.array([tup[0].values for tup in tups])
    return_data = [
        data.mean(axis=0),
        data.std(axis=0),
        data.max(axis=0),
        data.min(axis=0),
        np.percentile(data, q=2.5, axis=0),
        np.percentile(data, q=97.5, axis=0),
    ]
    return_dfs = [
        pd.DataFrame(data=ret_dat, columns=dfs[0].columns, index=dfs[0].index)
        for ret_dat in return_data
    ]
    rst_df = pd.concat(
        return_dfs,
        axis=0,
        keys=["mean", "std", "max", "min", "lower_conf", "upper_conf"],
    )

    rst_df.to_csv(f"results/{fname}")
    states = [tup[1] for tup in tups]

    with open(f"states/{identifier}.pickle", "wb") as handle:
        pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return rst_df
