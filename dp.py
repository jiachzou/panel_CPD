# %%
import numpy as np
import matplotlib.pyplot as plt

# Generate data
# np.random.seed(0)
# A = np.array([1, 4, 2, 6])
# varw = 0.5
# sig = np.sqrt(varw)
# x = np.concatenate([sig*np.random.randn(20) + A[0],
#                     sig*np.random.randn(30) + A[1],
#                     sig*np.random.randn(15) + A[2],
#                     sig*np.random.randn(35) + A[3]])
# N = len(x)
# #%%
# # %%
# import numpy as np


# Kmax= 5
# Tau = [10,18,19,20,21,
#        43,44,48,49,50,
#        51,60,61,62,63,64]
# ratio = 0.7
# Tau = np.array(list(range(100)))


def rDP(x, Tau, ratio, Kmax):
    """
    x is either T-dim array or T x N matrix
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

    # return Tau


# %%


# actual_jumps = rDP(x, Tau, ratio, Kmax)

# plt.plot(x)
# for i in range(len(actual_jumps)):
#     plt.axvline(actual_jumps[i])

# # %%
# actual_jumps
# # %%
