# Functions for initializing and integrating the deletion frequency
# spectrum forward in time.
#


from . import util

from scipy.sparse.linalg import factorized
from scipy.sparse import identity
from scipy.special import gammaln
from scipy.sparse import csc_matrix, csr_matrix
import scipy.special as scisp
from mpmath import hyp1f1

import copy
import pickle
import numpy as np
import os


def integrate_crank_nicolson(
    X,
    nu,
    t,
    dt=0.001,
    theta_snp=0.001,
    theta_del=0.001,
    s_del=None,
    h_del=0.5,
    mutation_model="ISM",
    polarized="False",
):
    """
    Integrate the frequency spectrum using the Crank-Nicolson scheme:
    mutation_model: If 'ISM', theta_snp and theta_del are scalars. If 'recurrent'
        or 'reversible', then theta_snp and theta_del are lists of length 2. If
        they are passed as scalars, the thetas are converted to a symmetric
        migration rate model with the given scalar values.
    """
    if mutation_model == "ISM":
        if not (isinstance(theta_snp, float) and isinstance(theta_del, float)):
            raise ValueError("thetas must be scalar floats if mutation model is ISM")
    elif mutation_model == "recurrent" or mutation_model == "reversible":
        if not isinstance(theta_snp, list):
            if isinstance(theta_snp, float):
                theta_snp = [theta_snp, theta_snp]
            else:
                raise ValueError("theta_snp must be float or list of length 2")
        if not isinstance(theta_del, list):
            if isinstance(theta_del, float):
                theta_del = [theta_del, theta_del]
            else:
                raise ValueError("theta_del must be float or list of length 2")
        if X.data.sum() == 0:
            print("initializing spectrum")
            X.data = initialize_del_spectrum(
                X.n, s_del, h_del, mutation_model, theta_del
            )
            X.integrate(
                1,
                40,
                dt=0.01,
                mutation_model=mutation_model,
                theta_del=theta_del,
                theta_snp=theta_snp,
                s_del=s_del,
                h_del=h_del,
            )
    else:
        raise ValueError("mutation model must be ISM or recurrent or reversible")

    if t < 0:
        raise ValueError("integration time must be positive")
    elif t == 0:
        return X.data
    else:
        data = copy.copy(X.data)

    if not callable(nu):
        if nu <= 0:
            raise ValueError("population size must be positive")
        N0 = N1 = nu

    N0_prev = 0
    N1_prev = 0

    D = drift_matrix(X.n)
    if mutation_model == "ISM":
        U, U_null = mutation_matrix_ism(X.n, theta_snp, theta_del)
    else:
        U = mutation_matrix_recurrent(X.n, theta_snp, theta_del)

    if s_del is not None:
        J = calcJK_2(X.n)
        S = selection_matrix(X.n, s_del, h_del)

    t_elapsed = 0
    while t_elapsed < t:
        # at some point, might want to implement adaptive time steps.
        # for now, we don't

        if t_elapsed + dt > t:
            dt = t - t_elapsed

        if callable(nu):
            N0 = nu(t_elapsed)
            N1 = nu(t_elapsed + dt)

        if t_elapsed == 0 or N0_prev != N0 or N1_prev != N1 or dt != dt_prev:
            Ab0 = U + D / (2 * N0)
            Ab1 = U + D / (2 * N1)
            if s_del is not None:
                Ab0 += S.dot(J)
                Ab1 += S.dot(J)
            Ab_fwd = identity(Ab0.shape[0], format="csc") + dt / 2.0 * Ab0
            Ab_bwd = factorized(identity(Ab1.shape[0], format="csc") - dt / 2.0 * Ab1)

        # ensure that the total mutation rate stays constant at theta
        if mutation_model == "ISM":
            null_factor = _get_null_factor(data, X.n)
            data = Ab_bwd(Ab_fwd.dot(data) + dt * null_factor * U_null)
        else:
            data = Ab_bwd(Ab_fwd.dot(data))

        if polarized is True and mutation_model is not "ISM":
            for j in range(X.n):
                data[util.get_idx(X.n, 0, j)] += data[util.get_idx(X.n, X.n - j, j)]
                data[util.get_idx(X.n, X.n - j, j)] = 0

        N0_prev = N0
        N1_prev = N1
        dt_prev = dt

        # check here for negative or nan values, for adaptive time stepping

        t_elapsed += dt

    return data


####
# Equilibrium frequency spectrum for selected sites with h=0.5
####


def equilibrium_biallelic(n, gamma, h, mutation_model, theta):
    """
    n: sample size
    gamma: scaled selection coefficient (2Ns)
    mutation_model: can be ISM or recurrent (or reversible)
    theta: scaled mutation rate.
    """
    if mutation_model == "ISM":
        print("warning:  not implemented for ISM")
        return np.zeros(n + 1)
    elif mutation_model in ["recurrent", "reversible"]:
        if h != 0.5:
            print(
                "warning: with recurrent mutations, steady state only returned "
                " for h = 1/2."
            )
        if not isinstance(theta, list):
            theta_fd = theta_bd = theta
        else:
            theta_fd, theta_bd = theta
        fs = np.zeros(n + 1)
        if gamma is None or gamma == 0.0:
            for i in range(n + 1):
                fs[i] = (
                    scisp.gammaln(n + 1)
                    - scisp.gammaln(n - i + 1)
                    - scisp.gammaln(i + 1)
                    + scisp.gammaln(i + theta_fd)
                    + scisp.gammaln(n - i + theta_bd)
                )
            fs += (
                scisp.gammaln(theta_fd + theta_bd)
                - scisp.gammaln(theta_fd)
                - scisp.gammaln(theta_bd)
                - scisp.gammaln(n + theta_fd + theta_bd)
            )
            fs = np.exp(fs)
        else:
            ## unstable for large n
            for i in range(n + 1):
                fs[i] = np.exp(
                    scisp.gammaln(n + 1)
                    - scisp.gammaln(n - i + 1)
                    - scisp.gammaln(i + 1)
                    + scisp.gammaln(i + theta_fd)
                    + scisp.gammaln(n - i + theta_bd)
                    - scisp.gammaln(n + theta_fd + theta_bd)
                ) * hyp1f1(i + theta_fd, n + theta_fd + theta_bd, 2 * gamma)
        return fs / np.sum(fs)
    else:
        raise ValueError(
            f"{mutation_model} is not a valid mutation model, pick "
            "from either ISM or recurrent / reversible"
        )


def initialize_del_spectrum(n, gamma, h, mutation_model, theta):
    """
    Returns the data for the deletion spectrum, where we've initialized with the
    biallelic spectrum for deletions, and integrated to rough steady-state.
    n: sample size.
    gamma: 2Ns.
    h: dominance coefficient.
    mutation_model: from ISM, or recurrent / reversible.
    """
    fs_bi = equilibrium_biallelic(n, gamma, h, mutation_model, theta)
    data = np.zeros((n + 1) * (n + 2) // 2)
    for j in range(n + 1):
        for i in range(n + 1 - j):
            data[util.get_idx(n, i, j)] = fs_bi[j] / (n + 1 - j)
    return data


####
# Integration transition matrices
####


def drift_matrix(n):
    D = np.zeros(((n + 1) * (n + 2) // 2,) * 2)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            this_idx = util.get_idx(n, i, j)
            D[this_idx, this_idx] -= 2 * ((n - i - j) * i + (n - i - j) * j + i * j)
            if i < n and i + j + 1 <= n:
                D[util.get_idx(n, i + 1, j), this_idx] += (n - i - j) * i
            if i > 0:
                D[util.get_idx(n, i - 1, j), this_idx] += (n - i - j) * i
            if j < n and i + j + 1 <= n:
                D[util.get_idx(n, i, j + 1), this_idx] += (n - i - j) * j
            if j > 0:
                D[util.get_idx(n, i, j - 1), this_idx] += (n - i - j) * j
            if i < n and j > 0:
                D[util.get_idx(n, i + 1, j - 1), this_idx] += i * j
            if i > 0 and j < n:
                D[util.get_idx(n, i - 1, j + 1), this_idx] += i * j
    return csc_matrix(D)


def mutation_matrix_ism(n, theta_snp, theta_del):
    """
    Mutations handled in a pseudo-ISM framework.
    """
    # mutations from the void
    U_null = np.zeros((n + 1) * (n + 2) // 2)
    U_null[1] = n * theta_snp / 2
    U_null[n + 1] = n * theta_del / 2
    # mutations on the background of loci segregating for one of the other mut type
    U = np.zeros(((n + 1) * (n + 2) // 2,) * 2)
    # mutations a->A on Phi(0, j), with j deletions
    for j in range(1, n):
        U[util.get_idx(n, 1, j), util.get_idx(n, 0, j)] += (n - j) * theta_snp / 2
        U[util.get_idx(n, 0, j), util.get_idx(n, 0, j)] -= (n - j) * theta_snp / 2
    # mutations A/a->del on Phi(i, 0), with i copies of A
    for i in range(1, n):
        # del hits a (retain i copies of A)
        U[util.get_idx(n, i, 1), util.get_idx(n, i, 0)] += (n - i) * theta_del / 2
        U[util.get_idx(n, i, 0), util.get_idx(n, i, 0)] -= (n - i) * theta_del / 2
    for i in range(n - 1):
        # del hits A
        U[util.get_idx(n, i, 1), util.get_idx(n, i + 1, 0)] += (i + 1) * theta_del / 2
        U[util.get_idx(n, i + 1, 0), util.get_idx(n, i + 1, 0)] -= (
            (i + 1) * theta_del / 2
        )
    return csc_matrix(U), U_null


def _get_null_factor(x, n):
    snp_target_sum = np.sum([x[util.get_idx(n, i, 0)] for i in range(1, n + 1)])
    del_target_sum = np.sum([x[util.get_idx(n, 0, j)] for j in range(1, n + 1)])
    if snp_target_sum > 1:
        raise ValueError("theta_snp is too large")
    if del_target_sum > 1:
        raise ValueError("theta_del is too large")
    null_factor = np.zeros(len(x))
    null_factor[1] = 1 - del_target_sum
    null_factor[n + 1] = 1 - snp_target_sum
    return null_factor


def mutation_matrix_recurrent(n, theta_snp, theta_del):
    """
    n: The sample size.
    theta_snp: A list of length two, with theta_fwd and theta_bwd
        for a->A and A->a mutations.
    theta_del: A list of length two, with theta_fwd and theat_bwd
        for deletions and insertions.
    """
    U = np.zeros(((n + 1) * (n + 2) // 2,) * 2)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            # mutation from a -> A, takes (i, j) -> (i + 1, j)
            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= (
                theta_snp[0] / 2 * (n - i - j)
            )
            if n - i - j > 0:
                U[util.get_idx(n, i + 1, j), util.get_idx(n, i, j)] += (
                    theta_snp[0] / 2 * (n - i - j)
                )
            # mutation from A -> a, takes (i, j) -> (i - 1, j)
            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= theta_snp[1] / 2 * i
            if i > 0:
                U[util.get_idx(n, i - 1, j), util.get_idx(n, i, j)] += (
                    theta_snp[1] / 2 * i
                )
            # deletion mutation, takes (i, j) to (i, j + 1) and (i - 1, j + 1)
            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= (
                theta_del[0] / 2 * i
            )  # hits A
            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= (
                theta_del[0] / 2 * (n - i - j)  # hits a
            )
            if i > 0:
                U[util.get_idx(n, i - 1, j + 1), util.get_idx(n, i, j)] += (
                    theta_del[0] / 2 * i
                )
            if (n - i - j) > 0:
                U[util.get_idx(n, i, j + 1), util.get_idx(n, i, j)] += (
                    theta_del[0] / 2 * (n - i - j)
                )
            # insertion mutation, takes (i, j) to (i, j - 1) and (i + 1, j - 1)
            # insertions of derived and ancestral states are equally likely
            if j > 0:
                U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= theta_del[1] / 2 * j
                U[util.get_idx(n, i, j - 1), util.get_idx(n, i, j)] += (
                    theta_del[1] / 2 * j / 2
                )
                U[util.get_idx(n, i + 1, j - 1), util.get_idx(n, i, j)] += (
                    theta_del[1] / 2 * j / 2
                )
    return csc_matrix(U)


def selection_matrix(n, s_del, h_del):
    """
    To include selection with dominance, T_n depends on T_(n+2), which is
    estimated using the jackknife (calcJK_2). This means that the selection
    transition matrix has shape (size T_n) \times (size T_n+2).

    Selection operator goes like S.J.T
    """
    size_n = (n + 1) * (n + 2) // 2
    size_n_2 = (n + 3) * (n + 4) // 2
    S = np.zeros((size_n, size_n_2))
    for j in range(n + 1):
        for i in range(n + 1 - j):
            # if i + j == 0 or i == n or j == n:
            #     continue
            this_idx = util.get_idx(n, i, j)
            # S[this_idx, util.get_idx(n + 2, i + 1, j + 1)] -= (
            #    -2 * s_del * (i + 1) * (j + 1) * j
            # )
            # if i < n and j > 0:
            #    S[util.get_idx(n, i + 1, j - 1), util.get_idx(n + 2, i + 1, j + 1)] += (
            #        -2 * s_del * (i + 1) * (j + 1) * j
            #    )

            # incoming density
            S[this_idx, util.get_idx(n + 2, i, j + 2)] += (
                -2 * s_del * i * (j + 2) * (j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 2)] += (
                -2 * s_del * (j + 2) * (j + 1) * (n - i - j)
            )
            S[this_idx, util.get_idx(n + 2, i + 1, j + 1)] += (
                -2 * s_del * h_del * (i + 1) * i * (j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i + 1, j + 1)] += (
                -2 * s_del * h_del * (i + 1) * (j + 1) * (n - i - j)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 1)] += (
                -2 * s_del * h_del * i * (j + 1) * (n - i - j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 1)] += (
                -2 * s_del * h_del * (j + 1) * (n - i - j + 1) * (n - i - j)
            )
            S[this_idx, util.get_idx(n + 2, i + 1, j + 1)] += (
                -2 * s_del * h_del * (i + 1) * (j + 1) * j
            )
            S[this_idx, util.get_idx(n + 2, i + 1, j + 1)] += (
                -2 * s_del * h_del * (i + 1) * (j + 1) * (n - i - j)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 1)] += (
                -2 * s_del * h_del * i * (j + 1) * (n - i - j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 1)] += (
                -2 * s_del * h_del * (j + 1) * j * (n - i - j + 1)
            )
            # outgoing density
            S[this_idx, util.get_idx(n + 2, i + 1, j + 1)] -= (
                -2 * s_del * (j + 1) * j * (i + 1)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 1)] -= (
                -2 * s_del * (j + 1) * j * (n - i - j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i + 2, j)] -= (
                -2 * s_del * h_del * (i + 2) * (i + 1) * j
            )
            S[this_idx, util.get_idx(n + 2, i + 1, j)] -= (
                -2 * s_del * h_del * (i + 1) * j * (n - i - j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i + 1, j)] -= (
                -2 * s_del * h_del * (i + 1) * j * (n - i - j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i, j)] -= (
                -2 * s_del * h_del * j * (n - i - j + 2) * (n - i - j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 2)] -= (
                -2 * s_del * h_del * i * (j + 2) * (j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 1)] -= (
                -2 * s_del * h_del * i * (j + 1) * (n - i - j + 1)
            )
            S[this_idx, util.get_idx(n + 2, i + 1, j + 1)] -= (
                -2 * s_del * h_del * (i + 1) * (j + 1) * (n - i - j)
            )
            S[this_idx, util.get_idx(n + 2, i, j + 2)] -= (
                -2 * s_del * h_del * (j + 2) * (j + 1) * (n - i - j)
            )

    S *= 1 / (n + 2) / (n + 1)
    return csc_matrix(S)


####
# Jackknife functions, taken from moments.Triallele
####


# Cache jackknife matrices in ~/.moments/TwoLocus_cache by default
def set_cache_path(path="~/.deletions/jackknife_cache"):
    """
    Set directory in which jackknife matrices are cached, so they do not
    need to be recomputed each time.
    """
    global cache_path
    cache_path = os.path.expanduser(path)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)


cache_path = None
set_cache_path()


def closest_ij_2(i, j, n):
    # sort by closest to farthest
    # I think we will need to have a spread of three grid points in each direction - a rectangular box leads to an A matrix with rank < 6
    fi, fj = i / (n + 2.0), j / (n + 2.0)
    possible_ij = []
    for ii in range(1, n):
        for jj in range(1, n - ii):
            possible_ij.append((ii, jj))
    possible_ij = np.array(possible_ij)
    smallests = np.argpartition(
        np.sum((np.array([fi, fj]) - possible_ij / (1.0 * n)) ** 2, axis=1), 6
    )[:6]
    smallest_set = np.array([possible_ij[k] for k in smallests])
    distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj]) ** 2, axis=1)
    order = distances.argsort()
    ordered_set = np.array([smallest_set[ii] for ii in order])
    # ensure that we have an index range of three in each direction
    # if we don't, drop the last (farthest) point, and get next closest until we have three points in each direction
    i_range, j_range = (
        np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
        np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
    )
    next_index = 7
    while i_range < 2 or j_range < 2:
        smallests = np.argpartition(
            np.sum((np.array([fi, fj]) - possible_ij / (1.0 * n)) ** 2, axis=1),
            next_index,
        )[:next_index]
        smallest_set = np.array([possible_ij[k] for k in smallests])
        distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj]) ** 2, axis=1)
        order = distances.argsort()
        new_ordered_set = np.array([smallest_set[ii] for ii in order])
        ordered_set[-1] = new_ordered_set[-1]
        i_range, j_range = (
            np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
            np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
        )
        next_index += 1
    return ordered_set


def compute_alphas_2(i, j, ordered_set, n):
    A = np.zeros((6, 6))
    b = np.zeros(6)
    A[0] = 1
    A[1] = ordered_set[:, 0] + 1.0
    A[2] = ordered_set[:, 1] + 1.0
    A[3] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 0] + 2.0)
    A[4] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 1] + 1.0)
    A[5] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 1] + 2.0)
    b[0] = (n + 1.0) * (n + 2.0) / ((n + 3.0) * (n + 4.0))
    b[1] = (n + 1.0) * (n + 2.0) / ((n + 4.0) * (n + 5.0)) * (i + 1.0)
    b[2] = (n + 1.0) * (n + 2.0) / ((n + 4.0) * (n + 5.0)) * (j + 1.0)
    b[3] = (n + 1.0) * (n + 2.0) / ((n + 5.0) * (n + 6.0)) * (i + 1.0) * (i + 2.0)
    b[4] = (n + 1.0) * (n + 2.0) / ((n + 5.0) * (n + 6.0)) * (i + 1.0) * (j + 1.0)
    b[5] = (n + 1.0) * (n + 2.0) / ((n + 5.0) * (n + 6.0)) * (j + 1.0) * (j + 2.0)
    return np.dot(np.linalg.inv(A), b)


def find_iprime_1D(n, i):
    # get iprime/n closest to i/(n+2)
    iis = np.arange(n + 1)
    ii = np.argmin(abs(iis / (1.0 * n) - i / (n + 2.0)))
    if ii < 2:
        ii = 2
    if ii > n - 2:
        ii = n - 2
    return ii


def get_alphas_1D(ii, i, n):
    A = np.zeros((3, 3))
    A[0] = 1
    A[1] = ii + np.arange(3)
    A[2] = (ii + np.arange(3)) * (ii + np.arange(1, 4))
    b = np.array(
        [
            (n + 1.0) / (n + 3),
            (n + 1.0) * (n + 2) * (i + 1) / ((n + 3) * (n + 4)),
            (n + 1.0) * (n + 2) * (i + 1) * (i + 2) / ((n + 4) * (n + 5)),
        ]
    )
    return np.dot(np.linalg.inv(A), b)


# compute the quadratic two-dim Jackknife extrapolation for Phi_n to Phi_{n+2}
# i,j are the indices in the n+1 spectrum (just for interior points)
def calcJK_2(n):
    # check if cached, if so just load it
    jackknife_fname = f"jk_{n}_2.mtx"
    if os.path.isfile(os.path.join(cache_path, jackknife_fname)):
        with open(os.path.join(cache_path, jackknife_fname), "rb") as fin:
            try:
                J = pickle.load(fin)
            except:
                J = pickle.load(fin, encoding="Latin1")
        return J

    # size of J is size of n+1 spectrum x size of n spectrum
    # J = np.zeros(((n+3)*(n+4)/2,(n+1)*(n+2)/2))
    row = []
    col = []
    data = []

    for i in range(1, n + 2):
        for j in range(1, n + 2 - i):
            ordered_set = closest_ij_2(i, j, n)
            alphas = compute_alphas_2(i, j, ordered_set, n)
            index2 = util.get_idx(n + 2, i, j)
            for pair, alpha in zip(ordered_set, alphas):
                index = util.get_idx(n, pair[0], pair[1])
                # J[index2,index] = alpha
                row.append(index2)
                col.append(index)
                data.append(alpha)

    # jackknife for the biallelic edges (i=0, j=1:n, and j=0, i=1:n)
    # first for j = 0
    j = 0
    for i in range(1, n + 2):
        this_ind = util.get_idx(n + 2, i, j)
        ii = find_iprime_1D(n, i)
        alphas = get_alphas_1D(ii, i, n)
        # J[this_ind, util.get_idx(n,ii-1,j)] = alphas[0]
        # J[this_ind, util.get_idx(n,ii,j)] = alphas[1]
        # J[this_ind, util.get_idx(n,ii+1,j)] = alphas[2]
        row.append(this_ind)
        col.append(util.get_idx(n, ii - 1, 0))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(util.get_idx(n, ii, 0))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(util.get_idx(n, ii + 1, 0))
        data.append(alphas[2])

    i = 0
    for j in range(1, n + 2):
        this_ind = util.get_idx(n + 2, i, j)
        jj = find_iprime_1D(n, j)
        alphas = get_alphas_1D(jj, j, n)
        # J[this_ind, util.get_idx(n,i,jj-1)] = alphas[0]
        # J[this_ind, util.get_idx(n,i,jj)] = alphas[1]
        # J[this_ind, util.get_idx(n,i,jj+1)] = alphas[2]
        row.append(this_ind)
        col.append(util.get_idx(n, 0, jj - 1))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(util.get_idx(n, 0, jj))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(util.get_idx(n, 0, jj + 1))
        data.append(alphas[2])

    # jackknife along diagonal - 1D jk
    for i in range(1, n + 2):
        j = n + 2 - i
        this_ind = util.get_idx(n + 2, i, j)
        ii = find_iprime_1D(n, i)
        alphas = get_alphas_1D(ii, i, n)
        row.append(this_ind)
        col.append(util.get_idx(n, ii - 1, n - ii + 1))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(util.get_idx(n, ii, n - ii))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(util.get_idx(n, ii + 1, n - ii - 1))
        data.append(alphas[2])

    J = csr_matrix(
        (data, (row, col)),
        shape=(int((n + 3) * (n + 4) / 2), int((n + 1) * (n + 2) / 2)),
    )
    # cache J
    with open(os.path.join(cache_path, jackknife_fname), "wb+") as fout:
        pickle.dump(J, fout, pickle.HIGHEST_PROTOCOL)

    return J
