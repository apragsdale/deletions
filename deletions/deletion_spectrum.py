# Methods to compute the deletion+biallelic frequency spectrum, which is cast
# as a triallelic frequency spectrum, with the two derived states being a
# deletion and a new mutation. The structure is quite similar to the standard
# triallelic spectrum, but we handle the edges of the spectrum and the mutation
# model differently.

import numpy as np
import attr

from scipy.sparse.linalg import factorized
from scipy.sparse import identity
from scipy.special import gammaln
from scipy.sparse import csc_matrix, csr_matrix

import copy


def positive(self, attribute, value):
    if value <= 0:
        raise ValueError(f"{attribute.name} must be greater than zero")


def optional(func):
    """
    Wraps one or more validator functions with an "if not None" clause.
    """
    if isinstance(func, (tuple, list)):
        func_list = func
    else:
        func_list = [func]

    def validator(self, attribute, value):
        if value is not None:
            for func in func_list:
                func(self, attribute, value)

    return validator


@attr.s(repr=False)
class DelSpectrum:
    """
    Representation and manipulation of a "deletion frequency spectrum" array.

    data: The flattened triallelic spectrum as a numpy array.
          T_n(i,j) = {Phi_{n-j}(i)}, where i is the frequency of the
          derived allele, and j is the frequency of the deletion. Then
          T has size (1 + 2 + ... + n-1 + n + n+1).
    mask: An optional array of the same size as data, containing True
          and False.
    n: The sample size of T.
    """

    data: np.array = attr.ib(default=None)
    mask: np.array(bool) = attr.ib(default=None)
    n: int = attr.ib(default=None, validator=optional([positive]))
    folded: bool = attr.ib(default=False)

    def __attrs_post_init__(self):
        def _get_n_from_length(l):
            return (np.sqrt(1 + 8 * l) - 1) / 2 - 1

        if self.data is None:
            if self.n is None:
                raise ValueError("if data is None, must provide n")
            else:
                self.data = np.zeros((self.n + 1) * (self.n + 2) // 2)
        # ensure data length and n match, and assign n if not given
        n_from_data = _get_n_from_length(len(self.data))
        if np.isclose(n_from_data, np.rint(n_from_data)) is False:
            raise ValueError("length of data is not valid")
        if self.n is None:
            self.n = np.rint(n_from_data).astype(int)
        else:
            if np.rint(n_from_data).astype(int) != self.n:
                raise ValueError(f"data does not have valid length for n = {self.n}")
        # set up the mask if not provided
        if self.mask is None:
            self.mask = np.full(len(self.data), False, dtype=bool)
            # self.mask[0] = True
            # self.mask[self.n] = True
            # self.mask[-1] = True
        else:
            if len(self.mask) != len(self.data):
                raise ValueError("data and mask must be same length")

    def _as_masked_array(self):
        """
        Represent the deletion frequency spectrum as a masked numpy array.
        """
        data_square = np.zeros((self.n + 1, self.n + 1))
        mask_square = np.full((self.n + 1, self.n + 1), True, dtype=bool)
        c = 0
        for i in range(self.n + 1):
            data_square[i, : self.n + 1 - i] = self.data[c : c + self.n + 1 - i]
            mask_square[i, : self.n + 1 - i] *= self.mask[c : c + self.n + 1 - i]
            c += self.n + 1 - i
        T = np.ma.masked_array(data_square, mask_square, fill_value=0)
        return T

    def __repr__(self):
        T = self._as_masked_array()
        return "DelSpectrum(\n%s,\nsample_size=%s\nfolded=%s)" % (
            str(T),
            str(self.n),
            str(False),
        )

    def from_file(self):
        """
        Read the frequency spectrum from a file.

        The file format is:
        First line: n folded
        Next n+1 lines: Phi_j, for j in 0 to n
        Next n+1 lines: mask_j, for j in 0 to n
        """
        pass

    def to_file(self):
        """
        Write the frequency spectrum to file
        """
        pass

    def integrate(
        self,
        nu,
        t,
        theta_snp=0.001,
        theta_del=0.001,
        s_del=None,
        h_del=0.5,
        dt=0.001,
        mutation_model="ISM",
    ):
        """
        Integrate the deletion frequency spectrum forward in time.

        nu: The relative size of the population over this epoch. Can be
            constant scalar > 0, or a function defined over [0, t].
        t: The integration time for this epoch.
        theta_snp: The scaled mutation rate from a (ancetral) -> A (derived. Note
                   that theta_snp should be 4*Ne*u_snp, and not scaled by L, so
                   this is the \emph{per-base} theta, so theta_snp << 1.
        theta_del: The scaled mutation rate of deletions, also not scaled by L,
                   so that theta_del << 1.
        sel_params: To-do! Could think of many interesting selection scenarios...
                    think how to make it flexible but tractable/intuitive.
        """
        if self.folded is True:
            raise ValueError("cannot integrate folded spectrum")

        self.data = integrate_crank_nicolson(
            self,
            nu,
            t,
            dt=dt,
            theta_snp=theta_snp,
            theta_del=theta_del,
            s_del=s_del,
            h_del=h_del,
            mutation_model=mutation_model,
        )

        if mutation_model == "ISM":
            # mask corners
            self.mask[get_idx(self.n, 0, 0)] = True
            self.mask[get_idx(self.n, 0, self.n)] = True
            self.mask[get_idx(self.n, self.n, 0)] = True

    def project(self, n_proj):
        if n_proj == self.n:
            return self
        elif n_proj > self.n:
            raise ValueError("projection size must be smaller than n")

        # to do

    def marginalize(self, axis):
        """
        Marginalize SNPs (axis = 0), to get the allele frequency distribution
        of deletion variants, or deletions (axis=1) to get the AFS of derived
        (A) SNPs.

        Note that marginalizing over deletions treats deletions and ancestral
        states as equivalent, and only counts A variants... Careful!
        """
        F = np.zeros(self.n + 1)
        mask = np.full(self.n + 1, False, dtype=bool)
        mask[0] = mask[-1] = True
        F += np.sum(self._as_masked_array().data.T, axis=axis)
        return np.ma.masked_array(F, mask, fill_value=0)


####
# Integration function
####


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
            print("initializing with density in fixed states")
            X.data[get_idx(X.n, 0, 0)] = 1.0 / 4
            X.data[get_idx(X.n, X.n, 0)] = 1.0 / 4
            X.data[get_idx(X.n, 0, X.n)] = 1.0 / 2
            X.integrate(
                1,
                4000,
                mutation_model=mutation_model,
                dt=1,
                theta_del=theta_del,
                theta_snp=theta_snp,
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
            assert np.isclose(data.sum(), 1)

        N0_prev = N0
        N1_prev = N1
        dt_prev = dt

        # check here for negative or nan values, for adaptive time stepping

        t_elapsed += dt

    return data


####
# Integration transition matrices
####


def drift_matrix(n):
    D = np.zeros(((n + 1) * (n + 2) // 2,) * 2)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            this_idx = get_idx(n, i, j)
            D[this_idx, this_idx] -= 2 * ((n - i - j) * i + (n - i - j) * j + i * j)
            if i < n and i + j + 1 <= n:
                D[get_idx(n, i + 1, j), this_idx] += (n - i - j) * i
            if i > 0:
                D[get_idx(n, i - 1, j), this_idx] += (n - i - j) * i
            if j < n and i + j + 1 <= n:
                D[get_idx(n, i, j + 1), this_idx] += (n - i - j) * j
            if j > 0:
                D[get_idx(n, i, j - 1), this_idx] += (n - i - j) * j
            if i < n and j > 0:
                D[get_idx(n, i + 1, j - 1), this_idx] += i * j
            if i > 0 and j < n:
                D[get_idx(n, i - 1, j + 1), this_idx] += i * j
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
        U[get_idx(n, 1, j), get_idx(n, 0, j)] += (n - j) * theta_snp / 2
        U[get_idx(n, 0, j), get_idx(n, 0, j)] -= (n - j) * theta_snp / 2
    # mutations A/a->del on Phi(i, 0), with i copies of A
    for i in range(1, n):
        # del hits a (retain i copies of A)
        U[get_idx(n, i, 1), get_idx(n, i, 0)] += (n - i) * theta_del / 2
        U[get_idx(n, i, 0), get_idx(n, i, 0)] -= (n - i) * theta_del / 2
    for i in range(n - 1):
        # del hits A
        U[get_idx(n, i, 1), get_idx(n, i + 1, 0)] += (i + 1) * theta_del / 2
        U[get_idx(n, i + 1, 0), get_idx(n, i + 1, 0)] -= (i + 1) * theta_del / 2
    return csc_matrix(U), U_null


def _get_null_factor(x, n):
    snp_target_sum = np.sum([x[get_idx(n, i, 0)] for i in range(1, n + 1)])
    del_target_sum = np.sum([x[get_idx(n, 0, j)] for j in range(1, n + 1)])
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
            U[get_idx(n, i, j), get_idx(n, i, j)] -= theta_snp[0] / 2 * (n - i - j)
            if n - i - j > 0:
                U[get_idx(n, i + 1, j), get_idx(n, i, j)] += (
                    theta_snp[0] / 2 * (n - i - j)
                )
            # mutation from A -> a, takes (i, j) -> (i - 1, j)
            U[get_idx(n, i, j), get_idx(n, i, j)] -= theta_snp[1] / 2 * i
            if i > 0:
                U[get_idx(n, i - 1, j), get_idx(n, i, j)] += theta_snp[1] / 2 * i
            # deletion mutation, takes (i, j) to (i, j + 1) and (i - 1, j + 1)
            U[get_idx(n, i, j), get_idx(n, i, j)] -= theta_del[0] / 2 * i  # hits A
            U[get_idx(n, i, j), get_idx(n, i, j)] -= (
                theta_del[0] / 2 * (n - i - j)  # hits a
            )
            if i > 0:
                U[get_idx(n, i - 1, j + 1), get_idx(n, i, j)] += theta_del[0] / 2 * i
            if (n - i - j) > 0:
                U[get_idx(n, i, j + 1), get_idx(n, i, j)] += (
                    theta_del[0] / 2 * (n - i - j)
                )
            # insertion mutation, takes (i, j) to (i, j - 1) and (i + 1, j - 1)
            # insertions of derived and ancestral states are equally likely
            if j > 0:
                U[get_idx(n, i, j), get_idx(n, i, j)] -= theta_del[1] / 2 * j
                U[get_idx(n, i, j - 1), get_idx(n, i, j)] += theta_del[1] / 2 * j / 2
                U[get_idx(n, i + 1, j - 1), get_idx(n, i, j)] += (
                    theta_del[1] / 2 * j / 2
                )
    return csc_matrix(U)


def choose(n, i):
    return np.exp(gammaln(n + 1) - gammaln(n - i + 1) - gammaln(i + 1))


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
            this_idx = get_idx(n, i, j)
            #S[this_idx, get_idx(n + 2, i + 1, j + 1)] -= (
            #    -2 * s_del * (i + 1) * (j + 1) * j
            #)
            #if i < n and j > 0:
            #    S[get_idx(n, i + 1, j - 1), get_idx(n + 2, i + 1, j + 1)] += (
            #        -2 * s_del * (i + 1) * (j + 1) * j
            #    )

            # incoming density
            S[this_idx, get_idx(n + 2, i, j + 2)] += (
                -2 * s_del * i * (j + 2) * (j + 1)
            )
            S[this_idx, get_idx(n + 2, i, j + 2)] += (
                -2 * s_del * (j + 2) * (j + 1) * (n - i - j)
            )
            S[this_idx, get_idx(n + 2, i + 1, j + 1)] += (
                -2 * s_del * h_del * (i + 1) * i * (j + 1)
            )
            S[this_idx, get_idx(n + 2, i + 1, j + 1)] += (
                -2 * s_del * h_del * (i + 1) * (j + 1) * (n - i - j)
            )
            S[this_idx, get_idx(n + 2, i, j + 1)] += (
                -2 * s_del * h_del * i * (j + 1) * (n - i - j + 1)
            )
            S[this_idx, get_idx(n + 2, i, j + 1)] += (
                -2 * s_del * h_del * (j + 1) * (n - i - j + 1) * (n - i - j)
            )
            S[this_idx, get_idx(n + 2, i + 1, j + 1)] += (
                -2 * s_del * h_del * (i + 1) * (j + 1) * j
            )
            S[this_idx, get_idx(n + 2, i + 1, j + 1)] += (
                -2 * s_del * h_del * (i + 1) * (j + 1) * (n - i - j)
            )
            S[this_idx, get_idx(n + 2, i, j + 1)] += (
                -2 * s_del * h_del * i * (j + 1) * (n - i - j + 1)
            )
            S[this_idx, get_idx(n + 2, i, j + 1)] += (
                -2 * s_del * h_del * (j + 1) * j * (n - i - j + 1)
            )
            # outgoing density
            S[this_idx, get_idx(n + 2, i + 1, j + 1)] -= (
                -2 * s_del * (j + 1) * j * (i + 1)
            )
            S[this_idx, get_idx(n + 2, i, j + 1)] -= (
                -2 * s_del * (j + 1) * j * (n - i - j + 1)
            )
            S[this_idx, get_idx(n + 2, i + 2, j)] -= (
                -2 * s_del * h_del * (i + 2) * (i + 1) * j
            )
            S[this_idx, get_idx(n + 2, i + 1, j)] -= (
                -2 * s_del * h_del * (i + 1) * j * (n - i - j + 1)
            )
            S[this_idx, get_idx(n + 2, i + 1, j)] -= (
                -2 * s_del * h_del * (i + 1) * j * (n - i - j + 1)
            )
            S[this_idx, get_idx(n + 2, i, j)] -= (
                -2 * s_del * h_del * j * (n - i - j + 2) * (n - i - j + 1)
            )
            S[this_idx, get_idx(n + 2, i, j + 2)] -= (
                -2 * s_del * h_del * i * (j + 2) * (j + 1)
            )
            S[this_idx, get_idx(n + 2, i, j + 1)] -= (
                -2 * s_del * h_del * i * (j + 1) * (n - i - j + 1)
            )
            S[this_idx, get_idx(n + 2, i + 1, j + 1)] -= (
                -2 * s_del * h_del * (i + 1) * (j + 1) * (n - i - j)
            )
            S[this_idx, get_idx(n + 2, i, j + 2)] -= (
                -2 * s_del * h_del * (j + 2) * (j + 1) * (n - i - j)
            )

    S *= 1 / (n + 2) / (n + 1)
    return csc_matrix(S)


####
# Utility functions
####

_idx_cache = {}


def get_idx(n, i, j):
    try:
        return _idx_cache[n][(i, j)]
    except KeyError:
        _idx_cache.setdefault(n, {})
        _idx_cache[n] = cache_idx(n)
        return _idx_cache[n][(i, j)]


def cache_idx(n):
    indexes = {}
    c = 0
    for j in range(n + 1):
        for i in range(n + 1 - j):
            indexes[(i, j)] = c
            c += 1
    return indexes


####
# Jackknife functions, taken from moments.Triallele
####


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
    # size of J is size of n+1 spectrum x size of n spectrum
    # J = np.zeros(((n+3)*(n+4)/2,(n+1)*(n+2)/2))
    row = []
    col = []
    data = []

    for i in range(1, n + 2):
        for j in range(1, n + 2 - i):
            ordered_set = closest_ij_2(i, j, n)
            alphas = compute_alphas_2(i, j, ordered_set, n)
            index2 = get_idx(n + 2, i, j)
            for pair, alpha in zip(ordered_set, alphas):
                index = get_idx(n, pair[0], pair[1])
                # J[index2,index] = alpha
                row.append(index2)
                col.append(index)
                data.append(alpha)

    # jackknife for the biallelic edges (i=0, j=1:n, and j=0, i=1:n)
    # first for j = 0
    j = 0
    for i in range(1, n + 2):
        this_ind = get_idx(n + 2, i, j)
        ii = find_iprime_1D(n, i)
        alphas = get_alphas_1D(ii, i, n)
        # J[this_ind, get_idx(n,ii-1,j)] = alphas[0]
        # J[this_ind, get_idx(n,ii,j)] = alphas[1]
        # J[this_ind, get_idx(n,ii+1,j)] = alphas[2]
        row.append(this_ind)
        col.append(get_idx(n, ii - 1, 0))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(get_idx(n, ii, 0))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(get_idx(n, ii + 1, 0))
        data.append(alphas[2])

    i = 0
    for j in range(1, n + 2):
        this_ind = get_idx(n + 2, i, j)
        jj = find_iprime_1D(n, j)
        alphas = get_alphas_1D(jj, j, n)
        # J[this_ind, get_idx(n,i,jj-1)] = alphas[0]
        # J[this_ind, get_idx(n,i,jj)] = alphas[1]
        # J[this_ind, get_idx(n,i,jj+1)] = alphas[2]
        row.append(this_ind)
        col.append(get_idx(n, 0, jj - 1))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(get_idx(n, 0, jj))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(get_idx(n, 0, jj + 1))
        data.append(alphas[2])

    # jackknife along diagonal - 1D jk
    for i in range(1, n + 2):
        j = n + 2 - i
        this_ind = get_idx(n + 2, i, j)
        ii = find_iprime_1D(n, i)
        alphas = get_alphas_1D(ii, i, n)
        row.append(this_ind)
        col.append(get_idx(n, ii - 1, n - ii + 1))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(get_idx(n, ii, n - ii))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(get_idx(n, ii + 1, n - ii - 1))
        data.append(alphas[2])

    return csr_matrix(
        (data, (row, col)),
        shape=(int((n + 3) * (n + 4) / 2), int((n + 1) * (n + 2) / 2)),
    )
