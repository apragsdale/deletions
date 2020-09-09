# Methods to compute the deletion+biallelic frequency spectrum, which is cast
# as a triallelic frequency spectrum, with the two derived states being a
# deletion and a new mutation. The structure is quite similar to the standard
# triallelic spectrum, but we handle the edges of the spectrum and the mutation
# model differently.

import numpy as np
import attr
import copy

from . import util
from . import integration


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

        self.data = integration.integrate_crank_nicolson(
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
            self.mask[util.util(self.n, 0, 0)] = True
            self.mask[util.util(self.n, 0, self.n)] = True
            self.mask[util.util(self.n, self.n, 0)] = True

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
