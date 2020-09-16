# Requires matplotlib
# Pass sample size, required. Optional arguments are the projection
# size, piecewise constant history, selection and dominiance coefficients
# of structural variants.


import matplotlib.pylab as plt
import deletions
import argparse
import sys


def make_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--sample_size", "-n", type=int, default=None, help="Haploid sample size"
    )
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--projection_size",
        "-p",
        type=int,
        default=None,
        help="Projection size for plotting",
    )
    optional.add_argument(
        "--population_sizes",
        "-N",
        type=float,
        nargs="+",
        default=[],
        help="Series of piecewise constant relative population sizes, from past to present",
    )
    optional.add_argument(
        "--epoch_times",
        "-T",
        type=float,
        nargs="+",
        default=[],
        help="Series of piecewise epoch times, corresponding to the population sizes, in units of 2N generations",
    )
    optional.add_argument(
        "--selection_coeff",
        "-s",
        type=float,
        default=0.0,
        help="Selection coefficient against deletions",
    )
    optional.add_argument(
        "--dominance_coeff",
        type=float,
        default=0.5,
        help="Dominance coefficient of selected structural variants",
    )
    plotting = parser.add_argument_group("Optional plotting arguments")
    plotting.add_argument(
        "--log",
        type=bool,
        default=False,
    )
    plotting.add_argument(
        "--outfile",
        "-o",
        type=str,
        default=None,
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    assert len(args.population_sizes) == len(
        args.epoch_times
    ), "nus and Ts must be same length"
    nus = args.population_sizes
    tts = args.epoch_times

    s = args.selection_coeff
    h = args.dominance_coeff

    n = args.sample_size
    n_proj = args.projection_size
    if n_proj is None:
        n_proj = n
    elif n_proj > n:
        raise ValueError("n_proj must be less than n")

    # initialize the frequency spectrum
    T = deletions.DelSpectrum(n=n)
    T.integrate(1, 0, mutation_model="reversible", polarized=True, s_del=s, h_del=h)
    # integrate each epoch forward in time
    for nu, tt in zip(nus, tts):
        T.integrate(
            nu, tt, mutation_model="reversible", polarized=True, s_del=s, h_del=h
        )

    # plot marginal spectra projected to size given
    P = T.project_snps(n_proj)

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    for i in range(n - n_proj + 1):
        ax.plot(P[i][1:-1] / P[i][1:-1].sum(), 'o--', ms=2, lw=0.5, alpha=0.7, label=i)

    if args.log:
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Allele count')
    ax.legend(frameon=False, title="num. deleletions\nwith $n={0}$".format(n))

    fig.tight_layout()
    if args.outfile is not None:
        plt.savefig(args.outfile)
    plt.show()
