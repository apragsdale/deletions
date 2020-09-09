# Utility functions for deletions.


def choose(n, i):
    return np.exp(gammaln(n + 1) - gammaln(n - i + 1) - gammaln(i + 1))


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
