Manipulations and integration functions:
 - Caching of equilibrium spectra
 - Caching of Jackknife matrix
 - Indexing of spectra, so T[0] gives you T[:, 0], T[1] gives T[:, 1], etc
 - Folding A/a
 - Arithmetic on T, so we can do sum(T) instead of sum(T.data)

Plotting and examples:
 - Plotting of T[:, j], for j deletions
 - Comparison across different deletion frequencies for segregating SNPs

