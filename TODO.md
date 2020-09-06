Manipulations and integration functions:
 - Initialization of empty spectrum with expected SFS of deletions with given
   selection coefficient (integrate exact formulae)
 - Caching of equilibrium spectra
 - Profiling - should other items be cached? jackknife, e.g.?
 - Projection of full spectrum
 - Projection of 
 - Indexing of spectra, so T[0] gives you T[:, 0], T[1] gives T[:, 1], etc
 - Folding A/a
 - Loci fixed for A get mapped to ancestral, as an option, so that SFS is not
   symmetric

Plotting and examples:
 - Plotting of T[:, j], for j deletions
 - Comparison across different deletion frequencies for segregating SNPs

