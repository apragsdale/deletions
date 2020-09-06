# deletions
Compute the SFS in the presense of structural variants.

# installation
Requires:
 - numpy
 - scipy
 - attrs

Plotting features (which are not implemented yet, will) require:
 - matplotlib
 - seaborn

# usage
```
>> import deletions

>> X = deletions.DelSpectrum(n=40)

>> X.integrate(nu, T, mutation_model="reversible", s_del=-1, h_del=0.1)
```

Here, `nu` is the relative size, `T` is the integration time, mutation model can be `"reversible"` or `"ISM"`,
and `s_del` and `h_del` are the selection and dominance coefficients for selection acting against deletions.
Additional options: `theta_snp` and `theta_del` control the SNP and structural variant mutation rates.
