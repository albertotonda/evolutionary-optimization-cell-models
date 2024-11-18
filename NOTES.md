# Notes

## Open questions after experiments for EvoSTAR
1. Weird result on the Pareto optimization for the E.coli core model. For some reason, it looks extremely difficult to find solutions with $f_2 < 5$. Does this depend on the hyperparameters of NSGA-II, or is it something different?
2. Could it make sense to compare against the results in https://doi.org/10.1101/2023.12.11.570118? How?
3. We could try to run some 'decomposition' multi-objective approach. Code for CMA-ES is already in the repository. Ask Georgios Katsirelos for his quadratic optimization approach.