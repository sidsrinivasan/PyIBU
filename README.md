# Iterative Bayesian Unfolding for Measurement Error Mitigation

This package implements iterative Bayesian unfolding for mitigation
of measurement errors from quantum computers as decribed in the paper
https://arxiv.org/pdf/2210.12284.pdf.

## IBU Quick Start

Given a dictionary mapping bitstrings to observed counts `counts`, as well as
response matrices `matrices` for each active qubit, you can use IBU
for measurement error mitigation as follows.

1. Construct the `params` dictionary with the following keys.
   1. `method`: This can be `full` or `reduced`
   2. `library`: This can be either `jax` or `tensorflow`. Note that only `full`
   method has tensorflow support
   3. `num_qubits`: The integer number of qubits (or length of bitstrings)
   4. `max_iters`: The integer maximum iterations of IBU before termination; 
   recommended number is 100
   5. `tol`: The floating-point tolerance for difference in magnitude of 
   parameter updates for early stopping; recommended number is 1e-4
   6. `use_log`: `True` or `False` (recommended); computations can be done in 
   log space for numerical stability, but this is not usually needed
   7. `verbose`: `True` (recommended) or `False`; whether to provide verbose 
   updates 
   8. `ham_dist`: __For `reduced` method ONLY__; specify the hamming distance 
   from the observed bitstrings that should be tracked. Instead of estimating an
   error mitigated distribution with support over all 2^N bitstrings for an 
   N-qubit system, the distribution is only supported on bitstrings `ham_dist`
   away from the measured bitstrings; recommended `ham_dist` is `1`.
2. Create ibu object as `ibu = IBU(matrices, params)` where matrices are a list 
of single-qubit response matrices __IN REVERSE ORDER__ of bitstrings.
3. Set the observed counts with `ibu.set_obs(counts)`.
4. Initialize a guess with `ibu.initialize_guess()`.
5. Call the `train(max_iters, tol=tol, soln=soln)` function by specifying
the maximum number of IBU iterations `max_iters`, the tolerance for difference
in magnitude of parameter updates `tol` for early stopping, and (optionally)
a solution `soln` for evaluation. The `soln` may either be a dictionary of
mapping bitstrings to true probabilities or a list of bitstrings to extract the
error-mitigated probabilities of.

That's it! The train function will return the solution vector, the number of 
iterations it actually ran for, and an array tracking the performance of the
parameter on the provided `soln`.

## Additional Tips and Tricks

- Run IBU on a GPU/TPU to take advantage of parallelism. JAX will automatically
run on the GPU if your machine is set up correctly.
- The Jupyter notebook `tutorial.ipynb` is a walkthrough of applying IBU
on measurements made on a GHZ state using simulated data from qiskit. The
tutorial also compares `IBU` to `M3` and shows that IBU does not produce any
negative probabilities and typically produces estimates with lower error.

- Initialization can matter for IBU, and the package supports 3 different ways 
of supplying an initial guess.
  - If `initialize_guess()` is called with no arguments, the initial guess is a
  uniform distribution over all bitstrings (`full`) or over all supported
  bitstrings (`reduced`) is initialized.
  - If `initialize_guess(list)` or `initialize_guess(tuple)` is called, the 
  initial guess is a uniform distribution over all bitstrings specified in the 
  list/tuple.
  - If `initialize_guess(dict)` is called with a dictionary mapping bitstrings 
  to probabilities, the initial guess is the distribution given in the 
  dictionary. 
backend `quimb`. This method is __deprecated__ and is not guaranteed to work.
  



