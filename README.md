# Gomez

A pure Rust framework and implementation of (derivative-free) methods for
solving nonlinear (bound-constrained) systems of equations.

**Warning:** The code and API are still quite rough. Expect changes.

This library provides a variety of solvers of nonlinear equation systems with
*n* equations and *n* unknowns written entirely in Rust. Bound constraints for
variables are supported first-class, which is useful for engineering
applications. All solvers implement the same interface which is designed to give
full control over the process and allows to combine different components to
achieve the desired solution. The implemented methods are historically-proven
numerical methods or global optimization algorithms.

The convergence of the numerical methods is tested on several problems and the
implementation is benchmarked against with
[GSL](https://www.gnu.org/software/gsl/doc/html/multiroots.html) library.

## Algorithms

* [Trust region](trust_region::TrustRegion) -- Recommended method to be used as
  a default and it will just work in most of the cases.
* [Cuckoo search](cuckoo::Cuckoo) -- A global optimization algorithm useful for
  initial guesses search in combination with a numerical solver.
* [Nelder-Mead](nelder_mead::NelderMead) -- Not generally recommended, but may
  be useful for low-dimensionality problems with ill-defined Jacobian matrix.

## Roadmap

Listed *not* in order of priority.

* [Steffensen's method](https://en.wikipedia.org/wiki/Steffensen%27s_method) for
  1D systems.
* [Homotopy continuation
  method](http://homepages.math.uic.edu/~jan/srvart/node4.html) to compare the
  performance with Trust region method.
* Conjugate gradient method
* Experimentation with various global optimization techniques for initial
  guesses search
  * Evolutionary/nature-inspired algorithms
  * Bayesian optimization
* Focus on initial guesses search and tools in general
* High-level drivers encapsulating the low-level API for users that do not need
  the fine-grained control.

## License

Licensed under [MIT](LICENSE).

There are `gsl-wrapper` and `gsl-sys` crates which are licensed under the
[GPLv3](http://www.gnu.org/licenses/gpl-3.0.html) identically as
[GSL](https://www.gnu.org/software/gsl/) itself. This code is part of the
repository, but is not part of the Gomez library. Its purpose is solely for
comparison in Gomez benchmarks.
