# gomez

[![Build](https://img.shields.io/github/actions/workflow/status/datamole-ai/gomez/ci.yml?branch=main)](https://github.com/datamole-ai/gomez/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/datamole-ai/gomez/blob/main/LICENSE)
[![Cargo](https://img.shields.io/crates/v/gomez.svg)](https://crates.io/crates/gomez)
[![Documentation](https://docs.rs/gomez/badge.svg)](https://docs.rs/gomez)

_gomez_ is a framework and implementation for **mathematical optimization** and
solving **non-linear systems of equations**.

The library is written completely in Rust. Its focus is on being useful for
**practical problems** and having API that is simple for easy cases as well as
flexible for complicated ones. The name stands for ***g***lobal
***o***ptimization & ***n***on-linear ***e***quations ***s***olving, with a few
typos.

## Practical problems

The main goal is to be useful for practical problems. This is manifested by the
following features:

* _Derivative-free_. No algorithm requires an analytical derivative (gradient,
  Hessian, Jacobian). Methods that use derivatives approximate it using finite
  difference method<sup>1</sup>.
* _Constraints_ support. It is possible to specify the problem domain with
  constraints<sup>2</sup>, which is necessary for many engineering applications.
* Non-naive implementations. The code is not a direct translation of a textbook
  pseudocode. It's written with performance in mind and applies important
  techniques from numerical mathematics. It also tries to handle situations that
  hurt the methods but occur in practice.

<sup>1</sup> There is a plan to provide ways to override this approximation with
a real derivative.

<sup>2</sup> Currently, only unconstrained and box-bounded domains are
supported.

## Algorithms

* [Trust region](algo::trust_region) – Recommended method to be used as a
  default and it will just work in most cases.
* [LIPO](algo::lipo) – Global optimization algorithm useful for searching good
  initial guesses in combination with a numerical algorithm.
* [Steffensen](algo::steffensen) – Fast and lightweight method for solving
  one-dimensional systems of equations.
* [Nelder-Mead](algo::nelder_mead) – Direct search optimization method that does
  not use any derivatives.

This list will be extended in the future. But at the same time, having as many
algorithms as possible is _not_ the goal. Instead, the focus is on providing
quality implementations of battle-tested methods.

## Roadmap

Listed *not* in priority order.

* [Homotopy continuation
  method](http://homepages.math.uic.edu/~jan/srvart/node4.html) to compare the
  performance with the trust region method
* Conjugate gradient method
* Experimentation with various global optimization techniques for initial guess
  search
  * Evolutionary/nature-inspired algorithms
  * Bayesian optimization
* Focus on initial guesses search and tools for analysis in general

## License

Licensed under [MIT](LICENSE).

There are `gsl-wrapper` and `gsl-sys` crates which are licensed under the
[GPLv3](http://www.gnu.org/licenses/gpl-3.0.html) identically as
[GSL](https://www.gnu.org/software/gsl/) itself. This code is part of the
repository but is not part of the gomez library. Its purpose is solely for
comparison in the benchmarks.
