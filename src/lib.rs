#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![warn(missing_docs)]

//! _gomez_ is a framework and implementation for **mathematical optimization**
//! and solving **non-linear systems of equations**.
//!
//! The library is written completely in Rust. Its focus is on being useful for
//! **practical problems** and having API that is simple for easy cases as well
//! as flexible for complicated ones. The name stands for ***g***lobal
//! ***o***ptimization & ***n***on-linear ***e***quations ***s***olving, with a
//! few typos.
//!
//! ## Practical problems
//!
//! The main goal is to be useful for practical problems. This is manifested by
//! the following features:
//!
//! * _Derivative-free_. No algorithm requires an analytical derivative
//!   (gradient, Hessian, Jacobian). Methods that use derivatives approximate it
//!   using finite difference method<sup>1</sup>.
//! * _Constraints_ support. It is possible to specify the problem domain with
//!   constraints<sup>2</sup>, which is necessary for many engineering
//!   applications.
//! * Non-naive implementations. The code is not a direct translation of a
//!   textbook pseudocode. It's written with performance in mind and applies
//!   important techniques from numerical mathematics. It also tries to handle
//!   situations that hurt the methods but occur in practice.
//!
//! <sup>1</sup> There is a plan to provide ways to override this approximation
//! with a real derivative.
//!
//! <sup>2</sup> Currently, only unconstrained and box-bounded domains are
//! supported.
//!
//! ## Algorithms
//!
//! * [Trust region](algo::trust_region) – Recommended method to be used as a
//!   default and it will just work in most cases.
//! * [LIPO](algo::lipo) – Global optimization algorithm useful for searching
//!   good initial guesses in combination with a numerical algorithm.
//! * [Steffensen](algo::steffensen) – Fast and lightweight method for solving
//!   one-dimensional systems of equations.
//! * [Nelder-Mead](algo::nelder_mead) – Direct search optimization method that
//!   does not use any derivatives.
//!
//! This list will be extended in the future. But at the same time, having as
//! many algorithms as possible is _not_ the goal. Instead, the focus is on
//! providing quality implementations of battle-tested methods.
//!
//! ## Mathematical optimization
//!
//! Given a function _f: D → R_ from some domain _D_ (in _R<sup>n</sup>_) to the
//! real numbers and an initial point _x<sub>0</sub>_, the goal is to find a
//! point _x'_ such that _f(x')_ is a minimum. Note that gomez does not
//! guarantee that the minimum is global, although the focus is on global
//! optimization techniques.
//!
//! ### Example
//!
//! ```rust
//! use gomez::nalgebra as na;
//! use gomez::{Domain, Function, OptimizerDriver, Problem};
//! use na::{Dyn, IsContiguous};
//!
//! // Objective function is represented by a struct.
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! impl Problem for Rosenbrock {
//!     // Field type, f32 or f64.
//!     type Field = f64;
//!
//!     // Domain of the function.
//!     fn domain(&self) -> Domain<Self::Field> {
//!         Domain::unconstrained(2)
//!     }
//! }
//!
//! impl Function for Rosenbrock {
//!     // Body of the function, taking x and returning f(x).
//!     fn apply<Sx>(&self, x: &na::Vector<Self::Field, Dyn, Sx>) -> Self::Field
//!     where
//!         Sx: na::Storage<Self::Field, Dyn> + IsContiguous,
//!     {
//!         (self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2)
//!     }
//! }
//!
//! let f = Rosenbrock { a: 1.0, b: 1.0 };
//! let mut optimizer = OptimizerDriver::builder(&f)
//!     .with_initial(vec![-10.0, -5.0])
//!     .build();
//!
//! let (x, fx) = optimizer
//!     .find(|state| state.fx() <= 1e-6 || state.iter() >= 100)
//!     .expect("optimizer error");
//!
//! println!("f(x) = {fx}\tx = {x:?}");
//! # assert!(fx <= 1e-6);
//! ```
//!
//! See [`OptimizerDriver`] and [`OptimizerBuilder`](driver::OptimizerBuilder)
//! for additional options.
//!
//! ## Systems of equations
//!
//! Given a vector function _r: D → R<sup>n</sup>_, with _r<sub>i</sub>: D → R_
//! from some domain _D_ (in _R<sup>n</sup>_) to the real numbers for _i = 1, 2,
//! ..., n_, and an initial point _x<sub>0</sub>_, the goal is to find a point
//! _x'_ such that _r(x) = 0_. Note that there is no constraint on the form of
//! the equations _r<sub>i</sub>_ (compare with specialized solvers for [systems
//! of linear
//! equations](https://en.wikipedia.org/wiki/System_of_linear_equations)).
//!
//! ### Example
//!
//! ```rust
//! use gomez::nalgebra as na;
//! use gomez::{Domain, Problem, SolverDriver, System};
//! use na::{Dyn, IsContiguous};
//!
//! // System of equations is represented by a struct.
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! impl Problem for Rosenbrock {
//!     // Field type, f32 or f64.
//!     type Field = f64;
//!
//!     // Domain of the system.
//!     fn domain(&self) -> Domain<Self::Field> {
//!         Domain::unconstrained(2)
//!     }
//! }
//!
//! impl System for Rosenbrock {
//!     // Evaluation of the system (computing the residuals).
//!     fn eval<Sx, Srx>(
//!         &self,
//!         x: &na::Vector<Self::Field, Dyn, Sx>,
//!         rx: &mut na::Vector<Self::Field, Dyn, Srx>,
//!     ) where
//!         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
//!         Srx: na::storage::StorageMut<Self::Field, Dyn>,
//!     {
//!         rx[0] = (self.a - x[0]).powi(2);
//!         rx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
//!     }
//! }
//!
//! let r = Rosenbrock { a: 1.0, b: 1.0 };
//! let mut solver = SolverDriver::builder(&r)
//!     .with_initial(vec![-10.0, -5.0])
//!     .build();
//!
//! let (x, norm) = solver
//!     .find(|state| state.norm() <= 1e-6 || state.iter() >= 100)
//!     .expect("solver error");
//!
//! println!("|| r(x) || = {norm}\tx = {x:?}");
//! # assert!(norm <= 1e-6);
//! ```
//!
//! See [`SolverDriver`] and [`SolverBuilder`](driver::SolverBuilder) for
//! additional options.
//!
//! ## Custom algorithms
//!
//! It is possible to create a custom algorithm by implementing the
//! [`Optimizer`] and/or [`Solver`] trait. Then it can be used by the driver as
//! any other algorithm provided by gomez. Go see the documentation of the
//! traits to get more details.
//!
//! ```rust
//! # use gomez::nalgebra as na;
//! # use gomez::{Domain, Function, Optimizer, OptimizerDriver, Problem};
//! # use na::{storage::StorageMut, Dyn, IsContiguous, Vector};
//! #
//! # struct MyAlgo;
//! #
//! # impl MyAlgo {
//! #     fn new(_: &Rosenbrock, _: &Domain<f64>) -> Self {
//! #         Self
//! #     }
//! # }
//! #
//! # impl Optimizer<Rosenbrock> for MyAlgo {
//! #     const NAME: &'static str = "my algo";
//! #     type Error = std::convert::Infallible;
//! #
//! #     fn opt_next<Sx>(
//! #         &mut self,
//! #         f: &Rosenbrock,
//! #         dom: &Domain<f64>,
//! #         x: &mut Vector<f64, Dyn, Sx>,
//! #     ) -> Result<f64, Self::Error>
//! #     where
//! #         Sx: StorageMut<f64, Dyn> + IsContiguous,
//! #     {
//! #         Ok(0.0)
//! #     }
//! # }
//! #
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # impl Problem for Rosenbrock {
//! #     type Field = f64;
//! #
//! #     fn domain(&self) -> Domain<Self::Field> {
//! #         Domain::rect(vec![-10.0, -10.0], vec![10.0, 10.0])
//! #     }
//! # }
//! #
//! # impl Function for Rosenbrock {
//! #     fn apply<Sx>(&self, x: &na::Vector<Self::Field, Dyn, Sx>) -> Self::Field
//! #     where
//! #         Sx: na::Storage<Self::Field, Dyn> + IsContiguous,
//! #     {
//! #         (self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2)
//! #     }
//! # }
//! #
//! let f = Rosenbrock { a: 1.0, b: 1.0 };
//! let mut optimizer = OptimizerDriver::builder(&f)
//!     .with_algo(|f, dom| MyAlgo::new(f, dom))
//!     .build();
//! ```
//!
//! If you implement an algorithm, please reach out to discuss if we could
//! include it in gomez.
//!
//! ## Roadmap
//!
//! Listed *not* in priority order.
//!
//! * [Homotopy continuation
//!   method](http://homepages.math.uic.edu/~jan/srvart/node4.html) to compare
//!   the performance with the trust region method
//! * Conjugate gradient method
//! * Experimentation with various global optimization techniques for initial
//!   guess search
//!   * Evolutionary/nature-inspired algorithms
//!   * Bayesian optimization
//! * Focus on initial guesses search and tools for analysis in general
//!
//! ## License
//!
//! Licensed under MIT.

pub mod algo;
pub mod analysis;
mod core;
pub mod derivatives;
pub mod driver;

pub use core::*;
pub use driver::{OptimizerDriver, SolverDriver};

#[cfg(feature = "testing")]
pub mod testing;

#[cfg(not(feature = "testing"))]
pub(crate) mod testing;

pub use nalgebra;
