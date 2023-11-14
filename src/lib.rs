#![allow(clippy::many_single_char_names)]
#![allow(clippy::type_complexity)]
#![warn(missing_docs)]

//! # Gomez
//!
//! A pure Rust framework and implementation of (derivative-free) methods for
//! solving nonlinear (bound-constrained) systems of equations.
//!
//! This library provides a variety of solvers of nonlinear equation systems
//! with *n* equations and *n* unknowns written entirely in Rust. Bound
//! constraints for variables are supported first-class, which is useful for
//! engineering applications. All solvers implement the same interface which is
//! designed to give full control over the process and allows to combine
//! different components to achieve the desired solution. The implemented
//! methods are historically-proven numerical methods or global optimization
//! algorithms.
//!
//! The convergence of the numerical methods is tested on several problems and
//! the implementation is benchmarked against with
//! [GSL](https://www.gnu.org/software/gsl/doc/html/multiroots.html) library.
//!
//! ## Algorithms
//!
//! * [Trust region](algo::trust_region) -- Recommended method to be used as a
//!   default and it will just work in most of the cases.
//! * [LIPO](algo::lipo) -- A global optimization algorithm useful for initial
//!   guesses search in combination with a numerical solver.
//! * [Steffensen](algo::steffensen) -- Fast and lightweight method for
//!   one-dimensional systems.
//! * [Nelder-Mead](algo::nelder_mead) -- Not generally recommended, but may be
//!   useful for low-dimensionality problems with ill-defined Jacobian matrix.
//!
//! ## Problem
//!
//! The problem of solving systems of nonlinear equations (multidimensional root
//! finding) is about finding values of *n* variables given *n* equations that
//! have to be satisfied. In our case, we consider a general algebraic form of
//! the equations (compare with specialized solvers for [systems of linear
//! equations](https://en.wikipedia.org/wiki/System_of_linear_equations)).
//!
//! Mathematically, the problem is formulated as
//!
//! ```text
//! F(x) = 0,
//!
//! where F(x) = { f1(x), ..., fn(x) }
//! and x = { x1, ..., xn }
//! ```
//!
//! Moreover, it is possible to add bound constraints to the variables. That is:
//!
//! ```text
//! Li <= xi <= Ui for some bounds [L, U] for every i
//! ```
//!
//! The bounds can be negative/positive infinity, effectively making the
//! variable unconstrained.
//!
//! More sophisticated constraints (such as (in)equalities consisting of
//! multiple variables) are currently out of the scope of this library. If you
//! are in need of those, feel free to contribute with the API design
//! incorporating them and the implementation of appropriate solvers.
//!
//! When it comes to code, the problem is any type that implements the
//! [`System`] and [`Problem`] traits.
//!
//! ```rust
//! // Gomez is based on `nalgebra` crate.
//! use gomez::nalgebra as na;
//! use gomez::{Domain, Problem, System};
//! use na::{Dyn, IsContiguous};
//!
//! // A problem is represented by a type.
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! impl Problem for Rosenbrock {
//!     // The numeric type. Usually f64 or f32.
//!     type Field = f64;
//!
//!     // Specification for the domain. At the very least, the dimension
//!     // must be known.
//!     fn domain(&self) -> Domain<Self::Field> {
//!         Domain::unconstrained(2)
//!     }
//! }
//!
//! impl System for Rosenbrock {
//!     // Evaluate trial values of variables to the system.
//!     fn eval<Sx, Sfx>(
//!         &self,
//!         x: &na::Vector<Self::Field, Dyn, Sx>,
//!         fx: &mut na::Vector<Self::Field, Dyn, Sfx>,
//!     ) where
//!         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
//!         Sfx: na::storage::StorageMut<Self::Field, Dyn>,
//!     {
//!         // Compute the residuals of all equations.
//!         fx[0] = (self.a - x[0]).powi(2);
//!         fx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
//!     }
//! }
//! ```
//!
//! And that's it. There is no need for defining gradient vector, Hessian or
//! Jacobian matrices. The library uses [finite
//! difference](https://en.wikipedia.org/wiki/Finite_difference_method)
//! technique (usually sufficient in practice) or algorithms that are
//! derivative-free by definition.
//!
//! The previous example used unconstrained variable, but it is also possible to
//! specify bounds.
//!
//! ```rust
//! # use gomez::nalgebra as na;
//! # use gomez::*;
//! #
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! impl Problem for Rosenbrock {
//! #     type Field = f64;
//!     // ...
//!
//!     fn domain(&self) -> Domain<Self::Field> {
//!         [(-10.0, 10.0), (-10.0, 10.0)].into_iter().collect()
//!     }
//! }
//! ```
//!
//! ## Solving
//!
//! When you have your system available, you can use the [`SolverDriver`] to run
//! the iteration process until a stopping criterion is reached.
//!
//! ```rust
//! use gomez::SolverDriver;
//! # use gomez::nalgebra as na;
//! # use gomez::{Domain, Problem, System};
//! # use na::{Dyn, IsContiguous};
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
//! #         Domain::unconstrained(2)
//! #     }
//! # }
//! #
//! # impl System for Rosenbrock {
//! #     fn eval<Sx, Sfx>(
//! #         &self,
//! #         x: &na::Vector<Self::Field, Dyn, Sx>,
//! #         fx: &mut na::Vector<Self::Field, Dyn, Sfx>,
//! #     ) where
//! #         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
//! #         Sfx: na::storage::StorageMut<Self::Field, Dyn>,
//! #     {
//! #         fx[0] = (self.a - x[0]).powi(2);
//! #         fx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
//! #     }
//! # }
//!
//! let f = Rosenbrock { a: 1.0, b: 1.0 };
//! let mut solver = SolverDriver::builder(&f)
//!     .with_initial(vec![-10.0, -5.0])
//!     .build();
//!
//! let tolerance = 1e-6;
//!
//! let result = solver
//!     .find(|state| {
//!         println!(
//!             "iter = {}\t|| fx || = {}\tx = {:?}",
//!             state.iter(),
//!             state.norm(),
//!             state.x()
//!         );
//!         state.norm() <= tolerance || state.iter() >= 100
//!     })
//!     .expect("solver encountered an error");
//!
//! if result <= tolerance {
//!     println!("solved");
//! } else {
//!     println!("maximum number of iteration exceeded");
//! }
//! ```
//!
//! ## Roadmap
//!
//! Listed *not* in order of priority.
//!
//! * [Homotopy continuation
//!   method](http://homepages.math.uic.edu/~jan/srvart/node4.html) to compare
//!   the performance with Trust region method.
//! * Conjugate gradient method
//! * Experimentation with various global optimization techniques for initial
//!   guesses search
//!   * Evolutionary/nature-inspired algorithms
//!   * Bayesian optimization
//! * Focus on initial guesses search and tools in general
//! * High-level drivers encapsulating the low-level API for users that do not
//!   need the fine-grained control.
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
