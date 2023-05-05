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
//! * [Trust region](solver::trust_region) -- Recommended method to be used as a
//!   default and it will just work in most of the cases.
//! * [Cuckoo search](solver::cuckoo) -- A global optimization algorithm useful
//!   for initial guesses search in combination with a numerical solver.
//! * [Steffensen](solver::steffensen) -- Fast and lightweight method for
//!   one-dimensional systems.
//! * [Nelder-Mead](solver::nelder_mead) -- Not generally recommended, but may
//!   be useful for low-dimensionality problems with ill-defined Jacobian
//!   matrix.
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
//! [`System`](core::System) and [`Problem`](core::Problem) traits.
//!
//! ```rust
//! // Gomez is based on `nalgebra` crate.
//! use gomez::nalgebra as na;
//! use gomez::prelude::*;
//! use na::{Dim, DimName, IsContiguous};
//!
//! // A problem is represented by a type.
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! impl Problem for Rosenbrock {
//!     // The numeric type. Usually f64 or f32.
//!     type Scalar = f64;
//!     // The dimension of the problem. Can be either statically known or dynamic.
//!     type Dim = na::U2;
//!
//!     // Return the actual dimension of the system.
//!     fn dim(&self) -> Self::Dim {
//!         na::U2::name()
//!     }
//! }
//!
//! impl System for Rosenbrock {
//!     // Evaluate trial values of variables to the system.
//!     fn eval<Sx, Sfx>(
//!         &self,
//!         x: &na::Vector<Self::Scalar, Self::Dim, Sx>,
//!         fx: &mut na::Vector<Self::Scalar, Self::Dim, Sfx>,
//!     ) -> Result<(), ProblemError>
//!     where
//!         Sx: na::storage::Storage<Self::Scalar, Self::Dim> + IsContiguous,
//!         Sfx: na::storage::StorageMut<Self::Scalar, Self::Dim>,
//!     {
//!         // Compute the residuals of all equations.
//!         fx[0] = (self.a - x[0]).powi(2);
//!         fx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
//!
//!         Ok(())
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
//! By default, the variables are considered unconstrained, but for constrained
//! problems it is just matter of overriding the default implementation of the
//! [`domain`](core::Problem::domain) method.
//!
//! ```rust
//! # use gomez::nalgebra as na;
//! # use gomez::prelude::*;
//! # use na::{Dim, DimName};
//! #
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! impl Problem for Rosenbrock {
//! #     type Scalar = f64;
//! #     type Dim = na::U2;
//! #
//! #     fn dim(&self) -> Self::Dim {
//! #         na::U2::name()
//! #     }
//!     // ...
//!
//!     fn domain(&self) -> Domain<Self::Scalar> {
//!         vec![var!(-10.0, 10.0), var!(-10.0, 10.0)].into()
//!     }
//! }
//! ```
//!
//! ## Solving
//!
//! When you have your system available, pick a solver of your choice and
//! iterate until the solution under certain convergence criteria is found or
//! termination limits are reached.
//!
//! **NOTE:** The following example uses low-level solver interface where you
//! are in full control of the iteration process and access to all intermediate
//! results. We plan to provide high-level drivers for convenience in the
//! future.
//!
//! ```rust
//! use gomez::nalgebra as na;
//! use gomez::prelude::*;
//! // Pick your solver.
//! use gomez::solver::TrustRegion;
//! # use na::{Dim, DimName, IsContiguous};
//! #
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # impl Problem for Rosenbrock {
//! #     type Scalar = f64;
//! #     type Dim = na::U2;
//! #
//! #     fn dim(&self) -> Self::Dim {
//! #         na::U2::name()
//! #     }
//! # }
//! #
//! # impl System for Rosenbrock {
//! #     fn eval<Sx, Sfx>(
//! #         &self,
//! #         x: &na::Vector<Self::Scalar, Self::Dim, Sx>,
//! #         fx: &mut na::Vector<Self::Scalar, Self::Dim, Sfx>,
//! #     ) -> Result<(), ProblemError>
//! #     where
//! #         Sx: na::storage::Storage<Self::Scalar, Self::Dim> + IsContiguous,
//! #         Sfx: na::storage::StorageMut<Self::Scalar, Self::Dim>,
//! #     {
//! #         fx[0] = (self.a - x[0]).powi(2);
//! #         fx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
//! #
//! #         Ok(())
//! #     }
//! # }
//!
//! let f = Rosenbrock { a: 1.0, b: 1.0 };
//! let dom = f.domain();
//!
//! let mut solver = TrustRegion::new(&f, &dom);
//!
//! // Initial guess. Good choice helps the convergence of numerical methods.
//! let mut x = na::vector![-10.0, -5.0];
//!
//! // Residuals vector.
//! let mut fx = na::vector![0.0, 0.0];
//!
//! for i in 1.. {
//!     // Do one iteration in the solving process.
//!     solver
//!         .next(&f, &dom, &mut x, &mut fx)
//!         .expect("solver encountered an error");
//!
//!     println!(
//!         "iter = {}\t|| fx || = {}\tx = {:?}",
//!         i,
//!         fx.norm(),
//!         x.as_slice()
//!     );
//!
//!     // Check the termination criteria.
//!     if fx.norm() < 1e-6 {
//!         println!("solved");
//!         break;
//!     } else if i == 100 {
//!         println!("maximum number of iteration exceeded");
//!         break;
//!     }
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

pub mod analysis;
pub mod core;
pub mod derivatives;
pub mod population;
pub mod solver;
pub mod testing;

pub use nalgebra;

/// Gomez prelude.
pub mod prelude {
    pub use crate::{
        core::{Domain, Problem, ProblemError, Solver, System, Variable, VariableBuilder},
        var,
    };
}
