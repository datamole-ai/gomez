//! High-level API for solving and optimization.
//!
//! This module contains "drivers" that encapsulate all internal state and
//! provide a simple API to run the iterative process for solving or
//! optimization. This documentation describes usage for the solving systems of
//! equations, but the API is basically the same for optimization.
//!
//! The simplest way of using the driver is to initialize it with the defaults:
//!
//! ```rust
//! use gomez::SolverDriver;
//! # use gomez::{Domain, Problem};
//! #
//! # struct MySystem;
//! #
//! # impl MySystem {
//! #     fn new() -> Self {
//! #         Self
//! #     }
//! # }
//! #
//! # impl Problem for MySystem {
//! #     type Field = f64;
//! #
//! #     fn domain(&self) -> Domain<Self::Field> {
//! #         Domain::unconstrained(2)
//! #     }
//! # }
//!
//! let f = MySystem::new();
//!
//! let mut solver = SolverDriver::new(&f);
//! ```
//! If you need to specify additional settings, use the builder:
//!
//! ```rust
//! use gomez::SolverDriver;
//! # use gomez::{Domain, Problem};
//! #
//! # struct MySystem;
//! #
//! # impl MySystem {
//! #     fn new() -> Self {
//! #         Self
//! #     }
//! # }
//! #
//! # impl Problem for MySystem {
//! #     type Field = f64;
//! #
//! #     fn domain(&self) -> Domain<Self::Field> {
//! #         Domain::unconstrained(2)
//! #     }
//! # }
//!
//! let f = MySystem::new();
//!
//! let mut solver = SolverDriver::builder(&f)
//!     .with_initial(vec![10.0, -10.0])
//!     .with_algo(gomez::algo::NelderMead::new)
//!     .build();
//! ```
//!
//! Once you have the solver, you can use it to find the solution:
//!
//! ```rust
//! # use gomez::nalgebra as na;
//! # use gomez::{Domain, Problem, SolverDriver, System};
//! # use na::{Dyn, IsContiguous};
//! #
//! # struct MySystem;
//! #
//! # impl MySystem {
//! #     fn new() -> Self {
//! #         Self
//! #     }
//! # }
//! #
//! # impl Problem for MySystem {
//! #     type Field = f64;
//! #
//! #     fn domain(&self) -> Domain<Self::Field> {
//! #         Domain::unconstrained(2)
//! #     }
//! # }
//! #
//! # impl System for MySystem {
//! #     fn eval<Sx, Sfx>(
//! #         &self,
//! #         x: &na::Vector<Self::Field, Dyn, Sx>,
//! #         fx: &mut na::Vector<Self::Field, Dyn, Sfx>,
//! #     ) where
//! #         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
//! #         Sfx: na::storage::StorageMut<Self::Field, Dyn>,
//! #     {
//! #         fx[0] = x[0] + x[1] + 1.0;
//! #         fx[1] = (x[0] + x[1] - 1.0).powi(2);
//! #     }
//! # }
//! #
//! # let f = MySystem::new();
//! #
//! # let mut solver = SolverDriver::new(&f);
//! #
//! let result = solver.find(|state| state.norm() <= 1e-6 || state.iter() >= 100);
//! ```
//!
//! If you need more control over the iteration process, you can do the
//! iterations manually:
//!
//! ```rust
//! # use gomez::nalgebra as na;
//! # use gomez::{Domain, Problem, SolverDriver, System};
//! # use na::{Dyn, IsContiguous};
//! #
//! # struct MySystem;
//! #
//! # impl MySystem {
//! #     fn new() -> Self {
//! #         Self
//! #     }
//! # }
//! #
//! # impl Problem for MySystem {
//! #     type Field = f64;
//! #
//! #     fn domain(&self) -> Domain<Self::Field> {
//! #         Domain::unconstrained(2)
//! #     }
//! # }
//! #
//! # impl System for MySystem {
//! #     fn eval<Sx, Sfx>(
//! #         &self,
//! #         x: &na::Vector<Self::Field, Dyn, Sx>,
//! #         fx: &mut na::Vector<Self::Field, Dyn, Sfx>,
//! #     ) where
//! #         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
//! #         Sfx: na::storage::StorageMut<Self::Field, Dyn>,
//! #     {
//! #         fx[0] = x[0] + x[1] + 1.0;
//! #         fx[1] = (x[0] + x[1] - 1.0).powi(2);
//! #     }
//! # }
//! #
//! # let f = MySystem::new();
//! #
//! # let mut solver = SolverDriver::new(&f);
//! #
//! loop {
//!     let norm = solver.next().expect("no solver error");
//!     // ...
//! #   break;
//! }
//! ```

use nalgebra::{convert, DimName, Dyn, OVector, U1};

use crate::{algo::TrustRegion, Domain, Function, Optimizer, Problem, Solver, System};

struct Builder<'a, F: Problem, A> {
    f: &'a F,
    dom: Domain<F::Field>,
    algo: A,
    x0: OVector<F::Field, Dyn>,
}

impl<'a, F: Problem> Builder<'a, F, TrustRegion<F>> {
    fn new(f: &'a F) -> Self {
        let dom = f.domain();
        let algo = TrustRegion::new(f, &dom);

        let dim = Dyn(dom.dim());
        let x0 = OVector::from_element_generic(dim, U1::name(), convert(0.0));

        Self { f, dom, algo, x0 }
    }
}

impl<'a, F: Problem, A> Builder<'a, F, A> {
    fn with_initial(mut self, x0: Vec<F::Field>) -> Self {
        let dim = Dyn(self.dom.dim());
        self.x0 = OVector::from_vec_generic(dim, U1::name(), x0);
        self
    }

    fn with_algo<S2, FA>(self, factory: FA) -> Builder<'a, F, S2>
    where
        FA: FnOnce(&F, &Domain<F::Field>) -> S2,
    {
        let algo = factory(self.f, &self.dom);

        Builder {
            f: self.f,
            dom: self.dom,
            algo,
            x0: self.x0,
        }
    }

    fn build(mut self) -> Self {
        self.dom.project(&mut self.x0);
        self
    }
}

/// Builder for the [`SolverDriver`].
pub struct SolverBuilder<'a, F: Problem, A>(Builder<'a, F, A>);

impl<'a, F: Problem, A> SolverBuilder<'a, F, A> {
    /// Sets the initial point from which the iterative process starts.
    pub fn with_initial(self, x0: Vec<F::Field>) -> Self {
        Self(self.0.with_initial(x0))
    }

    /// Sets specific algorithm to be used.
    ///
    /// This builder method accepts a closure that takes the reference to the
    /// problem and its domain. For many algorithms in gomez, you can simply
    /// pass the `new` constructor directly (e.g., `TrustRegion::new`).
    pub fn with_algo<S2, FA>(self, factory: FA) -> SolverBuilder<'a, F, S2>
    where
        FA: FnOnce(&F, &Domain<F::Field>) -> S2,
    {
        SolverBuilder(self.0.with_algo(factory))
    }

    /// Builds the [`SolverDriver`].
    pub fn build(self) -> SolverDriver<'a, F, A> {
        let Builder { f, dom, algo, x0 } = self.0.build();
        let fx = x0.clone_owned();

        SolverDriver {
            f,
            dom,
            algo,
            x: x0,
            fx,
        }
    }
}

/// The driver for the process of solving a system of equations.
///
/// For default settings, use [`SolverDriver::new`]. For more flexibility, use
/// [`SolverDriver::builder`]. For the usage of the driver, see [module](self)
/// documentation.
pub struct SolverDriver<'a, F: Problem, A> {
    f: &'a F,
    dom: Domain<F::Field>,
    algo: A,
    x: OVector<F::Field, Dyn>,
    fx: OVector<F::Field, Dyn>,
}

impl<'a, F: Problem> SolverDriver<'a, F, TrustRegion<F>> {
    /// Returns the builder for specifying additional settings.
    pub fn builder(f: &'a F) -> SolverBuilder<'a, F, TrustRegion<F>> {
        SolverBuilder(Builder::new(f))
    }

    /// Initializes the driver with the default settings.
    pub fn new(f: &'a F) -> Self {
        SolverDriver::builder(f).build()
    }
}

impl<'a, F: Problem, S> SolverDriver<'a, F, S> {
    /// Returns reference to the current point.
    pub fn x(&self) -> &[F::Field] {
        self.x.as_slice()
    }

    /// Returns reference to the current residuals.
    pub fn fx(&self) -> &[F::Field] {
        self.fx.as_slice()
    }

    /// Returns norm of the residuals.
    pub fn norm(&self) -> F::Field {
        self.fx.norm()
    }
}

impl<'a, F: System, A: Solver<F>> SolverDriver<'a, F, A> {
    /// Does one iteration of the process, returning the norm of the residuals
    /// in case of no error.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<(&[F::Field], F::Field), A::Error> {
        self.algo
            .solve_next(self.f, &self.dom, &mut self.x, &mut self.fx)?;
        Ok((self.x.as_slice(), self.fx.norm()))
    }

    /// Runs the iterative process until given stopping criterion is satisfied.
    pub fn find<C>(&mut self, stop: C) -> Result<(&[F::Field], F::Field), A::Error>
    where
        C: Fn(SolverIterState<'_, F>) -> bool,
    {
        let mut iter = 0;

        loop {
            let norm = self.next()?.1;

            let state = SolverIterState {
                x: &self.x,
                fx: &self.fx,
                iter,
            };

            if stop(state) {
                return Ok((self.x.as_slice(), norm));
            }

            iter += 1;
        }
    }

    /// Returns the name of the used solver.
    pub fn name(&self) -> &str {
        A::NAME
    }
}

/// State of the current iteration.
pub struct SolverIterState<'a, F: Problem> {
    x: &'a OVector<F::Field, Dyn>,
    fx: &'a OVector<F::Field, Dyn>,
    iter: usize,
}

impl<'a, F: Problem> SolverIterState<'a, F> {
    /// Returns reference to the current point.
    pub fn x(&self) -> &[F::Field] {
        self.x.as_slice()
    }

    /// Returns reference to the current residuals.
    pub fn fx(&self) -> &[F::Field] {
        self.fx.as_slice()
    }

    /// Returns norm of the residuals.
    pub fn norm(&self) -> F::Field {
        self.fx.norm()
    }

    /// Returns the current iteration number.
    pub fn iter(&self) -> usize {
        self.iter
    }
}

/// Builder for the [`OptimizerDriver`].
pub struct OptimizerBuilder<'a, F: Problem, A>(Builder<'a, F, A>);

impl<'a, F: Problem, A> OptimizerBuilder<'a, F, A> {
    /// Sets the initial point from which the iterative process starts.
    pub fn with_initial(self, x0: Vec<F::Field>) -> Self {
        Self(self.0.with_initial(x0))
    }

    /// Sets specific algorithm to be used.
    ///
    /// This builder method accepts a closure that takes the reference to the
    /// problem and its domain. For many algorithms in gomez, you can simply
    /// pass the `new` constructor directly (e.g., `TrustRegion::new`).
    pub fn with_algo<S2, FA>(self, factory: FA) -> OptimizerBuilder<'a, F, S2>
    where
        FA: FnOnce(&F, &Domain<F::Field>) -> S2,
    {
        OptimizerBuilder(self.0.with_algo(factory))
    }

    /// Builds the [`OptimizerDriver`].
    pub fn build(self) -> OptimizerDriver<'a, F, A> {
        let Builder { f, dom, algo, x0 } = self.0.build();

        OptimizerDriver {
            f,
            dom,
            algo,
            x: x0,
            fx: convert(f64::INFINITY),
        }
    }
}

/// The driver for the process of solving a system of equations.
///
/// For default settings, use [`OptimizerDriver::new`]. For more flexibility,
/// use [`OptimizerDriver::builder`]. For the usage of the driver, see
/// [module](self) documentation.
pub struct OptimizerDriver<'a, F: Problem, A> {
    f: &'a F,
    dom: Domain<F::Field>,
    algo: A,
    x: OVector<F::Field, Dyn>,
    fx: F::Field,
}

impl<'a, F: Problem> OptimizerDriver<'a, F, TrustRegion<F>> {
    /// Returns the builder for specifying additional settings.
    pub fn builder(f: &'a F) -> OptimizerBuilder<'a, F, TrustRegion<F>> {
        OptimizerBuilder(Builder::new(f))
    }

    /// Initializes the driver with the default settings.
    pub fn new(f: &'a F) -> Self {
        OptimizerDriver::builder(f).build()
    }
}

impl<'a, F: Problem, A> OptimizerDriver<'a, F, A> {
    /// Returns reference to the current point.
    pub fn x(&self) -> &[F::Field] {
        self.x.as_slice()
    }

    /// Returns the current function value.
    pub fn fx(&self) -> F::Field {
        self.fx
    }
}

impl<'a, F: Function, A: Optimizer<F>> OptimizerDriver<'a, F, A> {
    /// Does one iteration of the process, returning the function value in case
    /// of no error.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<(&[F::Field], F::Field), A::Error> {
        self.algo
            .opt_next(self.f, &self.dom, &mut self.x)
            .map(|fx| (self.x.as_slice(), fx))
    }

    /// Runs the iterative process until given stopping criterion is satisfied.
    pub fn find<C>(&mut self, stop: C) -> Result<(&[F::Field], F::Field), A::Error>
    where
        C: Fn(OptimizerIterState<'_, F>) -> bool,
    {
        let mut iter = 0;

        loop {
            self.fx = self.next()?.1;

            let state = OptimizerIterState {
                x: &self.x,
                fx: self.fx,
                iter,
            };

            if stop(state) {
                return Ok((self.x.as_slice(), self.fx));
            }

            iter += 1;
        }
    }

    /// Returns the name of the used optimizer.
    pub fn name(&self) -> &str {
        A::NAME
    }
}

/// State of the current iteration.
pub struct OptimizerIterState<'a, F: Problem> {
    x: &'a OVector<F::Field, Dyn>,
    fx: F::Field,
    iter: usize,
}

impl<'a, F: Problem> OptimizerIterState<'a, F> {
    /// Returns reference to the current point.
    pub fn x(&self) -> &[F::Field] {
        self.x.as_slice()
    }

    /// Returns the current function value.
    pub fn fx(&self) -> F::Field {
        self.fx
    }

    /// Returns the current iteration number.
    pub fn iter(&self) -> usize {
        self.iter
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        algo::{NelderMead, Steffensen},
        testing::Sphere,
    };

    use super::*;

    struct WithDomain(pub Domain<f64>);

    impl Problem for WithDomain {
        type Field = f64;

        fn domain(&self) -> Domain<Self::Field> {
            self.0.clone()
        }
    }

    #[test]
    fn solver_basic_use_case() {
        let f = Sphere::new(4);
        let mut solver = SolverDriver::builder(&f)
            // Zeros are the root for sphere, there would be no point is such
            // test.
            .with_initial(vec![10.0; 4])
            .build();

        let tolerance = 1e-6;
        let (_, norm) = solver
            .find(|state| state.iter() >= 100 || state.norm() < tolerance)
            .unwrap();

        assert!(norm <= tolerance);
    }

    #[test]
    fn solver_custom() {
        let f = Sphere::new(1);
        let mut solver = SolverDriver::builder(&f)
            .with_algo(Steffensen::new)
            // Zeros is the root for sphere, there would be no point is such
            // test.
            .with_initial(vec![10.0])
            .build();

        let tolerance = 1e-6;
        let (_, norm) = solver
            .find(|state| state.iter() >= 100 || state.norm() < tolerance)
            .unwrap();

        assert!(norm <= tolerance);
    }

    #[test]
    fn solver_initial() {
        let x0 = vec![10.0; 4];

        let f = Sphere::new(4);
        let solver = SolverDriver::builder(&f).with_initial(x0.clone()).build();

        assert_eq!(solver.x(), &x0);
    }

    #[test]
    fn solver_initial_in_domain() {
        let f = WithDomain(Domain::rect(vec![0.0, 0.0], vec![1.0, 1.0]));
        let solver = SolverDriver::builder(&f)
            .with_initial(vec![10.0, -10.0])
            .build();

        assert_eq!(solver.x(), &[1.0, 0.0]);
    }

    #[test]
    fn optimizer_basic_use_case() {
        let f = Sphere::new(4);
        let mut optimizer = OptimizerDriver::builder(&f)
            // Zeros are the root for sphere, there would be no point is such
            // test.
            .with_initial(vec![10.0; 4])
            .build();

        let tolerance = 1e-6;
        let (_, value) = optimizer
            .find(|state| state.iter() >= 100 || state.fx() < tolerance)
            .unwrap();

        assert!(value <= tolerance);
    }

    #[test]
    fn optimizer_custom() {
        let f = Sphere::new(4);
        let mut optimizer = OptimizerDriver::builder(&f)
            .with_algo(NelderMead::new)
            // Zeros is the root for sphere, there would be no point is such
            // test.
            .with_initial(vec![10.0; 4])
            .build();

        let tolerance = 1e-6;
        let (_, value) = optimizer
            .find(|state| state.iter() >= 100 || state.fx() < tolerance)
            .unwrap();

        assert!(value <= tolerance);
    }

    #[test]
    fn optimizer_initial() {
        let x0 = vec![10.0; 4];

        let f = Sphere::new(4);
        let optimizer = OptimizerDriver::builder(&f)
            .with_initial(x0.clone())
            .build();

        assert_eq!(optimizer.x(), &x0);
    }

    #[test]
    fn optimizer_initial_in_domain() {
        let f = WithDomain(Domain::rect(vec![0.0, 0.0], vec![1.0, 1.0]));
        let optimizer = OptimizerDriver::builder(&f)
            .with_initial(vec![10.0, -10.0])
            .build();

        assert_eq!(optimizer.x(), &[1.0, 0.0]);
    }
}
