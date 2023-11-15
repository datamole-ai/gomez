//! High-level API for optimization and solving.
//!
//! This module contains "drivers" that encapsulate all internal state and
//! provide a simple API to run the iterative process for optimization or
//! solving. This documentation describes usage for the solving systems of
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
//! let r = MySystem::new();
//!
//! let mut solver = SolverDriver::new(&r);
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
//! let r = MySystem::new();
//!
//! let mut solver = SolverDriver::builder(&r)
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
//! #     fn eval<Sx, Srx>(
//! #         &self,
//! #         x: &na::Vector<Self::Field, Dyn, Sx>,
//! #         rx: &mut na::Vector<Self::Field, Dyn, Srx>,
//! #     ) where
//! #         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
//! #         Srx: na::storage::StorageMut<Self::Field, Dyn>,
//! #     {
//! #         rx[0] = x[0] + x[1] + 1.0;
//! #         rx[1] = (x[0] + x[1] - 1.0).powi(2);
//! #     }
//! # }
//! #
//! # let r = MySystem::new();
//! #
//! # let mut solver = SolverDriver::new(&r);
//! #
//! // Solution or solver error.
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
//! #     fn eval<Sx, Srx>(
//! #         &self,
//! #         x: &na::Vector<Self::Field, Dyn, Sx>,
//! #         rx: &mut na::Vector<Self::Field, Dyn, Srx>,
//! #     ) where
//! #         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
//! #         Srx: na::storage::StorageMut<Self::Field, Dyn>,
//! #     {
//! #         rx[0] = x[0] + x[1] + 1.0;
//! #         rx[1] = (x[0] + x[1] - 1.0).powi(2);
//! #     }
//! # }
//! #
//! # let r = MySystem::new();
//! #
//! # let mut solver = SolverDriver::new(&r);
//! #
//! loop {
//!     // Current point or solver error.
//!     let result = solver.next();
//!     // ...
//! #   break;
//! }
//! ```

use nalgebra::{convert, DimName, Dyn, OVector, U1};

use crate::{algo::TrustRegion, Domain, Function, Optimizer, Problem, Solver, System};

struct Builder<'a, P: Problem, A> {
    p: &'a P,
    dom: Domain<P::Field>,
    algo: A,
    x0: OVector<P::Field, Dyn>,
}

impl<'a, P: Problem> Builder<'a, P, TrustRegion<P>> {
    fn new(p: &'a P) -> Self {
        let dom = p.domain();
        let algo = TrustRegion::new(p, &dom);

        let dim = Dyn(dom.dim());
        let x0 = OVector::from_element_generic(dim, U1::name(), convert(0.0));

        Self { p, dom, algo, x0 }
    }
}

impl<'a, P: Problem, A> Builder<'a, P, A> {
    fn with_initial(mut self, x0: Vec<P::Field>) -> Self {
        let dim = Dyn(self.dom.dim());
        self.x0 = OVector::from_vec_generic(dim, U1::name(), x0);
        self
    }

    fn with_algo<S2, FA>(self, factory: FA) -> Builder<'a, P, S2>
    where
        FA: FnOnce(&P, &Domain<P::Field>) -> S2,
    {
        let algo = factory(self.p, &self.dom);

        Builder {
            p: self.p,
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
pub struct SolverBuilder<'a, R: Problem, A>(Builder<'a, R, A>);

impl<'a, R: Problem, A> SolverBuilder<'a, R, A> {
    /// Sets the initial point from which the iterative process starts.
    pub fn with_initial(self, x0: Vec<R::Field>) -> Self {
        Self(self.0.with_initial(x0))
    }

    /// Sets specific algorithm to be used.
    ///
    /// This builder method accepts a closure that takes the reference to the
    /// problem and its domain. For many algorithms in gomez, you can simply
    /// pass the `new` constructor directly (e.g., `TrustRegion::new`).
    pub fn with_algo<S2, FA>(self, factory: FA) -> SolverBuilder<'a, R, S2>
    where
        FA: FnOnce(&R, &Domain<R::Field>) -> S2,
    {
        SolverBuilder(self.0.with_algo(factory))
    }

    /// Builds the [`SolverDriver`].
    pub fn build(self) -> SolverDriver<'a, R, A> {
        let Builder {
            p: r,
            dom,
            algo,
            x0,
        } = self.0.build();
        let rx = x0.clone_owned();

        SolverDriver {
            r,
            dom,
            algo,
            x: x0,
            rx,
        }
    }
}

/// The driver for the process of solving a system of equations.
///
/// For default settings, use [`SolverDriver::new`]. For more flexibility, use
/// [`SolverDriver::builder`]. For the usage of the driver, see [module](self)
/// documentation.
pub struct SolverDriver<'a, R: Problem, A> {
    r: &'a R,
    dom: Domain<R::Field>,
    algo: A,
    x: OVector<R::Field, Dyn>,
    rx: OVector<R::Field, Dyn>,
}

impl<'a, R: Problem> SolverDriver<'a, R, TrustRegion<R>> {
    /// Returns the builder for specifying additional settings.
    pub fn builder(r: &'a R) -> SolverBuilder<'a, R, TrustRegion<R>> {
        SolverBuilder(Builder::new(r))
    }

    /// Initializes the driver with the default settings.
    pub fn new(r: &'a R) -> Self {
        SolverDriver::builder(r).build()
    }
}

impl<'a, R: Problem, S> SolverDriver<'a, R, S> {
    /// Returns reference to the current point.
    pub fn x(&self) -> &[R::Field] {
        self.x.as_slice()
    }

    /// Returns reference to the current residuals.
    pub fn rx(&self) -> &[R::Field] {
        self.rx.as_slice()
    }

    /// Returns norm of the residuals.
    pub fn norm(&self) -> R::Field {
        self.rx.norm()
    }
}

impl<'a, R: System, A: Solver<R>> SolverDriver<'a, R, A> {
    /// Performs one iteration of the process, returning the current point and
    /// norm of the residuals in case of no error.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<(&[R::Field], R::Field), A::Error> {
        self.algo
            .solve_next(self.r, &self.dom, &mut self.x, &mut self.rx)?;
        Ok((self.x.as_slice(), self.rx.norm()))
    }

    /// Runs the iterative process until given stopping criterion is satisfied.
    pub fn find<C>(&mut self, stop: C) -> Result<(&[R::Field], R::Field), A::Error>
    where
        C: Fn(SolverIterState<'_, R>) -> bool,
    {
        let mut iter = 0;

        loop {
            let norm = self.next()?.1;

            let state = SolverIterState {
                x: &self.x,
                rx: &self.rx,
                iter,
            };

            if stop(state) {
                return Ok((self.x.as_slice(), norm));
            }

            iter += 1;
        }
    }

    /// Returns the name of the solver.
    pub fn name(&self) -> &str {
        A::NAME
    }
}

/// State of the current iteration in the solving process.
pub struct SolverIterState<'a, R: Problem> {
    x: &'a OVector<R::Field, Dyn>,
    rx: &'a OVector<R::Field, Dyn>,
    iter: usize,
}

impl<'a, R: Problem> SolverIterState<'a, R> {
    /// Returns reference to the current point.
    pub fn x(&self) -> &[R::Field] {
        self.x.as_slice()
    }

    /// Returns reference to the current residuals.
    pub fn rx(&self) -> &[R::Field] {
        self.rx.as_slice()
    }

    /// Returns norm of the current residuals.
    pub fn norm(&self) -> R::Field {
        self.rx.norm()
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
        let Builder {
            p: f,
            dom,
            algo,
            x0,
        } = self.0.build();

        OptimizerDriver {
            f,
            dom,
            algo,
            x: x0,
            fx: convert(f64::INFINITY),
        }
    }
}

/// The driver for the process of optimizing a function.
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
    /// Performs one iteration of the process, returning the current point and
    /// function value in case of no error.
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

    /// Returns the name of the optimizer.
    pub fn name(&self) -> &str {
        A::NAME
    }
}

/// State of the current iteration if the optimization process.
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
        let r = Sphere::new(4);
        let mut solver = SolverDriver::builder(&r)
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
        let r = Sphere::new(1);
        let mut solver = SolverDriver::builder(&r)
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
