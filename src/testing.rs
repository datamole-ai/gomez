//! Testing systems and utilities useful for benchmarking, debugging and smoke
//! testing.
//!
//! [`ExtendedRosenbrock`] and [`Sphere`] are recommended for first tests.
//! Others can be used for specific conditions (e.g., singular Jacobian matrix).
//!
//! # References
//!
//! \[1\] [A Literature Survey of Benchmark Functions For Global Optimization
//! Problems](https://arxiv.org/abs/1308.4008)
//!
//! \[2\] [Handbook of Test Problems in Local and Global
//! Optimization](https://link.springer.com/book/10.1007/978-1-4757-3040-1)
//!
//! \[3\] [Numerical Methods for Unconstrained Optimization and Nonlinear
//! Equations](https://epubs.siam.org/doi/book/10.1137/1.9781611971200)
//!
//! \[4\] [HOMPACK: A Suite of Codes for Globally Convergent Homotopy
//! Algorithms](https://dl.acm.org/doi/10.1145/29380.214343)

use std::error::Error as StdError;

use nalgebra::{
    allocator::Allocator,
    storage::{Storage, StorageMut},
    vector, DVector, DefaultAllocator, Dim, DimName, Dynamic, IsContiguous, OVector, Vector, U1,
    U2,
};
use thiserror::Error;

use crate::{
    core::{Domain, Problem, ProblemError, Solver, System},
    var,
};

/// Extension of the [`System`] trait that provides additional information that
/// is useful for testing solvers.
pub trait TestSystem: System
where
    DefaultAllocator: Allocator<Self::Scalar, Self::Dim>,
    Self::Scalar: approx::RelativeEq,
{
    /// Standard initial values for the system. Using the same initial values is
    /// essential for fair comparison of methods.
    fn initials(&self) -> Vec<OVector<Self::Scalar, Self::Dim>>;

    /// A set of roots (if known and finite). This is mostly just for
    /// information, for example to know how close a solver got even if it
    /// failed. For testing if a given point is root, [`TestSystem::is_root`]
    /// should be used.
    fn roots(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        Vec::new()
    }

    /// Test if given point is a root of the system, given the tolerance `eps`.
    fn is_root<Sx>(&self, x: &Vector<Self::Scalar, Self::Dim, Sx>, eps: Self::Scalar) -> bool
    where
        Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
    {
        let mut fx = x.clone_owned();
        if self.eval(x, &mut fx).is_ok() {
            fx.norm() <= eps
        } else {
            false
        }
    }
}

/// [Extended Rosenbrock
/// function](https://en.wikipedia.org/wiki/Rosenbrock_function) \[1,3\] (also
/// known as Rosenbrock's valley or banana function).
///
/// The global minimum is inside a long, narrow, parabolic shaped flat valley.
/// The challenge is to find the solution inside the valley.
///
/// # References
///
/// \[1\] [A Literature Survey of Benchmark Functions For Global Optimization
/// Problems](https://arxiv.org/abs/1308.4008)
///
/// \[3\] [Numerical Methods for Unconstrained Optimization and Nonlinear
/// Equations](https://epubs.siam.org/doi/book/10.1137/1.9781611971200)
#[derive(Debug, Clone, Copy)]
pub struct ExtendedRosenbrock {
    n: usize,
    alpha: f64,
}

impl ExtendedRosenbrock {
    /// Initializes the system with given dimension.
    ///
    /// The dimension **must** be a multiplier of 2.
    pub fn new(n: usize) -> Self {
        Self::with_scaling(n, 1.0)
    }

    /// Initializes the system with given dimension and scaling factor.
    ///
    /// The dimension **must** be a multiplier of 2. The higher the scaling
    /// factor is, the more difficult the system is.
    pub fn with_scaling(n: usize, alpha: f64) -> Self {
        assert!(n > 0, "n must be greater than zero");
        assert!(n % 2 == 0, "n must be a multiple of 2");
        assert!(alpha > 0.0, "alpha must be greater than zero");
        Self { n, alpha }
    }
}

impl Default for ExtendedRosenbrock {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Problem for ExtendedRosenbrock {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dim(&self) -> Self::Dim {
        Dynamic::from_usize(self.n)
    }

    fn domain(&self) -> Domain<Self::Scalar> {
        (0..self.n)
            .map(|i| {
                if i % 2 == 0 {
                    self.alpha
                } else {
                    1.0 / self.alpha
                }
            })
            .map(|m| var!(m))
            .collect()
    }
}

impl System for ExtendedRosenbrock {
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        for i in 0..(self.n / 2) {
            let i1 = 2 * i;
            let i2 = 2 * i + 1;

            let x1 = x[i1] * self.alpha;
            let x2 = x[i2] / self.alpha;

            fx[i1] = 10.0 * (x2 - x1 * x1);
            fx[i2] = 1.0 - x1;
        }

        Ok(())
    }
}

impl TestSystem for ExtendedRosenbrock {
    fn initials(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        let init1 = DVector::from_iterator(
            self.n,
            (0..self.n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }),
        );

        let init2 = DVector::from_iterator(
            self.n,
            (0..self.n).map(|i| if i % 2 == 0 { 6.39 } else { -0.221 }),
        );

        vec![init1, init2]
    }

    fn roots(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        let root = (0..self.n).map(|i| {
            if i % 2 == 0 {
                1.0 / self.alpha
            } else {
                self.alpha
            }
        });

        vec![DVector::from_iterator(self.n, root)]
    }
}

/// Extended Powell function \[1,3\].
///
/// Both the gradient and the Jacobian matrix is singular in the solution.
///
/// # References
///
/// \[1\] [A Literature Survey of Benchmark Functions For Global Optimization
/// Problems](https://arxiv.org/abs/1308.4008)
///
/// \[3\] [Numerical Methods for Unconstrained Optimization and Nonlinear
/// Equations](https://epubs.siam.org/doi/book/10.1137/1.9781611971200)
#[derive(Debug, Clone, Copy)]
pub struct ExtendedPowell {
    n: usize,
}

impl ExtendedPowell {
    /// Initializes the system with given dimension.
    ///
    /// The dimension **must** be a multiplier of 4.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "n must be greater than zero");
        assert!(n % 4 == 0, "n must be a multiple of 4");
        Self { n }
    }
}

impl Default for ExtendedPowell {
    fn default() -> Self {
        Self::new(4)
    }
}

impl Problem for ExtendedPowell {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dim(&self) -> Self::Dim {
        Dynamic::from_usize(self.n)
    }
}

impl System for ExtendedPowell {
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        for i in 0..(self.n / 4) {
            let i1 = 4 * i;
            let i2 = 4 * i + 1;
            let i3 = 4 * i + 2;
            let i4 = 4 * i + 3;

            fx[i1] = x[i1] + 10.0 * x[i2];
            fx[i2] = 5f64.sqrt() * (x[i3] - x[i4]);
            fx[i3] = (x[i2] - 2.0 * x[i3]).powi(2);
            fx[i4] = 10f64.sqrt() * (x[i1] - x[i4]).powi(2);
        }

        Ok(())
    }
}

impl TestSystem for ExtendedPowell {
    fn initials(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        let init = DVector::from_iterator(
            self.n,
            (0..self.n).map(|i| match i % 4 {
                0 => 3.0,
                1 => -1.0,
                2 => 0.0,
                3 => 1.0,
                _ => unreachable!(),
            }),
        );

        vec![init]
    }

    fn roots(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        vec![DVector::from_element(self.n, 0.0)]
    }
}

/// Bullard and Biegler \[2\].
///
/// Many initial guesses (e.g., (1, 1)) lead numerical methods to a root that is
/// out of of the bounds.
///
/// # References
///
/// \[2\] [Handbook of Test Problems in Local and Global
/// Optimization](https://link.springer.com/book/10.1007/978-1-4757-3040-1)
#[derive(Debug, Clone, Copy)]
pub struct BullardBiegler(());

impl BullardBiegler {
    /// Initializes the system.
    pub fn new() -> Self {
        Self(())
    }
}

impl Default for BullardBiegler {
    fn default() -> Self {
        Self::new()
    }
}

impl Problem for BullardBiegler {
    type Scalar = f64;
    type Dim = U2;

    fn dim(&self) -> Self::Dim {
        U2::name()
    }

    fn domain(&self) -> Domain<Self::Scalar> {
        vec![var!(5.45e-6, 4.553), var!(2.196e-3, 18.21)].into()
    }
}

impl System for BullardBiegler {
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        fx[0] = 1e4 * x[0] * x[1] - 1.0;
        fx[1] = (-x[0]).exp() + (-x[1]).exp() - 1.001;

        Ok(())
    }
}

impl TestSystem for BullardBiegler {
    fn initials(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        let init1 = vector![0.1, 1.0];
        let init2 = vector![1.0, 1.0];
        vec![init1, init2]
    }

    fn roots(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        vec![vector![1.45e-5, 6.8933353]]
    }
}

/// [Sphere
/// function](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
/// \[1\].
///
/// This is a simple paraboloid which can be used in early development and
/// sanity checking as it can be considered a trivial problem.
///
/// # References
///
/// \[1\] [A Literature Survey of Benchmark Functions For Global Optimization
/// Problems](https://arxiv.org/abs/1308.4008)
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    n: usize,
}

impl Sphere {
    /// Initializes the system with given dimension.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "n must be greater than zero");
        Self { n }
    }
}

impl Default for Sphere {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Problem for Sphere {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dim(&self) -> Self::Dim {
        Dynamic::from_usize(self.n)
    }
}

impl System for Sphere {
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        for i in 0..self.n {
            fx[i] = x[i].powi(2);
        }

        Ok(())
    }
}

impl TestSystem for Sphere {
    fn initials(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        let init = DVector::from_iterator(
            self.n,
            (0..self.n).map(|i| if i % 2 == 0 { 10.0 } else { -10.0 }),
        );

        vec![init]
    }

    fn roots(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        vec![DVector::from_element(self.n, 0.0)]
    }
}

/// Brown function \[4\].
///
/// A function with ill-conditioned Jacobian matrix.
///
/// # References
///
/// \[4\] [HOMPACK: A Suite of Codes for Globally Convergent Homotopy
/// Algorithms](https://dl.acm.org/doi/10.1145/29380.214343)
#[derive(Debug, Clone, Copy)]
pub struct Brown {
    n: usize,
}

impl Brown {
    /// Initializes the system with given dimension.
    ///
    /// The dimension **must** be greater than 1.
    pub fn new(n: usize) -> Self {
        assert!(n > 1, "n must be greater than one");
        Self { n }
    }
}

impl Default for Brown {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Problem for Brown {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dim(&self) -> Self::Dim {
        Dynamic::from_usize(self.n)
    }
}

impl System for Brown {
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        fx[0] = x.iter().product::<f64>() - 1.0;

        for i in 1..self.n {
            fx[i] = x[i] + x.sum() - (self.n as f64 + 1.0);
        }

        Ok(())
    }
}

impl TestSystem for Brown {
    fn initials(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        let init = DVector::zeros_generic(Dynamic::from_usize(self.n), U1::name());
        vec![init]
    }
}

/// Exponential function \[4\].
///
/// A function whose zero path in [Homotopy continuation
/// methods](http://homepages.math.uic.edu/~jan/srvart/node4.html) has several
/// sharp turns.
///
/// # References
///
/// \[4\] [HOMPACK: A Suite of Codes for Globally Convergent Homotopy
/// Algorithms](https://dl.acm.org/doi/10.1145/29380.214343)
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    n: usize,
}

impl Exponential {
    /// Initializes the system with given dimension.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "n must be greater than zero");
        Self { n }
    }
}

impl Default for Exponential {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Problem for Exponential {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dim(&self) -> Self::Dim {
        Dynamic::from_usize(self.n)
    }
}

impl System for Exponential {
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        for i in 0..self.n {
            fx[i] = x[i] - (((i + 1) as f64) * x.sum()).cos().exp();
        }

        Ok(())
    }
}

impl TestSystem for Exponential {
    fn initials(&self) -> Vec<OVector<Self::Scalar, Self::Dim>> {
        let init = DVector::zeros_generic(Dynamic::from_usize(self.n), U1::name());
        vec![init]
    }
}

/// Solving error of the testing solver driver (see [`solve`]).
#[derive(Debug, Error)]
pub enum SolveError<E: StdError + 'static> {
    /// Error of the solver used.
    #[error("{0}")]
    Solver(#[from] E),
    /// Solver did not terminate.
    #[error("solver did not terminate")]
    Termination,
}

/// A simple solver driver that can be used in tests.
pub fn solve<F: TestSystem, S: Solver<F>>(
    f: &F,
    dom: &Domain<F::Scalar>,
    mut solver: S,
    mut x: OVector<F::Scalar, F::Dim>,
    max_iters: usize,
    tolerance: F::Scalar,
) -> Result<OVector<F::Scalar, F::Dim>, SolveError<S::Error>>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    S::Error: StdError,
{
    let mut fx = x.clone_owned();
    let mut iter = 0;

    loop {
        solver.next(f, dom, &mut x, &mut fx)?;

        if fx.norm() <= tolerance {
            return Ok(x);
        }

        if iter == max_iters {
            return Err(SolveError::Termination);
        } else {
            iter += 1;
        }
    }
}

/// Iterate the solver and inspect it in each iteration. This is useful for
/// testing evolutionary/nature-inspired algorithms.
pub fn iter<F: TestSystem, S: Solver<F>, G>(
    f: &F,
    dom: &Domain<F::Scalar>,
    mut solver: S,
    mut x: OVector<F::Scalar, F::Dim>,
    iters: usize,
    mut inspect: G,
) -> Result<(), S::Error>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    S::Error: StdError,
    G: FnMut(&S, &OVector<F::Scalar, F::Dim>, F::Scalar, usize),
{
    let mut fx = x.clone_owned();

    for iter in 0..iters {
        solver.next(f, dom, &mut x, &mut fx)?;
        inspect(&solver, &x, fx.norm(), iter);
    }

    Ok(())
}
