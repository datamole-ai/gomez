//! Testing problems and utilities useful for benchmarking, debugging and smoke
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

#![allow(unused)]

use std::error::Error as StdError;

use nalgebra::{
    dvector,
    storage::{Storage, StorageMut},
    DVector, Dim, DimName, Dyn, IsContiguous, OVector, Vector, U1,
};
use thiserror::Error;

use crate::core::{Domain, Function, Optimizer, Problem, Solver, System};

/// Extension of the [`Problem`] trait that provides additional information that
/// is useful for testing algorithms.
pub trait TestProblem: Problem {
    /// Standard initial values for the problem. Using the same initial values is
    /// essential for fair comparison of methods.
    fn initials(&self) -> Vec<OVector<Self::Field, Dyn>>;
}

/// Extension of the [`System`] trait that provides additional information that
/// is useful for testing solvers.
pub trait TestSystem: System + TestProblem
where
    Self::Field: approx::RelativeEq,
{
    /// A set of roots (if known and finite).
    ///
    /// This is mostly just for information, for example to know how close a
    /// solver got even if it failed. For testing if a given point is a
    /// solution, [`TestSystem::is_root`] should be used.
    fn roots(&self) -> Vec<OVector<Self::Field, Dyn>> {
        Vec::new()
    }

    /// Tests if given point is a solution of the system, given the tolerance
    /// `eps`.
    fn is_root<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>, eps: Self::Field) -> bool
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        let mut rx = x.clone_owned();
        self.eval(x, &mut rx);
        rx.norm() <= eps
    }
}

/// Extension of the [`Function`] trait that provides additional information
/// that is useful for testing optimizers.
pub trait TestFunction: Function + TestProblem
where
    Self::Field: approx::RelativeEq,
{
    /// A set of global optima (if known and finite).
    ///
    /// This is mostly just for information, for example to know how close an
    /// optimizer got even if it failed. For testing if a given point is a global
    /// optimum, [`TestFunction::is_optimum`] should be used.
    fn optima(&self) -> Vec<OVector<Self::Field, Dyn>> {
        Vec::new()
    }

    /// Test if given point is a global optimum of the system, given the
    /// tolerance `eps`.
    fn is_optimum<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>, eps: Self::Field) -> bool
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous;
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

    fn residuals<'a, Sx>(&self, x: &'a Vector<f64, Dyn, Sx>) -> impl Iterator<Item = f64> + 'a
    where
        Sx: Storage<f64, Dyn> + IsContiguous,
    {
        let alpha = self.alpha;
        (0..(self.n / 2)).flat_map(move |i| {
            let i1 = 2 * i;
            let i2 = 2 * i + 1;

            let x1 = x[i1] * alpha;
            let x2 = x[i2] / alpha;

            let rx1 = 10.0 * (x2 - x1 * x1);
            let rx2 = 1.0 - x1;

            [rx1, rx2].into_iter()
        })
    }
}

impl Default for ExtendedRosenbrock {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Problem for ExtendedRosenbrock {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        (0..self.n)
            .map(|i| {
                if i % 2 == 0 {
                    self.alpha
                } else {
                    1.0 / self.alpha
                }
            })
            .collect()
    }
}

impl System for ExtendedRosenbrock {
    fn eval<Sx, Srx>(
        &self,
        x: &Vector<Self::Field, Dyn, Sx>,
        rx: &mut Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
        Srx: StorageMut<Self::Field, Dyn>,
    {
        eval(self.residuals(x), rx)
    }

    fn norm<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        norm(self.residuals(x))
    }
}

impl TestProblem for ExtendedRosenbrock {
    fn initials(&self) -> Vec<OVector<Self::Field, Dyn>> {
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
}

impl TestSystem for ExtendedRosenbrock {
    fn roots(&self) -> Vec<OVector<Self::Field, Dyn>> {
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

    fn residuals<'a, Sx>(&self, x: &'a Vector<f64, Dyn, Sx>) -> impl Iterator<Item = f64> + 'a
    where
        Sx: Storage<f64, Dyn> + IsContiguous,
    {
        (0..(self.n / 4)).flat_map(move |i| {
            let i1 = 4 * i;
            let i2 = 4 * i + 1;
            let i3 = 4 * i + 2;
            let i4 = 4 * i + 3;

            let rx1 = x[i1] + 10.0 * x[i2];
            let rx2 = 5f64.sqrt() * (x[i3] - x[i4]);
            let rx3 = (x[i2] - 2.0 * x[i3]).powi(2);
            let rx4 = 10f64.sqrt() * (x[i1] - x[i4]).powi(2);

            [rx1, rx2, rx3, rx4].into_iter()
        })
    }
}

impl Default for ExtendedPowell {
    fn default() -> Self {
        Self::new(4)
    }
}

impl Problem for ExtendedPowell {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        Domain::unconstrained(self.n)
    }
}

impl System for ExtendedPowell {
    fn eval<Sx, Srx>(
        &self,
        x: &Vector<Self::Field, Dyn, Sx>,
        rx: &mut Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
        Srx: StorageMut<Self::Field, Dyn>,
    {
        eval(self.residuals(x), rx)
    }

    fn norm<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        norm(self.residuals(x))
    }
}

impl TestProblem for ExtendedPowell {
    fn initials(&self) -> Vec<OVector<Self::Field, Dyn>> {
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
}

impl TestSystem for ExtendedPowell {
    fn roots(&self) -> Vec<OVector<Self::Field, Dyn>> {
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

    fn residuals<'a, Sx>(&self, x: &'a Vector<f64, Dyn, Sx>) -> impl Iterator<Item = f64> + 'a
    where
        Sx: Storage<f64, Dyn> + IsContiguous,
    {
        [
            1e4 * x[0] * x[1] - 1.0,
            (-x[0]).exp() + (-x[1]).exp() - 1.001,
        ]
        .into_iter()
    }
}

impl Default for BullardBiegler {
    fn default() -> Self {
        Self::new()
    }
}

impl Problem for BullardBiegler {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        [(5.45e-6, 4.553), (2.196e-3, 18.21)].into_iter().collect()
    }
}

impl System for BullardBiegler {
    fn eval<Sx, Srx>(
        &self,
        x: &Vector<Self::Field, Dyn, Sx>,
        rx: &mut Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
        Srx: StorageMut<Self::Field, Dyn>,
    {
        eval(self.residuals(x), rx)
    }

    fn norm<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        norm(self.residuals(x))
    }
}

impl TestProblem for BullardBiegler {
    fn initials(&self) -> Vec<OVector<Self::Field, Dyn>> {
        let init1 = dvector![0.1, 1.0];
        let init2 = dvector![1.0, 1.0];
        vec![init1, init2]
    }
}

impl TestSystem for BullardBiegler {
    fn roots(&self) -> Vec<OVector<Self::Field, Dyn>> {
        vec![dvector![1.45e-5, 6.8933353]]
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

    fn residuals<'a, Sx>(&self, x: &'a Vector<f64, Dyn, Sx>) -> impl Iterator<Item = f64> + 'a
    where
        Sx: Storage<f64, Dyn> + IsContiguous,
    {
        x.iter().map(|xi| xi.powi(2))
    }
}

impl Default for Sphere {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Problem for Sphere {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        Domain::unconstrained(self.n)
    }
}

impl System for Sphere {
    fn eval<Sx, Srx>(
        &self,
        x: &Vector<Self::Field, Dyn, Sx>,
        rx: &mut Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
        Srx: StorageMut<Self::Field, Dyn>,
    {
        eval(self.residuals(x), rx)
    }

    fn norm<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        norm(self.residuals(x))
    }
}

impl TestProblem for Sphere {
    fn initials(&self) -> Vec<OVector<Self::Field, Dyn>> {
        let init = DVector::from_iterator(
            self.n,
            (0..self.n).map(|i| if i % 2 == 0 { 10.0 } else { -10.0 }),
        );

        vec![init]
    }
}

impl TestSystem for Sphere {
    fn roots(&self) -> Vec<OVector<Self::Field, Dyn>> {
        vec![DVector::from_element(self.n, 0.0)]
    }
}

impl TestFunction for Sphere {
    fn optima(&self) -> Vec<OVector<Self::Field, Dyn>> {
        <Sphere as TestSystem>::roots(self)
    }

    fn is_optimum<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>, eps: Self::Field) -> bool
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        self.apply(x).abs() <= eps
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

    fn residuals<'a, Sx>(&self, x: &'a Vector<f64, Dyn, Sx>) -> impl Iterator<Item = f64> + 'a
    where
        Sx: Storage<f64, Dyn> + IsContiguous,
    {
        let n = self.n as f64;
        let x_sum = x.sum();

        let r0x = x.iter().product::<f64>() - 1.0;
        let rix = x
            .iter()
            .skip(1)
            .copied()
            .map(move |xi| xi + x_sum - n + 1.0);

        std::iter::once(r0x).chain(rix)
    }
}

impl Default for Brown {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Problem for Brown {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        Domain::unconstrained(self.n)
    }
}

impl System for Brown {
    fn eval<Sx, Srx>(
        &self,
        x: &Vector<Self::Field, Dyn, Sx>,
        rx: &mut Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
        Srx: StorageMut<Self::Field, Dyn>,
    {
        eval(self.residuals(x), rx)
    }

    fn norm<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        norm(self.residuals(x))
    }
}

impl TestProblem for Brown {
    fn initials(&self) -> Vec<OVector<Self::Field, Dyn>> {
        let init = DVector::zeros_generic(Dyn::from_usize(self.n), U1::name());
        vec![init]
    }
}

impl TestSystem for Brown {}

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

    fn residuals<'a, Sx>(&self, x: &'a Vector<f64, Dyn, Sx>) -> impl Iterator<Item = f64> + 'a
    where
        Sx: Storage<f64, Dyn> + IsContiguous,
    {
        let x_sum = x.sum();

        x.iter()
            .copied()
            .enumerate()
            .map(move |(i, xi)| xi - (((i + 1) as f64) * x_sum).cos().exp())
    }
}

impl Default for Exponential {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Problem for Exponential {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        Domain::unconstrained(2)
    }
}

impl System for Exponential {
    fn eval<Sx, Srx>(
        &self,
        x: &Vector<Self::Field, Dyn, Sx>,
        rx: &mut Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
        Srx: StorageMut<Self::Field, Dyn>,
    {
        eval(self.residuals(x), rx)
    }

    fn norm<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        norm(self.residuals(x))
    }
}

impl TestProblem for Exponential {
    fn initials(&self) -> Vec<OVector<Self::Field, Dyn>> {
        let init = DVector::zeros_generic(Dyn::from_usize(self.n), U1::name());
        vec![init]
    }
}

impl TestSystem for Exponential {}

/// Any value of _x_ is a solution.
#[derive(Debug, Clone, Copy)]
pub struct InfiniteSolutions {
    n: usize,
}

impl InfiniteSolutions {
    /// Initializes the system with given dimension.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "n must be greater than zero");
        Self { n }
    }
}

impl Default for InfiniteSolutions {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl Problem for InfiniteSolutions {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        Domain::unconstrained(self.n)
    }
}

impl System for InfiniteSolutions {
    fn eval<Sx, Srx>(
        &self,
        _x: &Vector<Self::Field, Dyn, Sx>,
        rx: &mut Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
        Srx: StorageMut<Self::Field, Dyn>,
    {
        rx.fill(0.0);
    }

    fn norm<Sx>(&self, _: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        0.0
    }
}

impl TestProblem for InfiniteSolutions {
    fn initials(&self) -> Vec<OVector<Self::Field, Dyn>> {
        let init = DVector::zeros_generic(Dyn::from_usize(self.n), U1::name());
        vec![init]
    }
}

impl TestSystem for InfiniteSolutions {}

/// Optimization or solving error of the testing optimizer/solver driver (see
/// [`optimize`] and [`solve`]).
#[derive(Debug, Error)]
pub enum TestingError<E: StdError + 'static> {
    /// Error of the algorithm used.
    #[error("{0}")]
    Inner(#[from] E),
    /// Algorithm did not terminate.
    #[error("algorithm did not terminate")]
    Termination,
}

/// A simple solver driver that can be used in tests.
pub fn solve<R: TestSystem, A: Solver<R>>(
    r: &R,
    dom: &Domain<R::Field>,
    mut solver: A,
    mut x: OVector<R::Field, Dyn>,
    max_iters: usize,
    tolerance: R::Field,
) -> Result<OVector<R::Field, Dyn>, TestingError<A::Error>>
where
    A::Error: StdError,
{
    let mut rx = x.clone_owned();
    let mut iter = 0;

    loop {
        solver.solve_next(r, dom, &mut x, &mut rx)?;

        if rx.norm() <= tolerance {
            return Ok(x);
        }

        if iter == max_iters {
            return Err(TestingError::Termination);
        } else {
            iter += 1;
        }
    }
}

/// A simple optimization driver that can be used in tests.
pub fn optimize<F: Function, O: Optimizer<F>>(
    f: &F,
    dom: &Domain<F::Field>,
    mut optimizer: O,
    mut x: OVector<F::Field, Dyn>,
    min: F::Field,
    max_iters: usize,
    tolerance: F::Field,
) -> Result<OVector<F::Field, Dyn>, TestingError<O::Error>>
where
    O::Error: StdError,
{
    let mut iter = 0;

    loop {
        let fx = optimizer.opt_next(f, dom, &mut x)?;

        if fx <= min + tolerance {
            // Converged.
            return Ok(x);
        }

        if iter == max_iters {
            return Err(TestingError::Termination);
        } else {
            iter += 1;
        }
    }
}

/// Iterates the solver and inspects it in each iteration.
///
/// This is useful for testing evolutionary/nature-inspired algorithms.
pub fn iter<R: TestSystem, A: Solver<R>, G>(
    r: &R,
    dom: &Domain<R::Field>,
    mut solver: A,
    mut x: OVector<R::Field, Dyn>,
    iters: usize,
    mut inspect: G,
) -> Result<(), A::Error>
where
    A::Error: StdError,
    G: FnMut(&A, &OVector<R::Field, Dyn>, R::Field, usize),
{
    let mut rx = x.clone_owned();

    for iter in 0..iters {
        solver.solve_next(r, dom, &mut x, &mut rx)?;
        inspect(&solver, &x, rx.norm(), iter);
    }

    Ok(())
}

fn eval<Srx>(residuals: impl Iterator<Item = f64>, rx: &mut Vector<f64, Dyn, Srx>)
where
    Srx: StorageMut<f64, Dyn>,
{
    rx.iter_mut()
        .zip(residuals)
        .for_each(|(rix, res)| *rix = res);
}

fn norm(residuals: impl Iterator<Item = f64>) -> f64 {
    residuals.map(|res| res.powi(2)).sum::<f64>().sqrt()
}
