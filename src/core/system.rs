use nalgebra::{
    allocator::Allocator,
    storage::{Storage, StorageMut},
    DefaultAllocator, OVector, Vector,
};

use super::{
    base::{Error, Problem},
    domain::Domain,
};

/// The trait for defining equations systems.
///
/// ## Defining a system
///
/// A system is any type that implements [`System`] and [`Problem`] traits.
/// There are two required associated types (scalar type and dimension type) and
/// two required methods: [`eval`](System::eval) and [`dim`](Problem::dim).
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::prelude::*;
/// use na::{Dim, DimName};
///
/// // A problem is represented by a type.
/// struct Rosenbrock {
///     a: f64,
///     b: f64,
/// }
///
/// impl Problem for Rosenbrock {
///     // The numeric type. Usually f64 or f32.
///     type Scalar = f64;
///     // The dimension of the problem. Can be either statically known or dynamic.
///     type Dim = na::U2;
///
///     // Return the actual dimension of the system.
///     fn dim(&self) -> Self::Dim {
///         na::U2::name()
///     }
/// }
///
/// impl System for Rosenbrock {
///     // Evaluate trial values of variables to the system.
///     fn eval<Sx, Sfx>(
///         &self,
///         x: &na::Vector<Self::Scalar, Self::Dim, Sx>,
///         fx: &mut na::Vector<Self::Scalar, Self::Dim, Sfx>,
///     ) -> Result<(), Error>
///     where
///         Sx: na::storage::Storage<Self::Scalar, Self::Dim>,
///         Sfx: na::storage::StorageMut<Self::Scalar, Self::Dim>,
///     {
///         // Compute the residuals of all equations.
///         fx[0] = (self.a - x[0]).powi(2);
///         fx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
///
///         Ok(())
///     }
/// }
/// ```
pub trait System: Problem {
    /// Calculate the residuals of the system given values of the variables.
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), Error>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
        Sfx: StorageMut<Self::Scalar, Self::Dim>;
}

/// A wrapper type for systems that implements a standard mechanism for
/// repulsing solvers from solutions that have been already found and stored in
/// the archive.
///
/// **WARNING:** This is currently noop as the repulsion mechanism has not been
/// determined yet. But the technique is mentioned in [A Decomposition-based
/// Differential Evolution with Reinitialization for Nonlinear Equations
/// Systems](https://www.sciencedirect.com/science/article/abs/pii/S0950705119305933)
/// or [Testing Nelder-Mead Based Repulsion Algorithms for Multiple Roots of
/// Nonlinear Systems via a Two-Level Factorial Design of
/// Experiments](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0121844),
/// for example.
pub struct RepulsiveSystem<'f, F: System>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    f: &'f F,
    archive: Vec<OVector<F::Scalar, F::Dim>>,
}

impl<'f, F: System> RepulsiveSystem<'f, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Initializes the repulsive system by wrapping the original system.
    pub fn new(f: &'f F) -> Self {
        Self {
            f,
            archive: Vec::new(),
        }
    }

    /// Add a found solution to the archive.
    pub fn push(&mut self, root: OVector<F::Scalar, F::Dim>) {
        self.archive.push(root);
    }

    /// Get the size of the archive.
    pub fn len(&self) -> usize {
        self.archive.len()
    }

    /// Determine whether the archive is empty.
    pub fn is_empty(&self) -> bool {
        self.archive.is_empty()
    }

    /// Unpack the archive which contains all solutions found.
    pub fn unpack(self) -> Vec<OVector<F::Scalar, F::Dim>> {
        self.archive
    }
}

impl<'f, F: System> Problem for RepulsiveSystem<'f, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Scalar = F::Scalar;
    type Dim = F::Dim;

    fn dim(&self) -> Self::Dim {
        self.f.dim()
    }

    fn domain(&self) -> Domain<Self::Scalar> {
        self.f.domain()
    }
}

impl<'f, F: System> System for RepulsiveSystem<'f, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), Error>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        // TODO: RepulsiveSystem should adjust the residuals of the inner system
        // such that solvers tend to go away from the roots stored in the
        // archive.
        self.f.eval(x, fx)
    }
}
