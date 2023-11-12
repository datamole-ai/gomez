use nalgebra::{
    allocator::Allocator,
    storage::{Storage, StorageMut},
    DefaultAllocator, Dynamic, IsContiguous, OVector, Vector,
};

use super::{
    base::{Problem, ProblemError},
    domain::Domain,
};

/// The trait for defining equations systems.
///
/// ## Defining a system
///
/// A system is any type that implements [`System`] and [`Problem`] traits.
/// There is one required associated type (scalar type) and one required method
/// ([`eval`](System::eval)).
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::prelude::*;
/// use na::{Dynamic, IsContiguous};
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
///
///     // The domain of the problem (could be bound-constrained).
///     fn domain(&self) -> Domain<Self::Scalar> {
///         Domain::unconstrained(2)
///     }
/// }
///
/// impl System for Rosenbrock {
///     // Evaluate trial values of variables to the system.
///     fn eval<Sx, Sfx>(
///         &self,
///         x: &na::Vector<Self::Scalar, Dynamic, Sx>,
///         fx: &mut na::Vector<Self::Scalar, Dynamic, Sfx>,
///     ) -> Result<(), ProblemError>
///     where
///         Sx: na::storage::Storage<Self::Scalar, Dynamic> + IsContiguous,
///         Sfx: na::storage::StorageMut<Self::Scalar, Dynamic>,
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
        x: &Vector<Self::Scalar, Dynamic, Sx>,
        fx: &mut Vector<Self::Scalar, Dynamic, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Dynamic>;

    /// Calculate the residuals vector norm.
    ///
    /// The default implementation allocates a temporary vector for the
    /// residuals on every call. If you plan to solve the system by an
    /// optimizer, consider overriding the default implementation.
    fn norm<Sx>(&self, x: &Vector<Self::Scalar, Dynamic, Sx>) -> Result<Self::Scalar, ProblemError>
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous,
    {
        let mut fx = x.clone_owned();
        self.eval(x, &mut fx)?;
        Ok(fx.norm())
    }
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
pub struct RepulsiveSystem<'f, F: System> {
    f: &'f F,
    archive: Vec<OVector<F::Scalar, Dynamic>>,
}

impl<'f, F: System> RepulsiveSystem<'f, F> {
    /// Initializes the repulsive system by wrapping the original system.
    pub fn new(f: &'f F) -> Self {
        Self {
            f,
            archive: Vec::new(),
        }
    }

    /// Add a found solution to the archive.
    pub fn push(&mut self, root: OVector<F::Scalar, Dynamic>) {
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
    pub fn unpack(self) -> Vec<OVector<F::Scalar, Dynamic>> {
        self.archive
    }
}

impl<'f, F: System> Problem for RepulsiveSystem<'f, F> {
    type Scalar = F::Scalar;

    fn domain(&self) -> Domain<Self::Scalar> {
        self.f.domain()
    }
}

impl<'f, F: System> System for RepulsiveSystem<'f, F>
where
    DefaultAllocator: Allocator<F::Scalar, Dynamic>,
{
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Dynamic, Sx>,
        fx: &mut Vector<Self::Scalar, Dynamic, Sfx>,
    ) -> Result<(), ProblemError>
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Dynamic>,
    {
        // TODO: RepulsiveSystem should adjust the residuals of the inner system
        // such that solvers tend to go away from the roots stored in the
        // archive.
        self.f.eval(x, fx)
    }
}
