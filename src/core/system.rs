//! Abstractions and types for defining equation systems.
//!
//! # Defining a system
//!
//! A system is any type that implements [`System`] trait. There are two
//! required associated types (scalar type and dimension type) and two required
//! methods: [`apply_mut`](System::apply_mut) and [`dim`](System::dim).
//!
//! ```rust
//! use gomez::nalgebra as na;
//! use gomez::prelude::*;
//! use na::{Dim, DimName};
//!
//! // A problem is represented by a type.
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! impl System for Rosenbrock {
//!     // The numeric type. Usually f64 or f32.
//!     type Scalar = f64;
//!     // The dimension of the problem. Can be either statically known or dynamic.
//!     type Dim = na::U2;
//!
//!     // Apply trial values of variables to the system.
//!     fn apply_mut<Sx, Sfx>(
//!         &self,
//!         x: &na::Vector<Self::Scalar, Self::Dim, Sx>,
//!         fx: &mut na::Vector<Self::Scalar, Self::Dim, Sfx>,
//!     ) -> Result<(), SystemError>
//!     where
//!         Sx: na::storage::Storage<Self::Scalar, Self::Dim>,
//!         Sfx: na::storage::StorageMut<Self::Scalar, Self::Dim>,
//!     {
//!         // Compute the residuals of all equations.
//!         fx[0] = (self.a - x[0]).powi(2);
//!         fx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
//!
//!         Ok(())
//!     }
//!
//!     // Return the actual dimension of the system.
//!     fn dim(&self) -> Self::Dim {
//!         na::U2::name()
//!     }
//! }
//! ```

use nalgebra::{
    allocator::Allocator,
    storage::{Storage, StorageMut},
    DefaultAllocator, Dim, OVector, RealField, Vector,
};
use thiserror::Error;

use super::domain::Domain;

/// Error encountered while applying variables to the system.
#[derive(Debug, Error)]
pub enum SystemError {
    /// The number of variables does not match the dimensionality
    /// ([`System::dim`]) of the system.
    #[error("invalid dimensionality")]
    InvalidDimensionality,
    /// An invalid value (NaN, positive or negative infinity) of a residual
    /// occurred.
    #[error("invalid value encountered")]
    InvalidValue,
    /// A custom error specific to the system.
    #[error("{0}")]
    Custom(Box<dyn std::error::Error>),
}

/// The trait for defining equations systems.
pub trait System {
    /// Type of the scalar, usually f32 or f64.
    type Scalar: RealField;

    /// Dimension of the system. Can be fixed
    /// ([`Const`](nalgebra::base::dimension::Const)) or dynamic
    /// ([`Dynamic`](nalgebra::base::dimension::Dynamic)).
    type Dim: Dim;

    /// Calculate the residuals of the system given values of the variables.
    fn apply_mut<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), SystemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
        Sfx: StorageMut<Self::Scalar, Self::Dim>;

    /// Return the actual dimension of the system. This is needed for dynamic
    /// systems.
    fn dim(&self) -> Self::Dim;

    /// Get the domain (bound constraints) of the system. If not overridden, the
    /// system is unconstrained.
    fn domain(&self) -> Domain<Self::Scalar> {
        Domain::with_dim(self.dim().value())
    }
}

/// Some extensions methods for the [`System`] that may be found useful.
pub trait SystemExt: System {
    /// Calculate the residuals and return the squared norm of the residuals.
    fn apply_mut_norm_squared<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<Self::Scalar, SystemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
        Sfx: StorageMut<Self::Scalar, Self::Dim>;
}

impl<F: System> SystemExt for F {
    fn apply_mut_norm_squared<Sx, Sfx>(
        &self,
        x: &Vector<F::Scalar, F::Dim, Sx>,
        fx: &mut Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<F::Scalar, SystemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        self.apply_mut(x, fx)?;
        Ok(fx.norm_squared())
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

impl<'f, F: System> System for RepulsiveSystem<'f, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Scalar = F::Scalar;
    type Dim = F::Dim;

    fn apply_mut<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<(), SystemError>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        // TODO: RepulsiveSystem should adjust the residuals of the inner system
        // such that solvers tend to go away from the roots stored in the
        // archive.
        self.f.apply_mut(x, fx)
    }

    fn dim(&self) -> Self::Dim {
        self.f.dim()
    }

    fn domain(&self) -> Domain<Self::Scalar> {
        self.f.domain()
    }
}
