use nalgebra::{
    allocator::Allocator, storage::Storage, storage::StorageMut, DefaultAllocator, Vector,
};
use num_traits::Zero;

use super::{
    base::{Error, Problem},
    system::System,
};

/// The trait for defining functions.
///
/// ## Defining a function
///
/// A function is any type that implements [`Function`] and [`Problem`] traits.
/// There are two required associated types (scalar type and dimension type) and
/// two required methods: [`apply`](Function::apply) and [`dim`](Problem::dim).
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
/// impl Function for Rosenbrock {
///     // Apply trial values of variables to the function.
///     fn apply<Sx>(
///         &self,
///         x: &na::Vector<Self::Scalar, Self::Dim, Sx>,
///     ) -> Result<Self::Scalar, Error>
///     where
///         Sx: na::storage::Storage<Self::Scalar, Self::Dim>,
///     {
///         // Compute the function value.
///         Ok((self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2))
///     }
/// }
/// ```
pub trait Function: Problem {
    /// Calculate the function value given values of the variables.
    fn apply<Sx>(&self, x: &Vector<Self::Scalar, Self::Dim, Sx>) -> Result<Self::Scalar, Error>
    where
        Sx: Storage<Self::Scalar, Self::Dim>;

    /// Calculate the norm of residuals of the system given values of the
    /// variable for cases when the function is actually a system of equations.
    ///
    /// The optimizers should prefer calling this function because the
    /// implementation for systems reuse `fx` for calculating the residuals and
    /// do not make an unnecessary allocation for it.
    fn apply_eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<Self::Scalar, Error>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        let norm = self.apply(x)?;
        fx.fill(Self::Scalar::zero());
        fx[0] = norm;
        Ok(norm)
    }
}

impl<F: System> Function for F
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    fn apply<Sx>(&self, x: &Vector<Self::Scalar, Self::Dim, Sx>) -> Result<Self::Scalar, Error>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
    {
        let mut fx = x.clone_owned();
        self.apply_eval(x, &mut fx)
    }

    fn apply_eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, Sx>,
        fx: &mut Vector<Self::Scalar, Self::Dim, Sfx>,
    ) -> Result<Self::Scalar, Error>
    where
        Sx: Storage<Self::Scalar, Self::Dim>,
        Sfx: StorageMut<Self::Scalar, Self::Dim>,
    {
        self.eval(x, fx)?;
        let norm = fx.norm();
        Ok(norm)
    }
}
