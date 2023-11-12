use nalgebra::{
    allocator::Allocator, storage::Storage, storage::StorageMut, DefaultAllocator, Dynamic,
    IsContiguous, Vector,
};
use num_traits::Zero;

use super::{
    base::{Problem, ProblemError},
    system::System,
};

/// The trait for defining functions.
///
/// ## Defining a function
///
/// A function is any type that implements [`Function`] and [`Problem`] traits.
/// There is one required associated type (scalar) and one required method
/// ([`apply`](Function::apply)).
///
/// ```rust
/// use gomez::core::Function;
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
/// impl Function for Rosenbrock {
///     // Apply trial values of variables to the function.
///     fn apply<Sx>(
///         &self,
///         x: &na::Vector<Self::Scalar, Dynamic, Sx>,
///     ) -> Result<Self::Scalar, ProblemError>
///     where
///         Sx: na::storage::Storage<Self::Scalar, Dynamic> + IsContiguous,
///     {
///         // Compute the function value.
///         Ok((self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2))
///     }
/// }
/// ```
pub trait Function: Problem {
    /// Calculate the function value given values of the variables.
    fn apply<Sx>(
        &self,
        x: &Vector<Self::Scalar, Dynamic, Sx>,
    ) -> Result<Self::Scalar, ProblemError>
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous;

    /// Calculate the norm of residuals of the system given values of the
    /// variable for cases when the function is actually a system of equations.
    ///
    /// The optimizers should prefer calling this function because the
    /// implementation for systems reuse `fx` for calculating the residuals and
    /// do not make an unnecessary allocation for it.
    fn apply_eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Dynamic, Sx>,
        fx: &mut Vector<Self::Scalar, Dynamic, Sfx>,
    ) -> Result<Self::Scalar, ProblemError>
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Dynamic>,
    {
        let norm = self.apply(x)?;
        fx.fill(Self::Scalar::zero());
        fx[0] = norm;
        Ok(norm)
    }
}

impl<F: System> Function for F
where
    DefaultAllocator: Allocator<F::Scalar, Dynamic>,
{
    fn apply<Sx>(&self, x: &Vector<Self::Scalar, Dynamic, Sx>) -> Result<Self::Scalar, ProblemError>
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous,
    {
        let mut fx = x.clone_owned();
        self.apply_eval(x, &mut fx)
    }

    fn apply_eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Scalar, Dynamic, Sx>,
        fx: &mut Vector<Self::Scalar, Dynamic, Sfx>,
    ) -> Result<Self::Scalar, ProblemError>
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous,
        Sfx: StorageMut<Self::Scalar, Dynamic>,
    {
        self.eval(x, fx)?;
        let norm = fx.norm();
        Ok(norm)
    }
}

/// Extension trait for `Result<F::Scalar, Error>`.
pub trait FunctionResultExt<T> {
    /// If the result is [`ProblemError::InvalidValue`], `Ok(default)` is
    /// returned instead. The original result is returned otherwise.
    fn ignore_invalid_value(self, replace_with: T) -> Self;
}

impl<T> FunctionResultExt<T> for Result<T, ProblemError> {
    fn ignore_invalid_value(self, replace_with: T) -> Self {
        match self {
            Ok(value) => Ok(value),
            Err(ProblemError::InvalidValue) => Ok(replace_with),
            Err(error) => Err(error),
        }
    }
}
