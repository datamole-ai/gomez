use nalgebra::{storage::Storage, Dynamic, IsContiguous, Vector};

use super::{base::Problem, system::System};

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
///     fn apply<Sx>(&self, x: &na::Vector<Self::Scalar, Dynamic, Sx>) -> Self::Scalar
///     where
///         Sx: na::storage::Storage<Self::Scalar, Dynamic> + IsContiguous,
///     {
///         // Compute the function value.
///         (self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2)
///     }
/// }
/// ```
pub trait Function: Problem {
    /// Calculate the function value given values of the variables.
    fn apply<Sx>(&self, x: &Vector<Self::Scalar, Dynamic, Sx>) -> Self::Scalar
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous;
}

impl<F> Function for F
where
    F: System,
{
    fn apply<Sx>(&self, x: &Vector<Self::Scalar, Dynamic, Sx>) -> Self::Scalar
    where
        Sx: Storage<Self::Scalar, Dynamic> + IsContiguous,
    {
        self.norm(x)
    }
}
