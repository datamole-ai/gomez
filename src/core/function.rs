use nalgebra::{storage::Storage, Dynamic, IsContiguous, Vector};

use super::{base::Problem, system::System};

/// The trait for defining functions.
///
/// ## Defining a function
///
/// A function is any type that implements [`Function`] and [`Problem`] traits.
/// There is one required associated type (field) and one required method
/// ([`apply`](Function::apply)).
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::{Domain, Function, Problem};
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
///     type Field = f64;
///
///     // The domain of the problem (could be bound-constrained).
///     fn domain(&self) -> Domain<Self::Field> {
///         Domain::unconstrained(2)
///     }
/// }
///
/// impl Function for Rosenbrock {
///     // Apply trial values of variables to the function.
///     fn apply<Sx>(&self, x: &na::Vector<Self::Field, Dynamic, Sx>) -> Self::Field
///     where
///         Sx: na::storage::Storage<Self::Field, Dynamic> + IsContiguous,
///     {
///         // Compute the function value.
///         (self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2)
///     }
/// }
/// ```
pub trait Function: Problem {
    /// Calculate the function value given values of the variables.
    fn apply<Sx>(&self, x: &Vector<Self::Field, Dynamic, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dynamic> + IsContiguous;
}

impl<F> Function for F
where
    F: System,
{
    fn apply<Sx>(&self, x: &Vector<Self::Field, Dynamic, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dynamic> + IsContiguous,
    {
        self.norm(x)
    }
}
