use nalgebra::{storage::Storage, Dyn, IsContiguous, Vector};

use super::{base::Problem, system::System};

/// Definition of a function.
///
/// ## Defining a function
///
/// A function is any type that implements [`Function`] and [`Problem`] traits.
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::{Domain, Function, Problem};
/// use na::{Dyn, IsContiguous};
///
/// struct Rosenbrock {
///     a: f64,
///     b: f64,
/// }
///
/// impl Problem for Rosenbrock {
///     type Field = f64;
///
///     fn domain(&self) -> Domain<Self::Field> {
///         Domain::unconstrained(2)
///     }
/// }
///
/// impl Function for Rosenbrock {
///     fn apply<Sx>(&self, x: &na::Vector<Self::Field, Dyn, Sx>) -> Self::Field
///     where
///         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
///     {
///         // Compute the function value.
///         (self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2)
///     }
/// }
/// ```
pub trait Function: Problem {
    /// Calculates the function value in given point.
    fn apply<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous;
}

impl<F> Function for F
where
    F: System,
{
    fn apply<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        self.norm(x)
    }
}
