use nalgebra::{
    storage::{Storage, StorageMut},
    Dyn, IsContiguous, Vector,
};

use super::base::Problem;

/// Definition of a system of equations.
///
/// ## Defining a system
///
/// A system is any type that implements [`System`] and [`Problem`] traits.
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::{Domain, Problem, System};
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
/// impl System for Rosenbrock {
///     fn eval<Sx, Srx>(
///         &self,
///         x: &na::Vector<Self::Field, Dyn, Sx>,
///         rx: &mut na::Vector<Self::Field, Dyn, Srx>,
///     ) where
///         Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
///         Srx: na::storage::StorageMut<Self::Field, Dyn>,
///     {
///         // Compute the residuals of all equations.
///         rx[0] = (self.a - x[0]).powi(2);
///         rx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
///     }
/// }
/// ```
pub trait System: Problem {
    /// Calculates the system residuals in given point.
    fn eval<Sx, Srx>(
        &self,
        x: &Vector<Self::Field, Dyn, Sx>,
        rx: &mut Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
        Srx: StorageMut<Self::Field, Dyn>;

    /// Calculates the system residuals vector norm.
    ///
    /// The default implementation allocates a temporary vector for the
    /// residuals on every call. If you plan to solve the system by an
    /// optimizer, consider overriding the default implementation to avoid this
    /// allocation.
    fn norm<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dyn> + IsContiguous,
    {
        let mut rx = x.clone_owned();
        self.eval(x, &mut rx);
        rx.norm()
    }
}
