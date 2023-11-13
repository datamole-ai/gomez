use nalgebra::{
    storage::{Storage, StorageMut},
    Dynamic, IsContiguous, Vector,
};

use super::base::Problem;

/// The trait for defining equations systems.
///
/// ## Defining a system
///
/// A system is any type that implements [`System`] and [`Problem`] traits.
/// There is one required associated type (field type) and one required method
/// ([`eval`](System::eval)).
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::{Domain, Problem, System};
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
/// impl System for Rosenbrock {
///     // Evaluate trial values of variables to the system.
///     fn eval<Sx, Sfx>(
///         &self,
///         x: &na::Vector<Self::Field, Dynamic, Sx>,
///         fx: &mut na::Vector<Self::Field, Dynamic, Sfx>,
///     ) where
///         Sx: na::storage::Storage<Self::Field, Dynamic> + IsContiguous,
///         Sfx: na::storage::StorageMut<Self::Field, Dynamic>,
///     {
///         // Compute the residuals of all equations.
///         fx[0] = (self.a - x[0]).powi(2);
///         fx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
///     }
/// }
/// ```
pub trait System: Problem {
    /// Calculate the residuals of the system given values of the variables.
    fn eval<Sx, Sfx>(
        &self,
        x: &Vector<Self::Field, Dynamic, Sx>,
        fx: &mut Vector<Self::Field, Dynamic, Sfx>,
    ) where
        Sx: Storage<Self::Field, Dynamic> + IsContiguous,
        Sfx: StorageMut<Self::Field, Dynamic>;

    /// Calculate the residuals vector norm.
    ///
    /// The default implementation allocates a temporary vector for the
    /// residuals on every call. If you plan to solve the system by an
    /// optimizer, consider overriding the default implementation.
    fn norm<Sx>(&self, x: &Vector<Self::Field, Dynamic, Sx>) -> Self::Field
    where
        Sx: Storage<Self::Field, Dynamic> + IsContiguous,
    {
        let mut fx = x.clone_owned();
        self.eval(x, &mut fx);
        fx.norm()
    }
}
