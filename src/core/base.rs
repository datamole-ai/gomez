use nalgebra::RealField;

use super::domain::Domain;

/// The base trait for [`System`](super::system::System) and
/// [`Function`](super::function::Function).
pub trait Problem {
    /// Type of the scalar, usually f32 or f64.
    type Scalar: RealField + Copy;

    /// Get the domain (bound constraints) of the system. If not overridden, the
    /// system is unconstrained.
    fn domain(&self) -> Domain<Self::Scalar>;
}
