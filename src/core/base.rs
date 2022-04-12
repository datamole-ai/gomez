use nalgebra::{Dim, RealField};
use thiserror::Error;

use super::domain::Domain;

/// The base trait for [`System`](super::system::System) and
/// [`Function`](super::function::Function).
pub trait Problem {
    /// Type of the scalar, usually f32 or f64.
    type Scalar: RealField + Copy;

    /// Dimension of the system. Can be fixed
    /// ([`Const`](nalgebra::base::dimension::Const)) or dynamic
    /// ([`Dynamic`](nalgebra::base::dimension::Dynamic)).
    type Dim: Dim;

    /// Return the actual dimension of the system. This is needed for dynamic
    /// systems.
    fn dim(&self) -> Self::Dim;

    /// Get the domain (bound constraints) of the system. If not overridden, the
    /// system is unconstrained.
    fn domain(&self) -> Domain<Self::Scalar> {
        Domain::with_dim(self.dim().value())
    }
}

/// Error encountered while applying variables to the function.
#[derive(Debug, Error)]
pub enum Error {
    /// The number of variables does not match the dimensionality
    /// ([`Problem::dim`]) of the problem.
    #[error("invalid dimensionality")]
    InvalidDimensionality,
    /// An invalid value (NaN, positive or negative infinity) of a residual or
    /// the function value occurred.
    #[error("invalid value encountered")]
    InvalidValue,
    /// A custom error specific to the system or function.
    #[error("{0}")]
    Custom(Box<dyn std::error::Error>),
}
