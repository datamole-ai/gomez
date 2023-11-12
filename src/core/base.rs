use nalgebra::RealField;
use thiserror::Error;

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

/// Error encountered while applying variables to the problem.
#[derive(Debug, Error)]
pub enum ProblemError {
    /// The number of variables does not match the dimensionality of the problem
    /// domain.
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
