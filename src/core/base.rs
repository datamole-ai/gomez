use super::domain::Domain;

/// Trait implemented by real numbers.
pub trait RealField: nalgebra::RealField {
    /// Square root of double precision machine epsilon. This value is a
    /// standard constant for epsilons in approximating first-order
    /// derivate-based concepts.
    const EPSILON_SQRT: Self;

    /// Cubic root of double precision machine epsilon. This value is a standard
    /// constant for epsilons in approximating second-order derivate-based
    /// concepts.
    const EPSILON_CBRT: Self;
}

/// The base trait for [`System`](super::system::System) and
/// [`Function`](super::function::Function).
pub trait Problem {
    /// Type of the field, usually f32 or f64.
    type Field: RealField + Copy;

    /// Get the domain (bound constraints) of the system. If not overridden, the
    /// system is unconstrained.
    fn domain(&self) -> Domain<Self::Field>;
}

impl RealField for f32 {
    const EPSILON_SQRT: Self = 0.00034526698;
    const EPSILON_CBRT: Self = 0.0049215667;
}

impl RealField for f64 {
    const EPSILON_SQRT: Self = 0.000000014901161193847656;
    const EPSILON_CBRT: Self = 0.0000060554544523933395;
}
