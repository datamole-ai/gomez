use fastrand::Rng;
use fastrand_contrib::RngExt;

use super::domain::Domain;

/// Trait implemented by real numbers.
pub trait RealField: nalgebra::RealField {
    /// Square root of machine epsilon.
    ///
    /// This value is a standard constant for epsilons in approximating
    /// first-order derivate-based concepts.
    const EPSILON_SQRT: Self;

    /// Cubic root of machine epsilon.
    ///
    /// This value is a standard constant for epsilons in approximating
    /// second-order derivate-based concepts.
    const EPSILON_CBRT: Self;
}

impl RealField for f32 {
    const EPSILON_SQRT: Self = 0.00034526698;
    const EPSILON_CBRT: Self = 0.0049215667;
}

impl RealField for f64 {
    const EPSILON_SQRT: Self = 0.000000014901161193847656;
    const EPSILON_CBRT: Self = 0.0000060554544523933395;
}

/// The base trait for [`Function`](super::function::Function) and
/// [`System`](super::system::System).
pub trait Problem {
    /// Field type, f32 or f64.
    type Field: RealField + Copy;

    /// Domain of the problem.
    fn domain(&self) -> Domain<Self::Field>;
}

/// Trait for types that can be sampled.
pub trait Sample {
    /// Sample value from the whole range of the type.
    fn sample_any(rng: &mut Rng) -> Self;

    /// Sample from uniform distribution (inclusive on both sides).
    fn sample_uniform(lower: Self, upper: Self, rng: &mut Rng) -> Self;

    /// Sample from normal distribution.
    fn sample_normal(mu: Self, sigma: Self, rng: &mut Rng) -> Self;
}

impl Sample for f32 {
    fn sample_any(rng: &mut Rng) -> Self {
        // Sampling from the whole range is likely not desired. Choosing
        // sqrt(MAX) as an arbitrary bound.
        let max = f32::MAX.sqrt();
        let min = -max;
        rng.f32_range(min..=max)
    }

    fn sample_uniform(lower: Self, upper: Self, rng: &mut Rng) -> Self {
        rng.f32_range(lower..=upper)
    }

    fn sample_normal(mu: Self, sigma: Self, rng: &mut Rng) -> Self {
        rng.f32_normal_approx(mu, sigma)
    }
}

impl Sample for f64 {
    fn sample_any(rng: &mut Rng) -> Self {
        // Sampling from the whole range is likely not desired. Choosing
        // cbrt(MAX) as an arbitrary bound.
        let max = f64::MAX.cbrt();
        let min = -max;
        rng.f64_range(min..=max)
    }

    fn sample_uniform(lower: Self, upper: Self, rng: &mut Rng) -> Self {
        rng.f64_range(lower..=upper)
    }

    fn sample_normal(mu: Self, sigma: Self, rng: &mut Rng) -> Self {
        rng.f64_normal_approx(mu, sigma)
    }
}
