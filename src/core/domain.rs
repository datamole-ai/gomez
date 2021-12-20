//! Problem domain definition such as bound constraints for variables.

use std::iter::FromIterator;

use nalgebra::{storage::StorageMut, Dim, RealField, Vector};

/// [`Variable`] builder type.
#[derive(Debug, Clone, Copy)]
pub struct VariableBuilder<T: RealField>(Variable<T>);

impl<T: RealField> VariableBuilder<T> {
    fn new() -> Self {
        Self(Variable::new())
    }

    /// Set the variable bounds. See
    /// [`Variable::set_bounds`](Variable::set_bounds) for details.
    pub fn bounds(mut self, lower: T, upper: T) -> Self {
        self.0.set_bounds(lower, upper);
        self
    }

    /// Set the variable magnitude. See
    /// [`Variable::set_magnitude`](Variable::set_magnitude) for details.
    pub fn magnitude(mut self, magnitude: T) -> Self {
        self.0.set_magnitude(magnitude);
        self
    }

    /// Finalize the variable construction.
    pub fn finalize(self) -> Variable<T> {
        self.0
    }
}

/// Variable definition.
///
/// There are two pieces of information about each variable:
///
/// * [`Bounds`](Variable::set_bounds) are used as constraints during solving
/// * [`Magnitude`](Variable::set_magnitude) is used to compensate scaling
///   discrepancies between different variables.
#[derive(Debug, Clone, Copy)]
pub struct Variable<T: RealField> {
    bounds: (T, T),
    magnitude: T,
}

impl<T: RealField> Variable<T> {
    /// Creates new unconstrained variable with magnitude 1.
    pub fn new() -> Self {
        let inf = T::from_subset(&f64::INFINITY);

        Self {
            bounds: (-inf, inf),
            magnitude: T::one(),
        }
    }

    /// Returns variable builder which allows to add constraints and magnitude.
    pub fn builder() -> VariableBuilder<T> {
        VariableBuilder::new()
    }

    /// Set the variable bounds.
    ///
    /// If both bounds are finite value, the variable magnitude is automatically
    /// estimated by an internal heuristic.
    ///
    /// # Panics
    ///
    /// Panics if `lower > upper`.
    pub fn set_bounds(&mut self, lower: T, upper: T) -> &mut Self {
        assert!(lower <= upper, "invalid bounds");

        if lower.is_finite() && upper.is_finite() {
            self.magnitude = estimate_magnitude(lower, upper);
        }

        self.bounds = (lower, upper);
        self
    }

    /// Set the variable magnitude.
    ///
    /// # Panics
    ///
    /// Panics if `magnitude <= 0`.
    pub fn set_magnitude(&mut self, magnitude: T) -> &mut Self {
        assert!(magnitude > T::zero(), "magnitude must be positive");
        self.magnitude = magnitude;
        self
    }

    /// Get the lower bound.
    pub fn lower(&self) -> T {
        self.bounds.0
    }

    /// Get the upper bound.
    pub fn upper(&self) -> T {
        self.bounds.1
    }

    /// Get the magnitude.
    pub fn magnitude(&self) -> T {
        self.magnitude
    }

    /// Get the scale which is inverse of the magnitude.
    pub fn scale(&self) -> T {
        T::one() / self.magnitude
    }

    /// Check if a value is within the bounds.
    pub fn is_within(&self, value: &T) -> bool {
        value >= &self.lower() && value <= &self.upper()
    }

    /// Return value that is clamped to be in bounds.
    pub fn clamp(&self, value: T) -> T {
        if value < self.lower() {
            self.lower()
        } else if value > self.upper() {
            self.upper()
        } else {
            value
        }
    }
}

impl<T: RealField> Default for Variable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField> From<VariableBuilder<T>> for Variable<T> {
    fn from(def: VariableBuilder<T>) -> Self {
        def.finalize()
    }
}

fn estimate_magnitude<T: RealField>(lower: T, upper: T) -> T {
    let ten = T::from_subset(&10.0);
    let half = T::from_subset(&0.5);

    let avg = half * (lower.abs() + upper.abs());
    let magnitude = ten.powf(avg.abs().log10().trunc());

    // For [0, 0] range, the computed magnitude is undefined. We allow such
    // ranges to support fixing a variable to a value with existing API.
    if magnitude.is_finite() && magnitude > T::zero() {
        magnitude
    } else {
        T::one()
    }
}

/// A convenience macro over [`VariableBuilder`].
///
/// # Examples
///
/// ```rust
/// use gomez::prelude::*;
///
/// let x = var!(-10.0f64, 10.0);
/// assert_eq!(x.lower(), -10.0);
/// assert_eq!(x.upper(), 10.0);
///
/// let y = var!(5.0f64);
/// assert_eq!(y.magnitude(), 5.0);
///
/// let z = var!(-10.0f64, 10.0; 5.0);
/// assert_eq!(z.lower(), -10.0);
/// assert_eq!(z.upper(), 10.0);
/// assert_eq!(z.magnitude(), 5.0);
/// ```
#[macro_export]
macro_rules! var {
    ($lower:expr, $upper:expr; $magnitude:expr) => {
        $crate::core::Variable::builder()
            .bounds($lower, $upper)
            .magnitude($magnitude)
            .finalize()
    };
    ($lower:expr, $upper:expr) => {
        $crate::core::Variable::builder()
            .bounds($lower, $upper)
            .finalize()
    };
    ($magnitude:expr) => {
        $crate::core::Variable::builder()
            .magnitude($magnitude)
            .finalize()
    };
}

/// A set of [`Variable`] definitions.
// TODO: Add generic type for nalgebra dimension?
pub struct Domain<T: RealField> {
    vars: Vec<Variable<T>>,
}

impl<T: RealField> Domain<T> {
    /// Creates unconstrained domain with given dimension.
    pub fn with_dim(n: usize) -> Self {
        (0..n).map(|_| Variable::default()).collect()
    }

    /// Creates the domain from variable definitions.
    ///
    /// This should be used for constrained or with known magnitude variables.
    /// For unconstrained domains, use [`Domain::with_dim`] instead. Note that
    /// it is possible to create the domain from iterator over type [`Variable`]
    /// by calling [`collect`](Iterator::collect).
    pub fn with_vars(vars: Vec<Variable<T>>) -> Self {
        assert!(!vars.is_empty(), "empty domain");
        Self { vars }
    }

    /// Get the variable definitions.
    pub fn vars(&self) -> &[Variable<T>] {
        self.vars.as_slice()
    }
}

impl<T: RealField> FromIterator<Variable<T>> for Domain<T> {
    fn from_iter<I: IntoIterator<Item = Variable<T>>>(iter: I) -> Self {
        Self::with_vars(iter.into_iter().collect())
    }
}

impl<T: RealField> From<Vec<Variable<T>>> for Domain<T> {
    fn from(vars: Vec<Variable<T>>) -> Self {
        Self::with_vars(vars)
    }
}

/// Domain-related extension methods for [`Vector`], which is a common storage
/// for variable values.
pub trait VectorDomainExt<T: RealField, D: Dim> {
    /// Clamp all values within corresponding bounds and returns if the original
    /// value was outside of bounds (in other bounds, the point was not
    /// feasible).
    fn project(&mut self, dom: &Domain<T>) -> bool;
}

impl<T: RealField, D: Dim, S> VectorDomainExt<T, D> for Vector<T, D, S>
where
    S: StorageMut<T, D>,
{
    fn project(&mut self, dom: &Domain<T>) -> bool {
        let not_feasible = self
            .iter()
            .zip(dom.vars().iter())
            .any(|(xi, vi)| !vi.is_within(xi));

        if not_feasible {
            // The point is outside the feasible domain. We need to do the
            // projection.
            self.iter_mut()
                .zip(dom.vars().iter())
                .for_each(|(xi, vi)| *xi = vi.clamp(*xi));
        }

        not_feasible
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    macro_rules! magnitude_of {
        ($lower:expr, $upper:expr) => {
            Variable::builder()
                .bounds($lower, $upper)
                .finalize()
                .magnitude()
        };
    }

    #[test]
    fn magnitude() {
        assert_eq!(magnitude_of!(-1e10f64, 1e10).log10(), 10.0);
        assert_eq!(magnitude_of!(-1e4f64, -1e2).log10(), 3.0);
        assert_eq!(magnitude_of!(-6e-6f64, 9e-6).log10().trunc(), -5.0);

        assert_eq!(magnitude_of!(-6e-6f64, 9e-6) / 1e-5, 1.0);
    }

    #[test]
    fn magnitude_when_bound_is_zero() {
        assert_eq!(magnitude_of!(0f64, 1e2).log10(), 1.0);
        assert_eq!(magnitude_of!(-1e2f64, 0.0).log10(), 1.0);
    }

    #[test]
    fn edge_cases() {
        assert_eq!(magnitude_of!(0.0f64, 0.0), 1.0);
    }
}
