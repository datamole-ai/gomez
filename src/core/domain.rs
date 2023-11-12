//! Problem domain definition such as bound constraints for variables.

use std::iter::FromIterator;

use fastrand::Rng;
use na::{Dim, DimName};
use nalgebra as na;
use nalgebra::{storage::StorageMut, OVector, RealField, Vector};

use crate::core::Sample;

fn estimate_magnitude<T: RealField + Copy>(lower: T, upper: T) -> T {
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

/// Domain for a problem.
pub struct Domain<T: RealField + Copy> {
    lower: OVector<T, na::Dynamic>,
    upper: OVector<T, na::Dynamic>,
    scale: Option<OVector<T, na::Dynamic>>,
}

impl<T: RealField + Copy> Domain<T> {
    /// Creates unconstrained domain with given dimension.
    pub fn unconstrained(dim: usize) -> Self {
        assert!(dim > 0, "empty domain");

        let inf = T::from_subset(&f64::INFINITY);
        let n = na::Dynamic::new(dim);
        let one = na::Const::<1>;

        Self {
            lower: OVector::from_iterator_generic(n, one, (0..dim).map(|_| -inf)),
            upper: OVector::from_iterator_generic(n, one, (0..dim).map(|_| inf)),
            scale: None,
        }
    }

    /// Creates rectangular domain with given bounds.
    ///
    /// Positive and negative infinity can be used to indicate value unbounded
    /// in that dimension and direction. If the entire domain is unconstrained,
    /// use [`Domain::unconstrained`] instead.
    pub fn rect(lower: OVector<T, na::Dynamic>, upper: OVector<T, na::Dynamic>) -> Self {
        assert!(lower.ncols() == 1, "lower is not a column vector");
        assert!(upper.ncols() == 1, "upper is not a column vector");
        assert!(
            lower.ncols() == upper.ncols(),
            "lower and upper have different size"
        );

        let dim = lower.nrows();
        assert!(dim > 0, "empty domain");

        let scale = lower
            .iter()
            .copied()
            .zip(upper.iter().copied())
            .map(|(l, u)| estimate_magnitude(l, u));
        let scale = OVector::from_iterator_generic(na::Dynamic::new(dim), na::Const::<1>, scale);

        Self {
            lower,
            upper,
            scale: Some(scale),
        }
    }

    /// Sets a custom scale for the domain.
    ///
    /// Scale value of a variable is the inverse of the expected magnitude of
    /// that variable.
    pub fn with_scale(mut self, scale: OVector<T, na::Dynamic>) -> Self {
        assert!(scale.ncols() == 1, "scale is not a column vector");
        assert!(
            scale.ncols() == self.lower.ncols(),
            "scale has invalid dimension"
        );

        self.scale = Some(scale);
        self
    }

    /// Gets the dimension of the domain.
    pub fn dim(&self) -> usize {
        self.lower.nrows()
    }

    /// Gets the scale if available.
    ///
    /// Scale can be either provided by [`Domain::with_scale`] or estimated for
    /// a constrained domain. If there is no reliable way to estimate the scale
    /// (for unconstrained system), `None` is returned.
    pub fn scale(&self) -> Option<&OVector<T, na::Dynamic>> {
        self.scale.as_ref()
    }

    /// Projects given point into the domain.
    pub fn project<D, Sx>(&self, x: &mut Vector<T, D, Sx>) -> bool
    where
        D: Dim,
        Sx: StorageMut<T, D>,
    {
        let mut not_feasible = false;

        self.lower
            .iter()
            .zip(self.upper.iter())
            .zip(x.iter_mut())
            .for_each(|((li, ui), xi)| {
                if &*xi < li {
                    *xi = *li;
                    not_feasible = true;
                } else if &*xi > ui {
                    *xi = *ui;
                    not_feasible = true;
                }
            });

        not_feasible
    }

    /// Projects given point into the domain in given dimension.
    pub fn project_in<D, Sx>(&self, x: &mut Vector<T, D, Sx>, i: usize) -> bool
    where
        D: Dim,
        Sx: StorageMut<T, D>,
    {
        let li = self.lower[(i, 0)];
        let ui = self.upper[(i, 0)];
        let xi = &mut x[(i, 0)];

        if *xi < li {
            *xi = li;
            true
        } else if *xi > ui {
            *xi = ui;
            true
        } else {
            false
        }
    }

    /// Samples a point in the domain.
    pub fn sample<D, Sx>(&self, x: &mut Vector<T, D, Sx>, rng: &mut Rng)
    where
        D: Dim,
        Sx: StorageMut<T, D> + na::IsContiguous,
        T: Sample,
    {
        x.iter_mut()
            .zip(self.lower.iter().copied().zip(self.upper.iter().copied()))
            .for_each(|(xi, (li, ui))| {
                *xi = if !li.is_finite() || !ui.is_finite() {
                    let random = T::sample_any(rng);

                    if li.is_finite() || ui.is_finite() {
                        let clamped = random.max(li).min(ui);
                        let delta = clamped - random;
                        clamped + delta
                    } else {
                        random
                    }
                } else {
                    T::sample_uniform(li, ui, rng)
                };
            });
    }
}

impl<T: RealField + Copy> FromIterator<(T, T)> for Domain<T> {
    fn from_iter<I: IntoIterator<Item = (T, T)>>(iter: I) -> Self {
        let (lower, upper): (Vec<_>, Vec<_>) = iter.into_iter().unzip();

        let n = na::Dynamic::new(lower.len());

        let lower = OVector::from_vec_generic(n, na::U1::name(), lower);
        let upper = OVector::from_vec_generic(n, na::U1::name(), upper);

        Self::rect(lower, upper)
    }
}

impl<T: RealField + Copy> FromIterator<T> for Domain<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let one = T::from_subset(&1.0);
        let scale = iter
            .into_iter()
            .map(|magnitude| one / magnitude)
            .collect::<Vec<_>>();

        let dim = scale.len();
        let n = na::Dynamic::new(dim);

        let scale = OVector::from_vec_generic(n, na::U1::name(), scale);

        Self::unconstrained(dim).with_scale(scale)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn magnitude() {
        assert_eq!(estimate_magnitude(-1e10f64, 1e10).log10(), 10.0);
        assert_eq!(estimate_magnitude(-1e4f64, -1e2).log10(), 3.0);
        assert_eq!(estimate_magnitude(-6e-6f64, 9e-6).log10().trunc(), -5.0);

        assert_eq!(estimate_magnitude(-6e-6f64, 9e-6) / 1e-5, 1.0);
    }

    #[test]
    fn magnitude_when_bound_is_zero() {
        assert_eq!(estimate_magnitude(0f64, 1e2).log10(), 1.0);
        assert_eq!(estimate_magnitude(-1e2f64, 0.0).log10(), 1.0);
    }

    #[test]
    fn edge_cases() {
        assert_eq!(estimate_magnitude(0.0f64, 0.0), 1.0);
    }
}
