//! Problem domain definition (dimensionality, constraints).

use std::iter::FromIterator;

use fastrand::Rng;
use na::{Dim, DimName};
use nalgebra as na;
use nalgebra::{storage::StorageMut, OVector, Vector};

use crate::analysis::estimate_magnitude_from_bounds;
use crate::core::{RealField, Sample};

/// Domain for a problem.
#[derive(Clone)]
pub struct Domain<T: RealField + Copy> {
    lower: OVector<T, na::Dyn>,
    upper: OVector<T, na::Dyn>,
    scale: Option<OVector<T, na::Dyn>>,
}

impl<T: RealField + Copy> Domain<T> {
    /// Creates unconstrained domain with given dimensionality.
    pub fn unconstrained(dim: usize) -> Self {
        assert!(dim > 0, "empty domain");

        let inf = T::from_subset(&f64::INFINITY);
        let n = na::Dyn(dim);
        let one = na::Const::<1>;

        Self {
            lower: OVector::from_iterator_generic(n, one, (0..dim).map(|_| -inf)),
            upper: OVector::from_iterator_generic(n, one, (0..dim).map(|_| inf)),
            scale: None,
        }
    }

    /// Creates rectangular domain with given lower and upper bounds.
    ///
    /// Positive and negative infinity can be used to indicate a value unbounded
    /// in that dimension and direction. If the entire domain is unconstrained,
    /// use [`Domain::unconstrained`] instead.
    pub fn rect(lower: Vec<T>, upper: Vec<T>) -> Self {
        assert!(
            lower.len() == upper.len(),
            "lower and upper have different size"
        );

        let dim = lower.len();
        assert!(dim > 0, "empty domain");

        let scale = lower
            .iter()
            .copied()
            .zip(upper.iter().copied())
            .map(|(l, u)| estimate_magnitude_from_bounds(l, u));

        let dim = na::Dyn(dim);
        let scale = OVector::from_iterator_generic(dim, na::U1::name(), scale);
        let lower = OVector::from_iterator_generic(dim, na::U1::name(), lower);
        let upper = OVector::from_iterator_generic(dim, na::U1::name(), upper);

        Self {
            lower,
            upper,
            scale: Some(scale),
        }
    }

    /// Sets a custom scale for the domain.
    ///
    /// Scale of a variable is the inverse of its expected magnitude.
    /// Appropriate scaling may be crucial for an algorithm to work well on
    /// "poorly scaled" problems with highly varying magnitudes of its
    /// variables.
    pub fn with_scale(mut self, scale: Vec<T>) -> Self {
        assert!(
            scale.len() == self.lower.nrows(),
            "scale has invalid dimension"
        );

        let dim = na::Dyn(self.lower.nrows());
        let scale = OVector::from_iterator_generic(dim, na::U1::name(), scale);

        self.scale = Some(scale);
        self
    }

    /// Gets the dimensionality of the domain.
    pub fn dim(&self) -> usize {
        self.lower.nrows()
    }

    /// Gets the scale if available.
    ///
    /// Scale can be either provided by [`Domain::with_scale`] or estimated for
    /// a constrained domain. If there is no reliable way to estimate the scale
    /// (for unconstrained system), `None` is returned.
    pub fn scale(&self) -> Option<&OVector<T, na::Dyn>> {
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

        Self::unconstrained(scale.len()).with_scale(scale)
    }
}
