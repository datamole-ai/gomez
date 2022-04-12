//! Steffensen's method.
//!
//! [Steffensen's](https://en.wikipedia.org/wiki/Steffensen%27s_method) method
//! is a technique similar to Newton's method, but without using derivatives.
//!
//! **Important:** Only one-dimensional systems are supported.
//!
//! # References
//!
//! \[1\] [Wikipedia](https://en.wikipedia.org/wiki/Steffensen%27s_method)
//!
//! \[2\] [A variant of Steffensen's method of fourth-order convergence and its
//! applications](https://www.sciencedirect.com/science/article/pii/S0096300310002705)

use std::marker::PhantomData;

use getset::{CopyGetters, Setters};
use nalgebra::{storage::StorageMut, Dim, IsContiguous, Vector};
use thiserror::Error;

use crate::core::{Domain, Error, Problem, Solver, System, VectorDomainExt};

/// Variant of the Steffenen's method.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SteffensenVariant {
    /// Standard (simplest) variant.
    Standard,
    /// Variant from *A variant of Steffensen's method of fourth-order
    /// convergence and its applications*.
    Liu,
}

/// Options for [`Steffensen`] solver.
#[derive(Debug, Clone, CopyGetters, Setters)]
#[getset(get_copy = "pub", set = "pub")]
pub struct SteffensenOptions<F: Problem> {
    /// Variant of the Steffensen's method. Default: Liu (see
    /// [`SteffensenVariant`]).
    variant: SteffensenVariant,
    #[getset(skip)]
    _phantom: PhantomData<F::Scalar>,
}

impl<F: Problem> Default for SteffensenOptions<F> {
    fn default() -> Self {
        Self {
            variant: SteffensenVariant::Liu,
            _phantom: PhantomData,
        }
    }
}

/// Steffensen solver. See [module](self) documentation for more details.
pub struct Steffensen<F: Problem> {
    options: SteffensenOptions<F>,
}

impl<F: Problem> Steffensen<F> {
    /// Initializes Steffensen solver with default options.
    pub fn new(f: &F, dom: &Domain<F::Scalar>) -> Self {
        Self::with_options(f, dom, SteffensenOptions::default())
    }

    /// Initializes Steffensen solver with given options.
    pub fn with_options(_f: &F, _dom: &Domain<F::Scalar>, options: SteffensenOptions<F>) -> Self {
        Self { options }
    }

    /// Resets the internal state of the solver.
    pub fn reset(&mut self) {}
}

/// Error returned from [`Steffensen`] solver.
#[derive(Debug, Error)]
pub enum SteffensenError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    Problem(#[from] Error),
}

impl<F: System> Solver<F> for Steffensen<F> {
    const NAME: &'static str = "Steffensen";

    type Error = SteffensenError;

    fn next<Sx, Sfx>(
        &mut self,
        f: &F,
        dom: &Domain<<F>::Scalar>,
        x: &mut Vector<<F>::Scalar, <F>::Dim, Sx>,
        fx: &mut Vector<<F>::Scalar, <F>::Dim, Sfx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<<F>::Scalar, <F>::Dim> + IsContiguous,
        Sfx: StorageMut<<F>::Scalar, <F>::Dim>,
    {
        if f.dim().value() != 1 {
            return Err(SteffensenError::Problem(Error::InvalidDimensionality));
        }

        let SteffensenOptions { variant, .. } = self.options;

        let x0 = x[0];

        // Compute f(x).
        f.eval(x, fx)?;
        let fx0 = fx[0];

        match variant {
            SteffensenVariant::Standard => {
                // Compute z = f(x + f(x)) and f(z).
                x[0] += fx0;
                f.eval(x, fx)?;
                let fz0 = fx[0];

                // Compute the next point.
                x[0] = x0 - (fx0 * fx0) / (fz0 - fx0);

                // Compute f(x).
                f.eval(x, fx)?;
            }
            SteffensenVariant::Liu => {
                // Compute z = f(x + f(x)) and f(z).
                x[0] += fx0;
                let z0 = x[0];
                f.eval(x, fx)?;
                let fz0 = fx[0];

                // Compute f[x, z].
                let f_xz = (fz0 - fx0) / (z0 - x0);

                // Compute y = x - f(x) / f[x, z] and f(y).
                x[0] = x0 - fx0 / f_xz;
                let y0 = x[0];
                f.eval(x, fx)?;
                let fy0 = fx[0];

                // Compute f[x, y] and f[y, z].
                let f_xy = (fy0 - fx0) / (y0 - x0);
                let f_yz = (fz0 - fy0) / (z0 - y0);

                // Compute the next point.
                x[0] = y0 - (f_xy - f_yz + f_xz) / (f_xy * f_xy) * fy0;

                // Compute f(x).
                f.eval(x, fx)?;
            }
        }

        x.project(dom);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::*;

    use nalgebra::convert;

    #[test]
    fn sphere_standard() {
        let f = Sphere::new(1);
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let mut options = SteffensenOptions::default();
            options.set_variant(SteffensenVariant::Standard);
            let solver = Steffensen::with_options(&f, &dom, options);
            f.is_root(&solve(&f, &dom, solver, x, 40, eps).unwrap(), eps);
        }
    }

    #[test]
    fn sphere_liu() {
        let f = Sphere::new(1);
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let mut options = SteffensenOptions::default();
            options.set_variant(SteffensenVariant::Liu);
            let solver = Steffensen::with_options(&f, &dom, options);
            f.is_root(&solve(&f, &dom, solver, x, 15, eps).unwrap(), eps);
        }
    }
}
