//! Steffensen's method.
//!
//! [Steffensen's](https://en.wikipedia.org/wiki/Steffensen%27s_method) method
//! is a technique similar to Newton's method, but without using derivatives.
//!
//! **Important:** Only one-dimensional systems are supported.
//!
//! # References
//!
//! \[1\] [Steffensen's method](https://en.wikipedia.org/wiki/Steffensen%27s_method) on Wikipedia
//!
//! \[2\] [A variant of Steffensen's method of fourth-order convergence and its
//! applications](https://www.sciencedirect.com/science/article/pii/S0096300310002705)

use std::marker::PhantomData;

use getset::{CopyGetters, Setters};
use nalgebra::{convert, storage::StorageMut, Dyn, IsContiguous, Vector};
use thiserror::Error;

use crate::core::{Domain, Problem, Solver, System};

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
pub struct SteffensenOptions<P: Problem> {
    /// Variant of the Steffensen's method. Default: Liu (see
    /// [`SteffensenVariant`]).
    variant: SteffensenVariant,
    #[getset(skip)]
    _phantom: PhantomData<P::Field>,
}

impl<P: Problem> Default for SteffensenOptions<P> {
    fn default() -> Self {
        Self {
            variant: SteffensenVariant::Liu,
            _phantom: PhantomData,
        }
    }
}

/// Steffensen solver.
///
/// See [module](self) documentation for more details.
pub struct Steffensen<P: Problem> {
    options: SteffensenOptions<P>,
}

impl<P: Problem> Steffensen<P> {
    /// Initializes Steffensen solver with default options.
    pub fn new(p: &P, dom: &Domain<P::Field>) -> Self {
        Self::with_options(p, dom, SteffensenOptions::default())
    }

    /// Initializes Steffensen solver with given options.
    pub fn with_options(_p: &P, _dom: &Domain<P::Field>, options: SteffensenOptions<P>) -> Self {
        Self { options }
    }

    /// Resets the internal state of the solver.
    pub fn reset(&mut self) {}
}

/// Error returned from [`Steffensen`] solver.
#[derive(Debug, Error)]
pub enum SteffensenError {
    /// System is not one-dimensional.
    #[error("system is not one-dimensional")]
    InvalidDimensionality,
}

impl<R: System> Solver<R> for Steffensen<R> {
    const NAME: &'static str = "Steffensen";

    type Error = SteffensenError;

    fn solve_next<Sx, Srx>(
        &mut self,
        r: &R,
        dom: &Domain<R::Field>,
        x: &mut Vector<R::Field, Dyn, Sx>,
        rx: &mut Vector<R::Field, Dyn, Srx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<R::Field, Dyn> + IsContiguous,
        Srx: StorageMut<R::Field, Dyn>,
    {
        if dom.dim() != 1 {
            return Err(SteffensenError::InvalidDimensionality);
        }

        let SteffensenOptions { variant, .. } = self.options;

        let x0 = x[0];

        // Compute r0(x).
        r.eval(x, rx);
        let r0x = rx[0];

        if r0x == convert(0.0) {
            // No more solving to be done.
            return Ok(());
        }

        match variant {
            SteffensenVariant::Standard => {
                // Compute z = f(x + r0(x)) and r0(z).
                x[0] += r0x;
                r.eval(x, rx);
                let fz0 = rx[0];

                // Compute the next point.
                x[0] = x0 - (r0x * r0x) / (fz0 - r0x);

                // Compute r0(x).
                r.eval(x, rx);
            }
            SteffensenVariant::Liu => {
                // Compute z = f(x + r0(x)) and r0(z).
                x[0] += r0x;
                let z0 = x[0];
                r.eval(x, rx);
                let r0z = rx[0];

                // Compute r0[x, z].
                let r0_xz = (r0z - r0x) / (z0 - x0);

                // Compute y = x - r0(x) / r0[x, z] and r0(y).
                x[0] = x0 - r0x / r0_xz;
                let y0 = x[0];
                r.eval(x, rx);
                let r0y = rx[0];

                // Compute r0[x, y] and r0[y, z].
                let r0_xy = (r0y - r0x) / (y0 - x0);
                let r0_yz = (r0z - r0y) / (z0 - y0);

                // Compute the next point.
                x[0] = y0 - (r0_xy - r0_yz + r0_xz) / (r0_xy * r0_xy) * r0y;

                // Compute r0(x).
                r.eval(x, rx);
            }
        }

        dom.project(x);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::*;

    use nalgebra::{convert, DimName, OVector, U1};

    #[test]
    fn sphere_standard() {
        let f = Sphere::new(1);
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let mut options = SteffensenOptions::default();
            options.set_variant(SteffensenVariant::Standard);
            let solver = Steffensen::with_options(&f, &dom, options);
            assert!(f.is_root(&solve(&f, &dom, solver, x, 40, eps).unwrap(), eps));
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
            assert!(f.is_root(&solve(&f, &dom, solver, x, 15, eps).unwrap(), eps));
        }
    }

    #[test]
    fn infinite_solutions() {
        let f = InfiniteSolutions::default();
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let solver = Steffensen::new(&f, &dom);
            assert!(f.is_root(&solve(&f, &dom, solver, x, 25, eps).unwrap(), eps));
        }
    }

    #[test]
    fn stop_at_zero() {
        let f = Sphere::new(1);
        let dom = f.domain();
        let eps = convert(1e-12);
        let solver = Steffensen::new(&f, &dom);

        let x = OVector::from_element_generic(Dyn(dom.dim()), U1::name(), 0.0);
        assert!(f.is_root(&solve(&f, &dom, solver, x, 40, eps).unwrap(), eps));
    }
}
