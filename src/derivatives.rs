//! Tools for derivative-based methods.

use std::ops::Deref;

use nalgebra::{
    allocator::Allocator,
    convert,
    storage::{Storage, StorageMut},
    ComplexField, DefaultAllocator, OMatrix, RealField, Vector,
};
use num_traits::{One, Zero};
use thiserror::Error;

use crate::core::{Error, Problem, System};

/// Square root of double precision machine epsilon. This value is a standard
/// constant for epsilons in approximating derivate-based concepts.
pub const EPSILON_SQRT: f64 = 0.000000014901161193847656;

/// Error when computing the Jacobian matrix.
#[derive(Debug, Error)]
pub enum JacobianError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    Problem(#[from] Error),
}

/// Jacobian matrix of a system.
#[derive(Debug)]
pub struct Jacobian<F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    jac: OMatrix<F::Scalar, F::Dim, F::Dim>,
}

impl<F: Problem> Jacobian<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    /// Initializes the Jacobian matrix with zeros.
    pub fn zeros(f: &F) -> Self {
        Self {
            jac: OMatrix::zeros_generic(f.dim(), f.dim()),
        }
    }
}

impl<F: System> Jacobian<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    /// Compute Compute the Jacobian matrix of the system in given point with
    /// given scale of variables. See [`compute`](Jacobian::compute) for more
    /// details.
    pub fn new<Sx, Sscale, Sfx>(
        f: &F,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        scale: &Vector<F::Scalar, F::Dim, Sscale>,
        fx: &Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<Self, JacobianError>
    where
        Sx: StorageMut<F::Scalar, F::Dim>,
        Sscale: Storage<F::Scalar, F::Dim>,
        Sfx: Storage<F::Scalar, F::Dim>,
    {
        let mut jac = Self::zeros(f);
        jac.compute(f, x, scale, fx)?;
        Ok(jac)
    }

    /// Compute the Jacobian matrix of the system in given point with given
    /// scale of variables.
    ///
    /// The parameter `x` is mutable to allow temporary mutations avoiding
    /// unnecessary allocations, but after this method ends, the content of the
    /// vector is exactly the same as before.
    ///
    /// Information about variable scale is useful for problematic cases of
    /// finite differentiation (e.g., when the value is near zero).
    pub fn compute<Sx, Sscale, Sfx>(
        &mut self,
        f: &F,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        scale: &Vector<F::Scalar, F::Dim, Sscale>,
        fx: &Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<&mut Self, JacobianError>
    where
        Sx: StorageMut<F::Scalar, F::Dim>,
        Sscale: Storage<F::Scalar, F::Dim>,
        Sfx: Storage<F::Scalar, F::Dim>,
    {
        let eps: F::Scalar = convert(EPSILON_SQRT);

        for (j, mut col) in self.jac.column_iter_mut().enumerate() {
            let xj = x[j];

            // Compute the step size. We would like to have the step as small as
            // possible (to be as close to the zero -- i.e., real derivative --
            // the real derivative as possible). But at the same time, very
            // small step could cause F(x + e_j * step_j) ~= F(x) with very
            // small number of good digits.
            //
            // A reasonable way to balance these competing needs is to scale
            // each component by x_j itself. To avoid problems when x_j is close
            // to zero, it is modified to take the typical magnitude instead.
            let magnitude = F::Scalar::one() / scale[j];
            let step = eps * xj.abs().max(magnitude) * xj.copysign(F::Scalar::one());
            let step = if step == F::Scalar::zero() { eps } else { step };

            // Update the point.
            x[j] = xj + step;
            f.eval(x, &mut col)?;

            // Compute the derivative approximation: J[i, j] = (F(x + e_j * step_j) - F(x)) / step_j.
            col -= fx;
            col /= step;

            // Restore the original value.
            x[j] = xj;
        }

        Ok(self)
    }
}

impl<F: Problem> Deref for Jacobian<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    type Target = OMatrix<F::Scalar, F::Dim, F::Dim>;

    fn deref(&self) -> &Self::Target {
        &self.jac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{ExtendedPowell, ExtendedRosenbrock};

    use approx::assert_abs_diff_eq;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn rosenbrock_jacobian() {
        let mut x = dvector![2.0, 2.0];
        let scale = dvector![1.0, 1.0];
        let mut fx = dvector![0.0, 0.0];

        let func = ExtendedRosenbrock::new(2);
        func.eval(&x, &mut fx).unwrap();
        let jac = Jacobian::new(&func, &mut x, &scale, &fx);

        assert!(jac.is_ok());
        let jac = jac.unwrap();

        let expected = dmatrix![-40.0, 10.0; -1.0, 0.0];
        assert_abs_diff_eq!(&*jac, &expected, epsilon = 10e-6);
    }

    #[test]
    fn powell_jacobian_in_root() {
        let mut x = dvector![0.0, 0.0, 0.0, 0.0];
        let scale = dvector![1.0, 1.0, 1.0, 1.0];
        let mut fx = dvector![0.0, 0.0, 0.0, 0.0];

        let func = ExtendedPowell::new(4);
        func.eval(&x, &mut fx).unwrap();
        let jac = Jacobian::new(&func, &mut x, &scale, &fx);

        assert!(jac.is_ok());
        let jac = jac.unwrap();

        let expected = dmatrix![
            1.0, 10.0, 0.0, 0.0;
            0.0, 0.0, 5f64.sqrt(), -(5f64.sqrt());
            0.0, 0.0, 0.0, 0.0;
            0.0, 0.0, 0.0, 0.0
        ];
        assert_abs_diff_eq!(&*jac, &expected, epsilon = 10e-6);
    }
}
