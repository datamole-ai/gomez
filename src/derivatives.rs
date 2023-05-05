//! Tools for derivative-based methods.

use std::ops::Deref;

use nalgebra::{
    allocator::Allocator,
    convert,
    storage::{Storage, StorageMut},
    ComplexField, DefaultAllocator, Dim, DimName, IsContiguous, OMatrix, OVector, RealField,
    Vector, U1,
};
use num_traits::{One, Zero};
use thiserror::Error;

use crate::core::{Function, Problem, ProblemError, System};

/// Square root of double precision machine epsilon. This value is a standard
/// constant for epsilons in approximating first-order derivate-based concepts.
pub const EPSILON_SQRT: f64 = 0.000000014901161193847656;

/// Cubic root of double precision machine epsilon. This value is a standard
/// constant for epsilons in approximating second-order derivate-based concepts.
pub const EPSILON_CBRT: f64 = 0.0000060554544523933395;

/// Error when computing the Jacobian matrix.
#[derive(Debug, Error)]
pub enum JacobianError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    Problem(#[from] ProblemError),
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
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
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
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
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

/// Error when computing the gradient matrix.
#[derive(Debug, Error)]
pub enum GradientError {
    /// Error that occurred when evaluating the function.
    #[error("{0}")]
    Problem(#[from] ProblemError),
}

/// Gradient vector of a function.
#[derive(Debug)]
pub struct Gradient<F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    grad: OVector<F::Scalar, F::Dim>,
}

impl<F: Problem> Gradient<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Initializes the Gradient matrix with zeros.
    pub fn zeros(f: &F) -> Self {
        Self {
            grad: OVector::zeros_generic(f.dim(), U1::name()),
        }
    }
}

impl<F: Function> Gradient<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Compute Compute the gradient vector of the function in given point with
    /// given scale of variables. See [`compute`](Gradient::compute) for more
    /// details.
    pub fn new<Sx, Sscale>(
        f: &F,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        scale: &Vector<F::Scalar, F::Dim, Sscale>,
        fx: F::Scalar,
    ) -> Result<Self, GradientError>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
        Sscale: Storage<F::Scalar, F::Dim>,
    {
        let mut grad = Self::zeros(f);
        grad.compute(f, x, scale, fx)?;
        Ok(grad)
    }

    /// Compute the gradient vector of the function in given point with given
    /// scale of variables.
    ///
    /// The parameter `x` is mutable to allow temporary mutations avoiding
    /// unnecessary allocations, but after this method ends, the content of the
    /// vector is exactly the same as before.
    ///
    /// Information about variable scale is useful for problematic cases of
    /// finite differentiation (e.g., when the value is near zero).
    pub fn compute<Sx, Sscale>(
        &mut self,
        f: &F,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        scale: &Vector<F::Scalar, F::Dim, Sscale>,
        fx: F::Scalar,
    ) -> Result<&mut Self, GradientError>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
        Sscale: Storage<F::Scalar, F::Dim>,
    {
        let eps: F::Scalar = convert(EPSILON_SQRT);

        for i in 0..f.dim().value() {
            let xi = x[i];

            // See the implementation of Jacobian for details on computing step size.
            let magnitude = F::Scalar::one() / scale[i];
            let step = eps * xi.abs().max(magnitude) * F::Scalar::one().copysign(xi);
            let step = if step == F::Scalar::zero() { eps } else { step };

            // Update the point.
            x[i] = xi + step;
            let fxi = f.apply(x)?;

            // Compute the derivative approximation: grad[i] = (F(x + e_i * step_i) - F(x)) / step_i.
            self.grad[i] = (fxi - fx) / step;

            // Restore the original value.
            x[i] = xi;
        }

        Ok(self)
    }
}

impl<F: Problem> Deref for Gradient<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Target = OVector<F::Scalar, F::Dim>;

    fn deref(&self) -> &Self::Target {
        &self.grad
    }
}

/// Error when computing the Hessian matrix.
#[derive(Debug, Error)]
pub enum HessianError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    Problem(#[from] ProblemError),
}

/// Hessian matrix of a system.
#[derive(Debug)]
pub struct Hessian<F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    hes: OMatrix<F::Scalar, F::Dim, F::Dim>,
    steps: OVector<F::Scalar, F::Dim>,
    neighbors: OVector<F::Scalar, F::Dim>,
}

impl<F: Problem> Hessian<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Initializes the Hessian matrix with zeros.
    pub fn zeros(f: &F) -> Self {
        Self {
            hes: OMatrix::zeros_generic(f.dim(), f.dim()),
            steps: OVector::zeros_generic(f.dim(), U1::name()),
            neighbors: OVector::zeros_generic(f.dim(), U1::name()),
        }
    }
}

impl<F: Function> Hessian<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Compute Compute the Hessian matrix of the function in given point with
    /// given scale of variables. See [`compute`](Hessian::compute) for more
    /// details.
    pub fn new<Sx, Sscale>(
        f: &F,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        scale: &Vector<F::Scalar, F::Dim, Sscale>,
        fx: F::Scalar,
    ) -> Result<Self, HessianError>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
        Sscale: Storage<F::Scalar, F::Dim>,
    {
        let mut hes = Self::zeros(f);
        hes.compute(f, x, scale, fx)?;
        Ok(hes)
    }

    /// Compute the Hessian matrix of the function in given point with given
    /// scale of variables.
    ///
    /// The parameter `x` is mutable to allow temporary mutations avoiding
    /// unnecessary allocations, but after this method ends, the content of the
    /// vector is exactly the same as before.
    ///
    /// Information about variable scale is useful for problematic cases of
    /// finite differentiation (e.g., when the value is near zero).
    pub fn compute<Sx, Sscale>(
        &mut self,
        f: &F,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        scale: &Vector<F::Scalar, F::Dim, Sscale>,
        fx: F::Scalar,
    ) -> Result<&mut Self, HessianError>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
        Sscale: Storage<F::Scalar, F::Dim>,
    {
        let eps: F::Scalar = convert(EPSILON_CBRT);

        for i in 0..f.dim().value() {
            let xi = x[i];

            // See the implementation of Jacobian for details on computing step size.
            let magnitude = F::Scalar::one() / scale[i];
            let step = eps * xi.abs().max(magnitude) * F::Scalar::one().copysign(xi);
            let step = if step == F::Scalar::zero() { eps } else { step };

            // Store the step for Hessian calculation.
            self.steps[i] = step;

            // Update the point and store the function output.
            x[i] = xi + step;
            let fxi = f.apply(x)?;
            self.neighbors[i] = fxi;

            // Restore the original value.
            x[i] = xi;
        }

        for i in 0..f.dim().value() {
            let xi = x[i];
            let stepi = self.steps[i];

            // Prepare x_i + 2 * e_i.
            x[i] = xi + stepi + stepi;

            let fxi = f.apply(x)?;
            let fni = self.neighbors[i];

            x[i] = xi + stepi;

            self.hes[(i, i)] = ((fx - fni) + (fxi - fni)) / (stepi * stepi);

            for j in (i + 1)..f.dim().value() {
                let xj = x[j];
                let stepj = self.steps[j];

                x[j] = xj + stepj;

                let fxj = f.apply(x)?;
                let fnj = self.neighbors[j];

                let hij = ((fx - fni) + (fxj - fnj)) / (stepi * stepj);
                self.hes[(i, j)] = hij;
                self.hes[(j, i)] = hij;

                x[j] = xj;
            }

            x[i] = xi;
        }

        Ok(self)
    }
}

impl<F: Problem> Deref for Hessian<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Target = OMatrix<F::Scalar, F::Dim, F::Dim>;

    fn deref(&self) -> &Self::Target {
        &self.hes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{ExtendedPowell, ExtendedRosenbrock};

    use approx::assert_abs_diff_eq;
    use nalgebra::{dmatrix, dvector, Dynamic};

    struct MixedVars;

    impl Problem for MixedVars {
        type Scalar = f64;
        type Dim = Dynamic;

        fn dim(&self) -> Self::Dim {
            Dynamic::new(2)
        }
    }

    impl Function for MixedVars {
        fn apply<Sx>(
            &self,
            x: &Vector<Self::Scalar, Self::Dim, Sx>,
        ) -> Result<Self::Scalar, ProblemError>
        where
            Sx: Storage<Self::Scalar, Self::Dim> + IsContiguous,
        {
            // A simple, arbitrary function that produces Hessian matrix with
            // non-zero corners.
            let x1 = x[0];
            let x2 = x[1];

            Ok(x1.powi(2) + x1 * x2 + x2.powi(3))
        }
    }

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

    #[test]
    fn mixed_vars_gradient() {
        let mut x = dvector![3.0, -3.0];
        let scale = dvector![1.0, 1.0];

        let func = MixedVars;
        let fx = func.apply(&x).unwrap();
        let grad = Gradient::new(&func, &mut x, &scale, fx);

        assert!(grad.is_ok());
        let grad = grad.unwrap();

        let expected = dvector![3.0, 30.0];
        assert_abs_diff_eq!(&*grad, &expected, epsilon = 10e-6);
    }

    #[test]
    fn mixed_vars_hessian() {
        let mut x = dvector![3.0, -3.0];
        let scale = dvector![1.0, 1.0];

        let func = MixedVars;
        let fx = func.apply(&x).unwrap();
        let hes = Hessian::new(&func, &mut x, &scale, fx);

        assert!(hes.is_ok());
        let hes = hes.unwrap();

        let expected = dmatrix![2.0, 1.0; 1.0, -18.0];
        assert_abs_diff_eq!(&*hes, &expected, epsilon = 10e-3);
    }
}
