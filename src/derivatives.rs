//! Tools for derivative-based methods.

use std::ops::Deref;

use nalgebra::{
    storage::{Storage, StorageMut},
    ComplexField, DimName, Dynamic, IsContiguous, OMatrix, OVector, RealField, Vector, U1,
};
use num_traits::{One, Zero};

use crate::core::{Function, Problem, RealField as _, System};

/// Jacobian matrix of a system.
#[derive(Debug)]
pub struct Jacobian<F: Problem> {
    jac: OMatrix<F::Field, Dynamic, Dynamic>,
}

impl<F: Problem> Jacobian<F> {
    /// Initializes the Jacobian matrix with zeros.
    pub fn zeros(f: &F) -> Self {
        let dim = Dynamic::new(f.domain().dim());
        Self {
            jac: OMatrix::zeros_generic(dim, dim),
        }
    }
}

impl<F: System> Jacobian<F> {
    /// Compute Compute the Jacobian matrix of the system in given point with
    /// given scale of variables. See [`compute`](Jacobian::compute) for more
    /// details.
    pub fn new<Sx, Sscale, Sfx>(
        f: &F,
        x: &mut Vector<F::Field, Dynamic, Sx>,
        scale: &Vector<F::Field, Dynamic, Sscale>,
        fx: &Vector<F::Field, Dynamic, Sfx>,
    ) -> Self
    where
        Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
        Sscale: Storage<F::Field, Dynamic>,
        Sfx: Storage<F::Field, Dynamic>,
    {
        let mut jac = Self::zeros(f);
        jac.compute(f, x, scale, fx);
        jac
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
        x: &mut Vector<F::Field, Dynamic, Sx>,
        scale: &Vector<F::Field, Dynamic, Sscale>,
        fx: &Vector<F::Field, Dynamic, Sfx>,
    ) -> &mut Self
    where
        Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
        Sscale: Storage<F::Field, Dynamic>,
        Sfx: Storage<F::Field, Dynamic>,
    {
        let eps = F::Field::EPSILON_SQRT;

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
            let magnitude = F::Field::one() / scale[j];
            let step = eps * xj.abs().max(magnitude) * xj.copysign(F::Field::one());
            let step = if step == F::Field::zero() { eps } else { step };

            // Update the point.
            x[j] = xj + step;
            f.eval(x, &mut col);

            // Compute the derivative approximation: J[i, j] = (F(x + e_j * step_j) - F(x)) / step_j.
            col -= fx;
            col /= step;

            // Restore the original value.
            x[j] = xj;
        }

        self
    }
}

impl<F: Problem> Deref for Jacobian<F> {
    type Target = OMatrix<F::Field, Dynamic, Dynamic>;

    fn deref(&self) -> &Self::Target {
        &self.jac
    }
}

/// Gradient vector of a function.
#[derive(Debug)]
pub struct Gradient<F: Problem> {
    grad: OVector<F::Field, Dynamic>,
}

impl<F: Problem> Gradient<F> {
    /// Initializes the Gradient matrix with zeros.
    pub fn zeros(f: &F) -> Self {
        let dim = Dynamic::new(f.domain().dim());
        Self {
            grad: OVector::zeros_generic(dim, U1::name()),
        }
    }
}

impl<F: Function> Gradient<F> {
    /// Compute Compute the gradient vector of the function in given point with
    /// given scale of variables. See [`compute`](Gradient::compute) for more
    /// details.
    pub fn new<Sx, Sscale>(
        f: &F,
        x: &mut Vector<F::Field, Dynamic, Sx>,
        scale: &Vector<F::Field, Dynamic, Sscale>,
        fx: F::Field,
    ) -> Self
    where
        Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
        Sscale: Storage<F::Field, Dynamic>,
    {
        let mut grad = Self::zeros(f);
        grad.compute(f, x, scale, fx);
        grad
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
        x: &mut Vector<F::Field, Dynamic, Sx>,
        scale: &Vector<F::Field, Dynamic, Sscale>,
        fx: F::Field,
    ) -> &mut Self
    where
        Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
        Sscale: Storage<F::Field, Dynamic>,
    {
        let eps = F::Field::EPSILON_SQRT;

        for i in 0..x.nrows() {
            let xi = x[i];

            // See the implementation of Jacobian for details on computing step size.
            let magnitude = F::Field::one() / scale[i];
            let step = eps * xi.abs().max(magnitude) * F::Field::one().copysign(xi);
            let step = if step == F::Field::zero() { eps } else { step };

            // Update the point.
            x[i] = xi + step;
            let fxi = f.apply(x);

            // Compute the derivative approximation: grad[i] = (F(x + e_i * step_i) - F(x)) / step_i.
            self.grad[i] = (fxi - fx) / step;

            // Restore the original value.
            x[i] = xi;
        }

        self
    }
}

impl<F: Problem> Deref for Gradient<F> {
    type Target = OVector<F::Field, Dynamic>;

    fn deref(&self) -> &Self::Target {
        &self.grad
    }
}

/// Hessian matrix of a system.
#[derive(Debug)]
pub struct Hessian<F: Problem> {
    hes: OMatrix<F::Field, Dynamic, Dynamic>,
    steps: OVector<F::Field, Dynamic>,
    neighbors: OVector<F::Field, Dynamic>,
}

impl<F: Problem> Hessian<F> {
    /// Initializes the Hessian matrix with zeros.
    pub fn zeros(f: &F) -> Self {
        let dim = Dynamic::new(f.domain().dim());
        Self {
            hes: OMatrix::zeros_generic(dim, dim),
            steps: OVector::zeros_generic(dim, U1::name()),
            neighbors: OVector::zeros_generic(dim, U1::name()),
        }
    }
}

impl<F: Function> Hessian<F> {
    /// Compute Compute the Hessian matrix of the function in given point with
    /// given scale of variables. See [`compute`](Hessian::compute) for more
    /// details.
    pub fn new<Sx, Sscale>(
        f: &F,
        x: &mut Vector<F::Field, Dynamic, Sx>,
        scale: &Vector<F::Field, Dynamic, Sscale>,
        fx: F::Field,
    ) -> Self
    where
        Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
        Sscale: Storage<F::Field, Dynamic>,
    {
        let mut hes = Self::zeros(f);
        hes.compute(f, x, scale, fx);
        hes
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
        x: &mut Vector<F::Field, Dynamic, Sx>,
        scale: &Vector<F::Field, Dynamic, Sscale>,
        fx: F::Field,
    ) -> &mut Self
    where
        Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
        Sscale: Storage<F::Field, Dynamic>,
    {
        let eps = F::Field::EPSILON_CBRT;

        for i in 0..x.nrows() {
            let xi = x[i];

            // See the implementation of Jacobian for details on computing step size.
            let magnitude = F::Field::one() / scale[i];
            let step = eps * xi.abs().max(magnitude) * F::Field::one().copysign(xi);
            let step = if step == F::Field::zero() { eps } else { step };

            // Store the step for Hessian calculation.
            self.steps[i] = step;

            // Update the point and store the function output.
            x[i] = xi + step;
            let fxi = f.apply(x);
            self.neighbors[i] = fxi;

            // Restore the original value.
            x[i] = xi;
        }

        for i in 0..x.nrows() {
            let xi = x[i];
            let stepi = self.steps[i];

            // Prepare x_i + 2 * e_i.
            x[i] = xi + stepi + stepi;

            let fxi = f.apply(x);
            let fni = self.neighbors[i];

            x[i] = xi + stepi;

            self.hes[(i, i)] = ((fx - fni) + (fxi - fni)) / (stepi * stepi);

            for j in (i + 1)..x.nrows() {
                let xj = x[j];
                let stepj = self.steps[j];

                x[j] = xj + stepj;

                let fxj = f.apply(x);
                let fnj = self.neighbors[j];

                let hij = ((fx - fni) + (fxj - fnj)) / (stepi * stepj);
                self.hes[(i, j)] = hij;
                self.hes[(j, i)] = hij;

                x[j] = xj;
            }

            x[i] = xi;
        }

        self
    }
}

impl<F: Problem> Deref for Hessian<F> {
    type Target = OMatrix<F::Field, Dynamic, Dynamic>;

    fn deref(&self) -> &Self::Target {
        &self.hes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::Domain,
        testing::{ExtendedPowell, ExtendedRosenbrock},
    };

    use approx::assert_abs_diff_eq;
    use nalgebra::{dmatrix, dvector, Dynamic};

    struct MixedVars;

    impl Problem for MixedVars {
        type Field = f64;

        fn domain(&self) -> Domain<Self::Field> {
            Domain::unconstrained(2)
        }
    }

    impl Function for MixedVars {
        fn apply<Sx>(&self, x: &Vector<Self::Field, Dynamic, Sx>) -> Self::Field
        where
            Sx: Storage<Self::Field, Dynamic> + IsContiguous,
        {
            // A simple, arbitrary function that produces Hessian matrix with
            // non-zero corners.
            let x1 = x[0];
            let x2 = x[1];

            x1.powi(2) + x1 * x2 + x2.powi(3)
        }
    }

    #[test]
    fn rosenbrock_jacobian() {
        let mut x = dvector![2.0, 2.0];
        let scale = dvector![1.0, 1.0];
        let mut fx = dvector![0.0, 0.0];

        let func = ExtendedRosenbrock::new(2);
        func.eval(&x, &mut fx);
        let jac = Jacobian::new(&func, &mut x, &scale, &fx);

        let expected = dmatrix![-40.0, 10.0; -1.0, 0.0];
        assert_abs_diff_eq!(&*jac, &expected, epsilon = 10e-6);
    }

    #[test]
    fn powell_jacobian_in_root() {
        let mut x = dvector![0.0, 0.0, 0.0, 0.0];
        let scale = dvector![1.0, 1.0, 1.0, 1.0];
        let mut fx = dvector![0.0, 0.0, 0.0, 0.0];

        let func = ExtendedPowell::new(4);
        func.eval(&x, &mut fx);
        let jac = Jacobian::new(&func, &mut x, &scale, &fx);

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
        let fx = func.apply(&x);
        let grad = Gradient::new(&func, &mut x, &scale, fx);

        let expected = dvector![3.0, 30.0];
        assert_abs_diff_eq!(&*grad, &expected, epsilon = 10e-6);
    }

    #[test]
    fn mixed_vars_hessian() {
        let mut x = dvector![3.0, -3.0];
        let scale = dvector![1.0, 1.0];

        let func = MixedVars;
        let fx = func.apply(&x);
        let hes = Hessian::new(&func, &mut x, &scale, fx);

        let expected = dmatrix![2.0, 1.0; 1.0, -18.0];
        assert_abs_diff_eq!(&*hes, &expected, epsilon = 10e-3);
    }
}
