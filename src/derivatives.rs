//! Tools for derivative-based methods.

use std::ops::Deref;

use nalgebra::{
    convert,
    storage::{Storage, StorageMut},
    DimName, Dyn, IsContiguous, OMatrix, OVector, Vector, U1,
};

use crate::core::{Function, Problem, RealField, System};

/// Jacobian matrix of a system of equations.
#[derive(Debug)]
pub struct Jacobian<R: Problem> {
    jac: OMatrix<R::Field, Dyn, Dyn>,
}

impl<R: Problem> Jacobian<R> {
    /// Initializes the Jacobian matrix with zeros.
    pub fn zeros(r: &R) -> Self {
        let dim = Dyn(r.domain().dim());
        Self {
            jac: OMatrix::zeros_generic(dim, dim),
        }
    }
}

impl<R: System> Jacobian<R> {
    /// Computes the Jacobian matrix of the system of equations in given point
    /// with given scale of variables. See [`compute`](Jacobian::compute) for
    /// more details.
    pub fn new<Sx, Sscale, Srx>(
        r: &R,
        x: &mut Vector<R::Field, Dyn, Sx>,
        scale: &Vector<R::Field, Dyn, Sscale>,
        rx: &Vector<R::Field, Dyn, Srx>,
        eps: R::Field,
        step_rule: StepRule,
    ) -> Self
    where
        Sx: StorageMut<R::Field, Dyn> + IsContiguous,
        Sscale: Storage<R::Field, Dyn>,
        Srx: Storage<R::Field, Dyn>,
    {
        let mut jac = Self::zeros(r);
        jac.compute(r, x, scale, rx, eps, step_rule);
        jac
    }

    /// Computes the Jacobian matrix of the system of equations in given point
    /// with given scale of variables.
    ///
    /// The parameter `x` is mutable to allow temporary mutations avoiding
    /// unnecessary allocations, but after this method ends, the content of the
    /// vector is exactly the same as before.
    ///
    /// Information about variable scale is useful for problematic cases of
    /// finite differentiation (e.g., when the value is near zero).
    pub fn compute<Sx, Sscale, Srx>(
        &mut self,
        r: &R,
        x: &mut Vector<R::Field, Dyn, Sx>,
        scale: &Vector<R::Field, Dyn, Sscale>,
        rx: &Vector<R::Field, Dyn, Srx>,
        eps_rel: R::Field,
        step_rule: StepRule,
    ) -> &mut Self
    where
        Sx: StorageMut<R::Field, Dyn> + IsContiguous,
        Sscale: Storage<R::Field, Dyn>,
        Srx: Storage<R::Field, Dyn>,
    {
        let one: R::Field = convert(1.0);

        for (j, mut col) in self.jac.column_iter_mut().enumerate() {
            let xj = x[j];
            let step = step_rule.apply(xj, one / scale[j], eps_rel);

            // Update the point.
            x[j] = xj + step;
            r.eval(x, &mut col);

            // Compute the derivative approximation: J[i, j] = (r(x + e_j * step_j) - r(x)) / step_j.
            col -= rx;
            col /= step;

            // Restore the original value.
            x[j] = xj;
        }

        self
    }
}

impl<R: Problem> Deref for Jacobian<R> {
    type Target = OMatrix<R::Field, Dyn, Dyn>;

    fn deref(&self) -> &Self::Target {
        &self.jac
    }
}

/// Gradient vector of a function.
#[derive(Debug)]
pub struct Gradient<F: Problem> {
    grad: OVector<F::Field, Dyn>,
}

impl<F: Problem> Gradient<F> {
    /// Initializes the gradient vector with zeros.
    pub fn zeros(f: &F) -> Self {
        let dim = Dyn(f.domain().dim());
        Self {
            grad: OVector::zeros_generic(dim, U1::name()),
        }
    }
}

impl<F: Function> Gradient<F> {
    /// Computes the gradient vector of the function in given point with given
    /// scale of variables. See [`compute`](Gradient::compute) for more details.
    pub fn new<Sx, Sscale>(
        f: &F,
        x: &mut Vector<F::Field, Dyn, Sx>,
        scale: &Vector<F::Field, Dyn, Sscale>,
        fx: F::Field,
        eps_rel: F::Field,
        step_rule: StepRule,
    ) -> Self
    where
        Sx: StorageMut<F::Field, Dyn> + IsContiguous,
        Sscale: Storage<F::Field, Dyn>,
    {
        let mut grad = Self::zeros(f);
        grad.compute(f, x, scale, fx, eps_rel, step_rule);
        grad
    }

    /// Computes the gradient vector of the function in given point with given
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
        x: &mut Vector<F::Field, Dyn, Sx>,
        scale: &Vector<F::Field, Dyn, Sscale>,
        fx: F::Field,
        eps_rel: F::Field,
        step_rule: StepRule,
    ) -> &mut Self
    where
        Sx: StorageMut<F::Field, Dyn> + IsContiguous,
        Sscale: Storage<F::Field, Dyn>,
    {
        let one: F::Field = convert(1.0);

        for i in 0..x.nrows() {
            let xi = x[i];
            let step = step_rule.apply(xi, one / scale[i], eps_rel);

            // Update the point.
            x[i] = xi + step;
            let fxi = f.apply(x);

            // Compute the derivative approximation: grad[i] = (f(x + e_i * step_i) - f(x)) / step_i.
            self.grad[i] = (fxi - fx) / step;

            // Restore the original value.
            x[i] = xi;
        }

        self
    }
}

impl<F: Problem> Deref for Gradient<F> {
    type Target = OVector<F::Field, Dyn>;

    fn deref(&self) -> &Self::Target {
        &self.grad
    }
}

/// Hessian matrix of a function.
#[derive(Debug)]
pub struct Hessian<F: Problem> {
    hes: OMatrix<F::Field, Dyn, Dyn>,
    steps: OVector<F::Field, Dyn>,
    neighbors: OVector<F::Field, Dyn>,
}

impl<F: Problem> Hessian<F> {
    /// Initializes the Hessian matrix with zeros.
    pub fn zeros(f: &F) -> Self {
        let dim = Dyn(f.domain().dim());
        Self {
            hes: OMatrix::zeros_generic(dim, dim),
            steps: OVector::zeros_generic(dim, U1::name()),
            neighbors: OVector::zeros_generic(dim, U1::name()),
        }
    }
}

impl<F: Function> Hessian<F> {
    /// Computes the Hessian matrix of the function in given point with given
    /// scale of variables. See [`compute`](Hessian::compute) for more details.
    pub fn new<Sx, Sscale>(
        f: &F,
        x: &mut Vector<F::Field, Dyn, Sx>,
        scale: &Vector<F::Field, Dyn, Sscale>,
        fx: F::Field,
        eps_rel: F::Field,
        step_rule: StepRule,
    ) -> Self
    where
        Sx: StorageMut<F::Field, Dyn> + IsContiguous,
        Sscale: Storage<F::Field, Dyn>,
    {
        let mut hes = Self::zeros(f);
        hes.compute(f, x, scale, fx, eps_rel, step_rule);
        hes
    }

    /// Computes the Hessian matrix of the function in given point with given
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
        x: &mut Vector<F::Field, Dyn, Sx>,
        scale: &Vector<F::Field, Dyn, Sscale>,
        fx: F::Field,
        eps_rel: F::Field,
        step_rule: StepRule,
    ) -> &mut Self
    where
        Sx: StorageMut<F::Field, Dyn> + IsContiguous,
        Sscale: Storage<F::Field, Dyn>,
    {
        let one: F::Field = convert(1.0);

        for i in 0..x.nrows() {
            let xi = x[i];
            let step = step_rule.apply(xi, one / scale[i], eps_rel);

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
    type Target = OMatrix<F::Field, Dyn, Dyn>;

    fn deref(&self) -> &Self::Target {
        &self.hes
    }
}

/// Rule for computing the step in finite difference method.
#[derive(Debug, Default, Clone, Copy)]
#[non_exhaustive]
pub enum StepRule {
    /// _eps<sub>rel</sub>_
    Fixed,
    /// _eps<sub>rel</sub> * x_
    ValueScaled,
    /// _eps<sub>rel</sub> * |x|_
    #[default]
    AbsValueScaled,
    /// _eps<sub>rel</sub> * max(|x|, mag) * sign(x)_
    ValueOrMagScaled,
    /// _eps<sub>rel</sub> * max(|x|, mag)_
    AbsValueOrMagScaled,
    /// _eps<sub>rel</sub> * max(|x|, mag) * x_
    ValueOrMagTimesValueScaled,
    /// _eps<sub>rel</sub> * max(|x|, mag) * |x|_
    AbsValueOrMagTimesAbsValueScaled,
}

impl StepRule {
    fn apply<T: RealField + Copy>(&self, x: T, mag: T, eps_rel: T) -> T {
        // Compute the step size. We would like to have the step as small as
        // possible (to be as close to the zero -- i.e., real derivative -- as
        // possible). But at the same time, very small step could cause r(x +
        // e_j * step_j) ~= r(x) with very small number of good digits.
        //
        // We implement multiple rules. One is just using fixed relative epsilon
        // value. The other use some form of scaling by the variable value,
        // which balances the competing needs mentioned in the first paragraph.
        // Some of them also utilize the information of estimated typical
        // magnitude of the variable, mostly to avoid problems when the variable
        // value is close to zero. Some methods inherit the sign of the variable
        // value for the step size, while the others always use a positive
        // value.

        let one = convert(1.0);
        let scale = match self {
            StepRule::Fixed => one,
            StepRule::ValueScaled => x,
            StepRule::AbsValueScaled => x.abs(),
            StepRule::ValueOrMagScaled => x.abs().max(mag) * one.copysign(x),
            StepRule::AbsValueOrMagScaled => x.abs().max(mag),
            StepRule::ValueOrMagTimesValueScaled => x.abs().max(mag) * x,
            StepRule::AbsValueOrMagTimesAbsValueScaled => x.abs().max(mag) * x.abs(),
        };

        // Since some of the rules use the value of x directly, we need to check
        // for cases when the value is zero, in which case we need to fallback
        // to some non-zero step size.
        if scale != convert(0.0) {
            eps_rel * scale
        } else {
            eps_rel
        }
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
    use nalgebra::{dmatrix, dvector, Dyn};

    struct MixedVars;

    impl Problem for MixedVars {
        type Field = f64;

        fn domain(&self) -> Domain<Self::Field> {
            Domain::unconstrained(2)
        }
    }

    impl Function for MixedVars {
        fn apply<Sx>(&self, x: &Vector<Self::Field, Dyn, Sx>) -> Self::Field
        where
            Sx: Storage<Self::Field, Dyn> + IsContiguous,
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
        let jac = Jacobian::new(
            &func,
            &mut x,
            &scale,
            &fx,
            f64::EPSILON_SQRT,
            StepRule::default(),
        );

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
        let jac = Jacobian::new(
            &func,
            &mut x,
            &scale,
            &fx,
            f64::EPSILON_SQRT,
            StepRule::default(),
        );

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
        let grad = Gradient::new(
            &func,
            &mut x,
            &scale,
            fx,
            f64::EPSILON_SQRT,
            StepRule::default(),
        );

        let expected = dvector![3.0, 30.0];
        assert_abs_diff_eq!(&*grad, &expected, epsilon = 10e-6);
    }

    #[test]
    fn mixed_vars_hessian() {
        let mut x = dvector![3.0, -3.0];
        let scale = dvector![1.0, 1.0];

        let func = MixedVars;
        let fx = func.apply(&x);
        let hes = Hessian::new(
            &func,
            &mut x,
            &scale,
            fx,
            f64::EPSILON_CBRT,
            StepRule::default(),
        );

        let expected = dmatrix![2.0, 1.0; 1.0, -18.0];
        assert_abs_diff_eq!(&*hes, &expected, epsilon = 10e-3);
    }
}
