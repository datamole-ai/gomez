//! Various analyses for supporting the solving.

use nalgebra::{
    convert, storage::StorageMut, ComplexField, DimName, Dynamic, IsContiguous, OVector, RealField,
    Vector, U1,
};

use crate::{
    core::{Domain, RealField as _, System},
    derivatives::Jacobian,
};

/// Estimates magnitude of the variable given lower and upper bounds.
pub fn estimate_magnitude_from_bounds<T: RealField + Copy>(lower: T, upper: T) -> T {
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

/// Detects non-linear variables in the system of equations.
///
/// Based on \[1\]. Linear variables have no effect on Jacobian matrix, thus it
/// makes sense to focus on non-linear variables and their initial guesses for
/// Newton-based algorithms (e.g.,
/// [`TrustRegion`](crate::algo::trust_region)).
///
/// \[1\] [On the Choice of Initial Guesses for the Newton-Raphson
/// Algorithm](https://arxiv.org/abs/1911.12433)
pub fn detect_non_linear_vars_in_system<F, Sx, Sfx>(
    f: &F,
    dom: &Domain<F::Field>,
    x: &mut Vector<F::Field, Dynamic, Sx>,
    fx: &mut Vector<F::Field, Dynamic, Sfx>,
) -> Vec<usize>
where
    F: System,
    Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
    Sfx: StorageMut<F::Field, Dynamic>,
{
    let dim = Dynamic::new(dom.dim());
    let scale = dom
        .scale()
        .map(|scale| OVector::from_iterator_generic(dim, U1::name(), scale.iter().copied()))
        .unwrap_or_else(|| OVector::from_element_generic(dim, U1::name(), convert(1.0)));

    // Compute F'(x) in the initial point.
    f.eval(x, fx);
    let jac1 = Jacobian::new(f, x, &scale, fx);

    // Compute Newton step.
    let mut p = fx.clone_owned();
    p.neg_mut();

    let qr = jac1.clone_owned().qr();
    qr.solve_mut(&mut p);

    // Do Newton step.
    p *= convert::<_, F::Field>(0.001);
    *x += p;

    // Compute F'(x) after one Newton step.
    f.eval(x, fx);
    let jac2 = Jacobian::new(f, x, &scale, fx);

    // Linear variables have no effect on the Jacobian matrix. They can be
    // recognized by observing no change in corresponding columns (i.e.,
    // columns are constant) in two different points. Being constant is
    // checked with tolerance of sqrt(eps) because a change of such
    // magnitude is caused by finite difference approach and does not
    // correspond to the analytic reality. We are interested only in
    // nonlinear variables that have influence on Jacobian matrix.
    jac1.column_iter()
        .zip(jac2.column_iter())
        .enumerate()
        .filter(|(_, (c1, c2))| {
            c1.iter()
                .zip(c2.iter())
                .any(|(a, b)| (*a - *b).abs() > F::Field::EPSILON_SQRT)
        })
        .map(|(col, _)| col)
        .collect()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use crate::Problem;

    use super::*;

    #[test]
    fn magnitude() {
        assert_eq!(estimate_magnitude_from_bounds(-1e10f64, 1e10).log10(), 10.0);
        assert_eq!(estimate_magnitude_from_bounds(-1e4f64, -1e2).log10(), 3.0);
        assert_eq!(
            estimate_magnitude_from_bounds(-6e-6f64, 9e-6)
                .log10()
                .trunc(),
            -5.0
        );

        assert_eq!(estimate_magnitude_from_bounds(-6e-6f64, 9e-6) / 1e-5, 1.0);
    }

    #[test]
    fn magnitude_when_bound_is_zero() {
        assert_eq!(estimate_magnitude_from_bounds(0f64, 1e2).log10(), 1.0);
        assert_eq!(estimate_magnitude_from_bounds(-1e2f64, 0.0).log10(), 1.0);
    }

    #[test]
    fn magnitude_edge_cases() {
        assert_eq!(estimate_magnitude_from_bounds(0.0f64, 0.0), 1.0);
    }

    struct NonLinearTest;

    impl Problem for NonLinearTest {
        type Field = f64;

        fn domain(&self) -> Domain<Self::Field> {
            Domain::unconstrained(2)
        }
    }

    impl System for NonLinearTest {
        fn eval<Sx, Sfx>(
            &self,
            x: &Vector<Self::Field, Dynamic, Sx>,
            fx: &mut Vector<Self::Field, Dynamic, Sfx>,
        ) where
            Sx: nalgebra::Storage<Self::Field, Dynamic> + IsContiguous,
            Sfx: StorageMut<Self::Field, Dynamic>,
        {
            fx[0] = x[0];
            fx[1] = x[1].powi(2);
        }
    }

    #[test]
    fn non_linear_vars_detection_basic() {
        let f = NonLinearTest;
        let dom = f.domain();

        let mut x = nalgebra::dvector![2.0, 2.0];
        let mut fx = x.clone_owned();

        assert_eq!(
            detect_non_linear_vars_in_system(&f, &dom, &mut x, &mut fx),
            vec![1]
        );
    }
}
