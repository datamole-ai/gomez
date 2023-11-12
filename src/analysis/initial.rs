//! Analysis on initial guesses for Newton-based algorithms.
//!
//! Based on \[1\]. The analysis provide insights for focusing on which
//! variables need to be improved regarding initial guess for Newton-based
//! algorithms (e.g., [`TrustRegion`](crate::solver::trust_region)).
//!
//! \[1\] [On the Choice of Initial Guesses for the Newton-Raphson
//! Algorithm](https://arxiv.org/abs/1911.12433)

use std::marker::PhantomData;

use nalgebra::{
    convert, storage::StorageMut, ComplexField, DimName, Dynamic, IsContiguous, OVector, Vector, U1,
};

use crate::{
    core::{Domain, Problem, System},
    derivatives::{Jacobian, EPSILON_SQRT},
};

/// Initial guesses analyzer. See [module](self) documentation for more details.
pub struct InitialGuessAnalysis<F: Problem> {
    nonlinear: Vec<usize>,
    ty: PhantomData<F>,
}

impl<F: System> InitialGuessAnalysis<F> {
    /// Analyze the system in given point.
    pub fn analyze<Sx, Sfx>(
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, Dynamic, Sx>,
        fx: &mut Vector<F::Scalar, Dynamic, Sfx>,
    ) -> Self
    where
        Sx: StorageMut<F::Scalar, Dynamic> + IsContiguous,
        Sfx: StorageMut<F::Scalar, Dynamic>,
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
        p *= convert::<_, F::Scalar>(0.001);
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
        let nonlinear = jac1
            .column_iter()
            .zip(jac2.column_iter())
            .enumerate()
            .filter(|(_, (c1, c2))| {
                c1.iter()
                    .zip(c2.iter())
                    .any(|(a, b)| (*a - *b).abs() > convert(EPSILON_SQRT))
            })
            .map(|(col, _)| col)
            .collect();

        Self {
            nonlinear,
            ty: PhantomData,
        }
    }

    /// Returns indices of variables that have influence on the Jacobian matrix
    /// of the system.
    ///
    /// Linear variables do not have influence on the Jacobian matrix (the
    /// corresponding columns remain constant during the solving process), thus
    /// they are not important for Newton-based algorithms. Thus, one should
    /// focus on initial guesses for the variables returned by this getter.
    pub fn nonlinear(&self) -> &[usize] {
        &self.nonlinear
    }
}
