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
    allocator::Allocator, convert, storage::StorageMut, ComplexField, DefaultAllocator, DimMin,
    DimName, OVector, Vector, U1,
};
use thiserror::Error;

use crate::{
    core::{Domain, System, SystemError},
    derivatives::{Jacobian, JacobianError, EPSILON_SQRT},
};

/// Error returned from [`InitialGuessAnalysis`] solver.
#[derive(Debug, Error)]
pub enum InitialGuessAnalysisError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    System(#[from] SystemError),
    /// Error that occurred when computing the Jacobian matrix.
    #[error("{0}")]
    Jacobian(#[from] JacobianError),
}

/// Initial guesses analyzer. See [module](self) documentation for more details.
pub struct InitialGuessAnalysis<F: System> {
    nonlinear: Vec<usize>,
    ty: PhantomData<F>,
}

impl<F: System> InitialGuessAnalysis<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    F::Dim: DimMin<F::Dim, Output = F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, <F::Dim as DimMin<F::Dim>>::Output>,
{
    /// Analyze the system in given point.
    pub fn analyze<Sx, Sfx>(
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        fx: &mut Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<Self, InitialGuessAnalysisError>
    where
        Sx: StorageMut<F::Scalar, F::Dim>,
        Sfx: StorageMut<F::Scalar, F::Dim>,
    {
        let scale_iter = dom.vars().iter().map(|var| var.scale());
        let scale = OVector::from_iterator_generic(f.dim(), U1::name(), scale_iter);

        // Compute F'(x) in the initial point.
        f.apply(x, fx)?;
        let jac1 = Jacobian::new(f, x, &scale, fx)?;

        // Compute Newton step.
        let mut p = fx.clone_owned();
        p.neg_mut();

        let qr = jac1.clone_owned().qr();
        qr.solve_mut(&mut p);

        // Do Newton step.
        p *= convert::<_, F::Scalar>(0.001);
        *x += p;

        // Compute F'(x) after one Newton step.
        f.apply(x, fx)?;
        let jac2 = Jacobian::new(f, x, &scale, fx)?;

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

        Ok(Self {
            nonlinear,
            ty: PhantomData,
        })
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
