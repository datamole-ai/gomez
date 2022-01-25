use nalgebra::{storage::StorageMut, Vector};

use super::domain::Domain;
use super::system::System;

/// Common interface for all solvers.
///
/// All solvers implement a common interface defined by the [`Solver`] trait.
/// The essential method is [`next`](Solver::next) which takes variables *x* and
/// computes the next step. Thus it represents one iteration in the process.
/// Repeated call to this method should converge *x* to the solution in
/// successful cases.
///
/// If you implement a solver, please consider make it a contribution to this
/// library.
///
/// ## Implementing a solver
///
/// Here is an implementation of a random solver (if such a thing can be called
/// a solver) which randomly generates values in a hope that a solution can be
/// found with enough luck.
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::core::{Domain, Solver, System, SystemError};
/// use na::{storage::StorageMut, Vector};
/// use rand::Rng;
/// use rand_distr::{uniform::SampleUniform, Distribution, Uniform};
///
/// struct Random<R> {
///     rng: R,
/// }
///
/// impl<R> Random<R> {
///     fn new(rng: R) -> Self {
///         Self { rng }
///     }
/// }
///
/// impl<F: System, R: Rng> Solver<F> for Random<R>
/// where
///     F::Scalar: SampleUniform,
/// {
///     const NAME: &'static str = "Random";
///     type Error = SystemError;
///
///     fn next<Sx, Sfx>(
///         &mut self,
///         f: &F,
///         dom: &Domain<F::Scalar>,
///         x: &mut Vector<F::Scalar, F::Dim, Sx>,
///         fx: &mut Vector<F::Scalar, F::Dim, Sfx>,
///     ) -> Result<(), Self::Error>
///     where
///         Sx: StorageMut<F::Scalar, F::Dim>,
///         Sfx: StorageMut<F::Scalar, F::Dim>,
///     {
///         // Randomly sample within the bounds.
///         x.iter_mut().zip(dom.vars().iter()).for_each(|(xi, vi)| {
///             *xi = Uniform::new_inclusive(vi.lower(), vi.upper()).sample(&mut self.rng)
///         });
///
///         // We must compute the residuals.
///         f.apply(x, fx)?;
///
///         Ok(())
///     }
/// }
/// ```
pub trait Solver<F: System> {
    /// Name of the solver.
    const NAME: &'static str;

    /// Error type of the iteration. Represents an invalid operation during
    /// computing the next step. It is usual that one of the error kinds is
    /// propagation of the [`SystemError`](super::SystemError) from the system.
    type Error;

    /// Computes the next step in the solving process.
    ///
    /// The value of `x` is the current values of variables. After the method
    /// returns, `x` should hold the variable values of the performed step and
    /// `fx` *must* contain residuals of that step as computed by
    /// [`System::apply`].
    ///
    /// It is implementation error not to compute the residuals of the computed
    /// step.
    ///
    /// The implementations *can* assume that subsequent calls to `next` pass
    /// the value of `x` as was outputted in the previous iteration by the same
    /// method.
    fn next<Sx, Sfx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        fx: &mut Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<F::Scalar, F::Dim>,
        Sfx: StorageMut<F::Scalar, F::Dim>;
}
