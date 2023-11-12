use nalgebra::{storage::StorageMut, Dynamic, IsContiguous, Vector};

use super::{domain::Domain, system::System};

/// Common interface for all solvers.
///
/// All solvers implement a common interface defined by the [`Solver`] trait.
/// The essential method is [`solve_next`](Solver::solve_next) which takes
/// variables *x* and computes the next step. Thus it represents one iteration
/// in the process. Repeated call to this method should converge *x* to the
/// solution in successful cases.
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
/// use gomez::*;
/// use na::{storage::StorageMut, Dynamic, IsContiguous, Vector};
/// use rand::Rng;
/// use rand_distr::{uniform::SampleUniform, Distribution, Standard};
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
///     F::Field: SampleUniform,
///     Standard: Distribution<F::Field>,
/// {
///     const NAME: &'static str = "Random";
///     type Error = std::convert::Infallible;
///
///     fn solve_next<Sx, Sfx>(
///         &mut self,
///         f: &F,
///         dom: &Domain<F::Field>,
///         x: &mut Vector<F::Field, Dynamic, Sx>,
///         fx: &mut Vector<F::Field, Dynamic, Sfx>,
///     ) -> Result<(), Self::Error>
///     where
///         Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
///         Sfx: StorageMut<F::Field, Dynamic>,
///     {
///         // Randomly sample in the domain.
///         dom.sample(x, &mut self.rng);
///
///         // We must compute the residuals.
///         f.eval(x, fx);
///
///         Ok(())
///     }
/// }
/// ```
pub trait Solver<F: System> {
    /// Name of the solver.
    const NAME: &'static str;

    /// Error type of the iteration. Represents an invalid operation during
    /// computing the next step.
    type Error;

    /// Computes the next step in the solving process.
    ///
    /// The value of `x` is the current values of variables. After the method
    /// returns, `x` should hold the variable values of the performed step and
    /// `fx` *must* contain residuals of that step as computed by
    /// [`System::eval`].
    ///
    /// It is implementation error not to compute the residuals of the computed
    /// step.
    ///
    /// The implementations *can* assume that subsequent calls to `next` pass
    /// the value of `x` as was outputted in the previous iteration by the same
    /// method.
    fn solve_next<Sx, Sfx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Field>,
        x: &mut Vector<F::Field, Dynamic, Sx>,
        fx: &mut Vector<F::Field, Dynamic, Sfx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
        Sfx: StorageMut<F::Field, Dynamic>;
}
