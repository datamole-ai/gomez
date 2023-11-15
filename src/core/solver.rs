use nalgebra::{storage::StorageMut, Dyn, IsContiguous, Vector};

use super::{domain::Domain, system::System};

/// Interface of a solver.
///
/// A solver is an iterative algorithm which takes a point _x_ and computes the
/// next step in the solving process. Repeated calls to the next step should
/// eventually converge into a solution _x'_ in successful cases.
///
/// If you implement a solver, please reach out to discuss if we could include
/// it in gomez.
///
/// ## Implementing a solver
///
/// Here is an implementation of a random "solver" which randomly generates
/// values in a hope that a solution can be found with enough luck.
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::{Domain, Sample, Solver, System};
/// use na::{storage::StorageMut, Dyn, IsContiguous, Vector};
/// use fastrand::Rng;
///
/// struct Random {
///     rng: Rng,
/// }
///
/// impl Random {
///     fn new(rng: Rng) -> Self {
///         Self { rng }
///     }
/// }
///
/// impl<R: System> Solver<R> for Random
/// where
///     R::Field: Sample,
/// {
///     const NAME: &'static str = "Random";
///     type Error = std::convert::Infallible;
///
///     fn solve_next<Sx, Srx>(
///         &mut self,
///         r: &R,
///         dom: &Domain<R::Field>,
///         x: &mut Vector<R::Field, Dyn, Sx>,
///         rx: &mut Vector<R::Field, Dyn, Srx>,
///     ) -> Result<(), Self::Error>
///     where
///         Sx: StorageMut<R::Field, Dyn> + IsContiguous,
///         Srx: StorageMut<R::Field, Dyn>,
///     {
///         // Randomly sample in the domain.
///         dom.sample(x, &mut self.rng);
///
///         // We must compute the residuals.
///         r.eval(x, rx);
///
///         Ok(())
///     }
/// }
/// ```
pub trait Solver<R: System> {
    /// Name of the solver.
    const NAME: &'static str;

    /// Error while computing the next step.
    type Error;

    /// Computes the next step in the solving process.
    ///
    /// The value of `x` is the current point. After the method returns, `x`
    /// should hold the variable values of the performed step and `rx` _must_
    /// contain residuals of that step as computed by [`System::eval`].
    ///
    /// The implementations _can_ assume that subsequent calls to `solve_next`
    /// pass the value of `x` as was returned in the previous iteration.
    fn solve_next<Sx, Srx>(
        &mut self,
        r: &R,
        dom: &Domain<R::Field>,
        x: &mut Vector<R::Field, Dyn, Sx>,
        rx: &mut Vector<R::Field, Dyn, Srx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<R::Field, Dyn> + IsContiguous,
        Srx: StorageMut<R::Field, Dyn>;
}
