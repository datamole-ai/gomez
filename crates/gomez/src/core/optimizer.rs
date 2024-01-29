use nalgebra::{storage::StorageMut, Dyn, IsContiguous, Vector};

use super::{domain::Domain, function::Function};

/// Interface of an optimizer.
///
/// An optimizer is an iterative algorithm which takes a point _x_ and computes
/// the next step in the optimization process. Repeated calls to the next step
/// should eventually converge into a minimum _x'_.
///
/// If you implement an optimizer, please reach out to discuss if we could
/// include it in gomez.
///
/// ## Implementing an optimizer
///
/// Here is an implementation of a random "optimizer" which randomly generates
/// values in a hope that a minimum can be found with enough luck.
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::{Domain, Function, Optimizer, Sample};
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
/// impl<F: Function> Optimizer<F> for Random
/// where
///     F::Field: Sample,
/// {
///     const NAME: &'static str = "Random";
///     type Error = std::convert::Infallible;
///
///     fn opt_next<Sx>(
///         &mut self,
///         f: &F,
///         dom: &Domain<F::Field>,
///         x: &mut Vector<F::Field, Dyn, Sx>,
///     ) -> Result<F::Field, Self::Error>
///     where
///         Sx: StorageMut<F::Field, Dyn> + IsContiguous,
///     {
///         // Randomly sample in the domain.
///         dom.sample(x, &mut self.rng);
///
///         // We must compute the value.
///         Ok(f.apply(x))
///     }
/// }
/// ```
pub trait Optimizer<F: Function> {
    /// Name of the optimizer.
    const NAME: &'static str;

    /// Error while computing the next step.
    type Error;

    /// Computes the next step in the optimization process.
    ///
    /// The value of `x` is the current point. After the method returns, `x`
    /// should hold the variable values of the performed step and the return
    /// value _must_ be the function value of that step as computed by
    /// [`Function::apply`].
    ///
    /// The implementations _can_ assume that subsequent calls to `opt_next`
    /// pass the value of `x` as was returned in the previous iteration
    fn opt_next<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Field>,
        x: &mut Vector<F::Field, Dyn, Sx>,
    ) -> Result<F::Field, Self::Error>
    where
        Sx: StorageMut<F::Field, Dyn> + IsContiguous;
}
