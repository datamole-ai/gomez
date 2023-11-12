use nalgebra::{storage::StorageMut, Dynamic, IsContiguous, Vector};

use super::{domain::Domain, function::Function};

/// Common interface for all optimizers.
///
/// All optimizers implement a common interface defined by the [`Optimizer`]
/// trait. The essential method is [`opt_next`](Optimizer::opt_next) which takes
/// variables *x* and computes the next step. Thus it represents one iteration
/// in the process. Repeated call to this method should move *x* towards the
/// minimum in successful cases.
///
/// If you implement an optimizer, please consider make it a contribution to
/// this library.
///
/// ## Implementing an optimizer
///
/// Here is an implementation of a random optimizer (if such a thing can be
/// called an optimizer) which randomly generates values in a hope that
/// eventually goes to the minimum with enough luck.
///
/// ```rust
/// use gomez::nalgebra as na;
/// use gomez::core::*;
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
/// impl<F: Function, R: Rng> Optimizer<F> for Random<R>
/// where
///     F::Field: SampleUniform,
///     Standard: Distribution<F::Field>,
/// {
///     const NAME: &'static str = "Random";
///     type Error = std::convert::Infallible;
///
///     fn opt_next<Sx>(
///         &mut self,
///         f: &F,
///         dom: &Domain<F::Field>,
///         x: &mut Vector<F::Field, Dynamic, Sx>,
///     ) -> Result<F::Field, Self::Error>
///     where
///         Sx: StorageMut<F::Field, Dynamic> + IsContiguous,
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

    /// Error type of the iteration. Represents an invalid operation during
    /// computing the next step.
    type Error;

    /// Computes the next step in the optimization process.
    ///
    /// The value of `x` is the current values of variables. After the method
    /// returns, `x` should hold the variable values of the performed step and
    /// the return value *must* be the function value of that step as computed
    /// by [`Function::apply`].
    ///
    /// It is implementation error not to return function value corresponding to
    /// the computed step.
    ///
    /// The implementations *can* assume that subsequent calls to `next` pass
    /// the value of `x` as was outputted in the previous iteration by the same
    /// method.
    fn opt_next<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Field>,
        x: &mut Vector<F::Field, Dynamic, Sx>,
    ) -> Result<F::Field, Self::Error>
    where
        Sx: StorageMut<F::Field, Dynamic> + IsContiguous;
}
