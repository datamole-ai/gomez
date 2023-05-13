//! Global optimization method based on Lipschitz constant with local
//! optimization extension.
//!
//! # References
//!
//! \[1\] [Global optimization of Lipschitz
//! functions](https://arxiv.org/abs/1703.02628)
//!
//! \[2\] [A Global Optimization Algorithm Worth
//! Using](http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html)

use getset::{CopyGetters, Setters};
use log::{debug, trace};
use nalgebra::{
    allocator::Allocator, convert, try_convert, ComplexField, DefaultAllocator, Dim, DimName,
    IsContiguous, OVector, RealField, StorageMut, Vector, U1,
};
use num_traits::{One, Zero};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Bernoulli, Distribution, Standard, Uniform};
use thiserror::Error;

use crate::core::{Domain, Function, Optimizer, Problem, ProblemError, Solver, System};

use super::NelderMead;

/// Specification for the alpha parameter for (1 + alpha)^i meshgrid for
/// Lipschitz constant estimation.
#[derive(Debug, Clone, Copy)]
pub enum AlphaInit<S> {
    /// Fixed value.
    Fixed(S),
    /// FIxed value divided by the dimensionality of the problem.
    ScaledByDim(S),
}

/// Strategy for choosing potential minimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PotentialMinimizer {
    /// Any valid potential minimizer is chosen.
    Any,
    /// The best potential minimizer out of a finite number of trials is chosen.
    Best,
}

/// Options for [`Lipo`] solver.
#[derive(Debug, Clone, CopyGetters, Setters)]
#[getset(get_copy = "pub", set = "pub")]
pub struct LipoOptions<F: Problem> {
    /// Probability for Bernoulli distribution of exploring (evaluating sampled
    /// point unconditionally) the space.
    p_explore: f64,
    /// Alpha parameter for (1 + alpha)^i meshgrid for Lipschitz constant
    /// estimation.
    alpha: AlphaInit<F::Scalar>,
    /// Number of sampling trials. If no potential minimizer is found after this
    /// number of trials, the solver returns error.
    sampling_trials: usize,
    /// Strategy for choosing potential minimizer.
    potential_minimizer: PotentialMinimizer,
    /// Number of iterations for local optimization.
    local_optimization_iters: usize,
}

impl<F: Problem> Default for LipoOptions<F> {
    fn default() -> Self {
        Self {
            p_explore: 0.1,
            alpha: AlphaInit::ScaledByDim(convert(0.01)),
            sampling_trials: 5000,
            potential_minimizer: PotentialMinimizer::Best,
            local_optimization_iters: 5,
        }
    }
}

/// LIPO solver. See [module](self) documentation for more details.
pub struct Lipo<F: Problem, R>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    options: LipoOptions<F>,
    alpha: F::Scalar,
    xs: Vec<OVector<F::Scalar, F::Dim>>,
    ys: Vec<F::Scalar>,
    best: usize,
    k: F::Scalar,
    k_inf: F::Scalar,
    rng: R,
    bernoulli: Bernoulli,
    tmp: OVector<F::Scalar, F::Dim>,
    x_tmp: OVector<F::Scalar, F::Dim>,
    local_optimizer: NelderMead<F>,
    iter: usize,
}

impl<F: Problem, R: Rng> Lipo<F, R>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Initializes LIPO solver with default options.
    pub fn new(f: &F, dom: &Domain<F::Scalar>, rng: R) -> Self {
        Self::with_options(f, dom, LipoOptions::default(), rng)
    }

    /// Initializes LIPO solver with given options.
    pub fn with_options(f: &F, dom: &Domain<F::Scalar>, options: LipoOptions<F>, rng: R) -> Self {
        let p_explore = options.p_explore.clamp(0.0, 1.0);

        let alpha = match options.alpha {
            AlphaInit::Fixed(alpha) => alpha,
            AlphaInit::ScaledByDim(alpha) => alpha / convert(f.dim().value() as f64),
        };

        Self {
            options,
            alpha,
            xs: Vec::new(),
            ys: Vec::new(),
            best: 0,
            k: F::Scalar::zero(),
            k_inf: F::Scalar::zero(),
            rng,
            bernoulli: Bernoulli::new(p_explore).unwrap(),
            tmp: OVector::zeros_generic(f.dim(), U1::name()),
            x_tmp: OVector::zeros_generic(f.dim(), U1::name()),
            local_optimizer: NelderMead::new(f, dom),
            iter: 0,
        }
    }

    /// Resets the internal state of the solver.
    pub fn reset(&mut self) {
        self.xs.clear();
        self.ys.clear();
        self.best = 0;
        self.k = F::Scalar::zero();
        self.k_inf = F::Scalar::zero();
        self.iter = 0;
    }

    /// Adds an externally evaluated point into the collection of evaluations.
    ///
    /// This is useful if there is another solver used in combination with LIPO.
    /// If there is an evaluation available from the other method, adding it to
    /// the LIPO solver gives extra information for free.
    pub fn add_evaluation(
        &mut self,
        x: OVector<F::Scalar, F::Dim>,
        y: F::Scalar,
    ) -> Result<(), LipoError> {
        let alpha = self.alpha;

        let Self {
            xs,
            ys,
            k,
            k_inf,
            tmp,
            ..
        } = self;

        if !xs.is_empty() {
            // By definition, the k estimation is done by finding a minimum k_l
            // from a sequence of ks, such that
            //
            //     max_{i != j} |f(x_i) - f(x_j)| / || x_i - x_j || <= k_l
            //
            // This leads to a quadratic algorithm. However, with the knowledge
            // of k previously calculated on x_1, ..., x_t, we can only
            // calculate potential increase of k with respect to the new
            // x_{t+1}.

            for (xi, yi) in xs.iter().zip(ys.iter().copied()) {
                xi.sub_to(&x, tmp);
                let dist = tmp.norm();
                let ki = (yi - y).abs() / dist;

                if ki > *k_inf {
                    *k_inf = ki;
                }

                debug!("|| x - xi || = {}", dist);
            }

            let it = try_convert(((*k_inf).ln() / (F::Scalar::one() + alpha).ln()).ceil())
                .unwrap_or_default() as i32;
            *k = (F::Scalar::one() + alpha).powi(it);

            if !k.is_finite() {
                return Err(LipoError::InfiniteLipschitzConstant);
            }

            debug!("Lipschitz constant k = {}", *k);
        }

        // Add the new point to the evaluated points.
        xs.push(x);
        ys.push(y);

        Ok(())
    }
}

/// Error returned from [`Lipo`] solver.
#[derive(Debug, Error)]
pub enum LipoError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    Problem(#[from] ProblemError),
    /// Error when no potential minimizer is found after number sampling trials.
    #[error("potential minimizer was not found after specified number of trials")]
    PotentialMinimizerNotFound,
    /// Error when estimated Lipschitz constant became infinite.
    #[error("estimated Lipschitz constant became infinite")]
    InfiniteLipschitzConstant,
}

impl<F: Function, R: Rng> Lipo<F, R>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    F::Scalar: SampleUniform,
    Standard: Distribution<F::Scalar>,
{
    fn next_inner<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
    ) -> Result<F::Scalar, LipoError>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
    {
        let LipoOptions {
            sampling_trials,
            potential_minimizer,
            local_optimization_iters,
            ..
        } = self.options;

        let Self {
            xs,
            ys,
            best,
            k,
            rng,
            bernoulli,
            tmp,
            x_tmp,
            local_optimizer,
            iter,
            ..
        } = self;

        if xs.is_empty() {
            debug!("first iteration, just evaluating");
            // First iteration. We just evaluate the initial point and store it.
            let error = f.apply(x)?;

            xs.push(x.clone_owned());
            ys.push(error);

            return Ok(error);
        }

        if local_optimization_iters == 0 || *iter % local_optimization_iters == 0 {
            local_optimizer.reset();
            debug!("number of evaluated points = {}", xs.len());

            sample_uniform::<F, _, _>(x, dom, rng);

            // Generate a few random points in the beginning of the optimization to
            // make the estimation of lower bound sensible.
            let initialization = xs.len() < f.dim().value().max(3);

            // Exploitation mode is allowed only when there is enough points
            // evaluated and the Lipschitz constant is estimated. Then there is
            // randomness involved in choosing whether we explore or exploit.
            if !initialization && *k != F::Scalar::zero() && !bernoulli.sample(rng) {
                debug!("exploitation mode");

                let mut tmp_best = ys[*best];

                let mut found = false;
                for _ in 0..sampling_trials {
                    // Calculate the lower bound by max_i f(x_i) - k * || x - x_i ||.
                    let bound = xs
                        .iter()
                        .zip(ys.iter().copied())
                        .map(|(xi, yi)| {
                            x.sub_to(xi, tmp);
                            let norm = tmp.norm();

                            yi - *k * norm
                        })
                        .reduce(|max, z| if z > max { z } else { max })
                        .unwrap();

                    trace!(
                        "bound <= best: {:?} <= {:?} for {:?}",
                        bound,
                        tmp_best,
                        x.as_slice()
                    );

                    if bound <= tmp_best {
                        found = true;

                        match potential_minimizer {
                            PotentialMinimizer::Any => {
                                break;
                            }
                            PotentialMinimizer::Best => {
                                if found {
                                    trace!("found better potential minimizer");
                                }

                                // Backup the valid point and continue sampling to
                                // try to find a better one.
                                x_tmp.copy_from(x);
                                sample_uniform::<F, _, _>(x, dom, rng);
                                tmp_best = bound;
                                continue;
                            }
                        }
                    } else {
                        trace!("unsuitable point, sample new");
                        sample_uniform::<F, _, _>(x, dom, rng);
                    }
                }

                if !found {
                    debug!(
                        "did not find any potential minimizer after {} sampling trials",
                        sampling_trials
                    );
                    return Err(LipoError::PotentialMinimizerNotFound);
                }

                if potential_minimizer == PotentialMinimizer::Best {
                    // If the "find best" strategy is used, we need to copy the best
                    // point to the vector x.
                    x.copy_from(x_tmp);
                }
            } else {
                debug!("exploration mode");

                // In exploration mode, the point is evaluated and added
                // unconditionally.
            }
        }

        let error = if local_optimization_iters > 0 {
            debug!(
                "local optimization iteration: {}",
                (*iter % local_optimization_iters) + 1
            );

            // Do not fail the optimization on an error from the local
            // optimization.
            match local_optimizer.next(f, dom, x) {
                Ok(error) => error,
                Err(_) => f.apply(x)?,
            }
        } else {
            f.apply(x)?
        };

        // New point is better then the current best, so we update it.
        if error < ys[*best] {
            debug!("global best improved: {} -> {}", ys[*best], error);
            *best = xs.len();
        }

        self.add_evaluation(x.clone_owned(), error)?;

        let Self {
            xs, ys, best, iter, ..
        } = self;

        debug!("sample fx = {}\tbest fx = {}", error, ys[*best]);

        *iter += 1;
        x.copy_from(&xs[*best]);
        Ok(ys[*best])
    }
}

impl<F: Function, R: Rng> Optimizer<F> for Lipo<F, R>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    F::Scalar: SampleUniform,
    Standard: Distribution<F::Scalar>,
{
    const NAME: &'static str = "LIPO";

    type Error = LipoError;

    fn next<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<<F>::Scalar>,
        x: &mut Vector<<F>::Scalar, <F>::Dim, Sx>,
    ) -> Result<<F>::Scalar, Self::Error>
    where
        Sx: StorageMut<<F>::Scalar, <F>::Dim> + IsContiguous,
    {
        self.next_inner(f, dom, x)
    }
}

impl<F: System, R: Rng> Solver<F> for Lipo<F, R>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    F::Scalar: SampleUniform,
    Standard: Distribution<F::Scalar>,
{
    const NAME: &'static str = "LIPO";

    type Error = LipoError;

    fn next<Sx, Sfx>(
        &mut self,
        f: &F,
        dom: &Domain<<F>::Scalar>,
        x: &mut Vector<<F>::Scalar, <F>::Dim, Sx>,
        fx: &mut Vector<<F>::Scalar, <F>::Dim, Sfx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<<F>::Scalar, <F>::Dim> + IsContiguous,
        Sfx: StorageMut<<F>::Scalar, <F>::Dim>,
    {
        self.next_inner(f, dom, x)?;
        f.eval(x, fx)?;
        Ok(())
    }
}

fn sample_uniform<F: Problem, Sx, R: Rng>(
    x: &mut Vector<F::Scalar, F::Dim, Sx>,
    dom: &Domain<F::Scalar>,
    rng: &mut R,
) where
    Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
    F::Scalar: SampleUniform,
    Standard: Distribution<F::Scalar>,
{
    x.iter_mut().zip(dom.vars().iter()).for_each(|(xi, vi)| {
        *xi = if !vi.lower().is_finite() || !vi.upper().is_finite() {
            let random: F::Scalar = rng.gen();

            if vi.lower().is_finite() || vi.upper().is_finite() {
                let clamped = random.max(vi.lower()).min(vi.upper());
                let delta = clamped - random;
                clamped + delta
            } else {
                random
            }
        } else {
            Uniform::new_inclusive(vi.lower(), vi.upper()).sample(rng)
        };
    });
}

#[cfg(test)]
mod tests {
    use rand::{rngs::SmallRng, SeedableRng};

    use super::*;

    use crate::testing::*;

    #[test]
    fn sphere_optimization() {
        let n = 2;

        let f = Sphere::new(n);
        let dom = f.domain();
        let eps = convert(1e-3);
        let rng = SmallRng::from_seed([3; 32]);
        let mut options = LipoOptions::default();
        options
            .set_local_optimization_iters(0)
            .set_potential_minimizer(PotentialMinimizer::Any);

        for x in f.initials() {
            let optimizer = Lipo::with_options(&f, &dom, options.clone(), rng.clone());
            optimize(&f, &dom, optimizer, x, convert(0.0), 250, eps).unwrap();
        }
    }
}