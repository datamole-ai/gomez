//! Trust region optimization method.
//!
//! More than a particular algorithm, [trust
//! region](https://en.wikipedia.org/wiki/Trust_region) methods is actually sort
//! of a framework of various techniques. This also applies to the
//! implementation of this method in gomez; it is composed of multiple
//! techniques that are applied in specific cases. The basis is [Powell's dogleg
//! method](https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method), while a
//! variant of
//! [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
//! method is used when using newton direction is not possible.
//!
//! # References
//!
//! \[1\] [Numerical
//! Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5)
//!
//! \[2\] [Methods for Non-Linear Least Squares
//! Problems](https://api.semanticscholar.org/CorpusID:64217935)
//!
//! \[3\] [Numerical Methods for Unconstrained Optimization and Nonlinear
//! Equations](https://epubs.siam.org/doi/book/10.1137/1.9781611971200)
//!
//! \[4\] [A Modified Two Steps Levenberg-Marquardt Method for Nonlinear
//! Equations](https://www.sciencedirect.com/science/article/pii/S0377042715002666)
//!
//! \[5\] [Implementation of scaled Hybrid algorithm in
//! GSL](https://git.savannah.gnu.org/cgit/gsl.git/tree/multiroots)

use getset::{CopyGetters, Setters};
use log::debug;
use nalgebra::{
    allocator::{Allocator, Reallocator},
    convert,
    storage::StorageMut,
    ComplexField, DefaultAllocator, Dim, DimMin, DimName, OMatrix, OVector, RealField, Vector, U1,
};
use num_traits::{One, Zero};
use thiserror::Error;

use crate::{
    core::{Domain, Error, Problem, Solver, System, VectorDomainExt},
    derivatives::{Jacobian, JacobianError, EPSILON_SQRT},
};

/// Specification for initial value of trust region size.
#[derive(Debug, Clone, Copy)]
pub enum DeltaInit<S> {
    /// Fixed value.
    Fixed(S),
    /// Estimated from Jacobian matrix in the initial point.
    Estimated,
}

/// Options for [`TrustRegion`] solver.
#[derive(Debug, Clone, CopyGetters, Setters)]
#[getset(get_copy = "pub", set = "pub")]
pub struct TrustRegionOptions<F: Problem> {
    /// Minimum allowed trust region size. Default: `f64::EPSILON.sqrt()`.
    delta_min: F::Scalar,
    /// Maximum allowed trust region size. Default: `1e9`.
    delta_max: F::Scalar,
    /// Initial trust region size. Default: estimated (see [`DeltaInit`]).
    delta_init: DeltaInit<F::Scalar>,
    /// Minimum scaling factor for lambda in Levenberg-Marquardt step. Default:
    /// `1e-10`.
    mu_min: F::Scalar,
    /// Threshold for gain ratio to shrink trust region size if lower. Default:
    /// `0.25`.
    shrink_thresh: F::Scalar,
    /// Threshold for gain ratio to expand trust region size if higher. Default:
    /// `0.75`.
    expand_thresh: F::Scalar,
    /// Threshold for gain ratio that needs to be exceeded to accept the
    /// calculated step. Default: `0.0001`.
    accept_thresh: F::Scalar,
    /// Number of step rejections that are allowed to happen before returning
    /// [`TrustRegionError::NoProgress`] error. Default: `10`.
    rejections_thresh: usize,
    /// Determines whether steps that increase the error can be accepted.
    /// Default: `true`.
    allow_ascent: bool,
}

impl<F: Problem> Default for TrustRegionOptions<F> {
    fn default() -> Self {
        Self {
            delta_min: convert(EPSILON_SQRT),
            delta_max: convert(1e9),
            delta_init: DeltaInit::Estimated,
            mu_min: convert(1e-10),
            shrink_thresh: convert(0.25),
            expand_thresh: convert(0.75),
            accept_thresh: convert(0.0001),
            rejections_thresh: 10,
            allow_ascent: true,
        }
    }
}

/// Trust region solver. See [module](self) documentation for more details.
pub struct TrustRegion<F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    options: TrustRegionOptions<F>,
    delta: F::Scalar,
    mu: F::Scalar,
    scale: OVector<F::Scalar, F::Dim>,
    jac: Jacobian<F>,
    q_tr_fx_neg: OVector<F::Scalar, F::Dim>,
    newton: OVector<F::Scalar, F::Dim>,
    grad_neg: OVector<F::Scalar, F::Dim>,
    cauchy: OVector<F::Scalar, F::Dim>,
    jac_tr_jac: OMatrix<F::Scalar, F::Dim, F::Dim>,
    p: OVector<F::Scalar, F::Dim>,
    temp: OVector<F::Scalar, F::Dim>,
    iter: usize,
    rejections_cnt: usize,
}

impl<F: Problem> TrustRegion<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    /// Initializes trust region solver with default options.
    pub fn new(f: &F, dom: &Domain<F::Scalar>) -> Self {
        Self::with_options(f, dom, TrustRegionOptions::default())
    }

    /// Initializes trust region solver with given options.
    pub fn with_options(f: &F, dom: &Domain<F::Scalar>, options: TrustRegionOptions<F>) -> Self {
        let delta_init = match options.delta_init {
            DeltaInit::Fixed(fixed) => fixed,
            // Zero is recognized in the function `next`.
            DeltaInit::Estimated => F::Scalar::zero(),
        };
        let scale_iter = dom.vars().iter().map(|var| var.scale());

        Self {
            options,
            delta: delta_init,
            mu: convert(0.5),
            scale: OVector::from_iterator_generic(f.dim(), U1::name(), scale_iter),
            jac: Jacobian::zeros(f),
            q_tr_fx_neg: OVector::zeros_generic(f.dim(), U1::name()),
            newton: OVector::zeros_generic(f.dim(), U1::name()),
            grad_neg: OVector::zeros_generic(f.dim(), U1::name()),
            cauchy: OVector::zeros_generic(f.dim(), U1::name()),
            jac_tr_jac: OMatrix::zeros_generic(f.dim(), f.dim()),
            p: OVector::zeros_generic(f.dim(), U1::name()),
            temp: OVector::zeros_generic(f.dim(), U1::name()),
            iter: 1,
            rejections_cnt: 0,
        }
    }
}

/// Error returned from [`TrustRegion`] solver.
#[derive(Debug, Error)]
pub enum TrustRegionError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    Problem(#[from] Error),
    /// Error that occurred when computing the Jacobian matrix.
    #[error("{0}")]
    Jacobian(#[from] JacobianError),
    /// Could not take any valid step.
    #[error("neither newton nor steepest descent step can be taken from the point")]
    NoValidStep,
    /// Maximum number of step rejections exceeded.
    #[error("cannot make progress")]
    NoProgress,
}

impl<F: System> Solver<F> for TrustRegion<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    F::Dim: DimMin<F::Dim, Output = F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, <F::Dim as DimMin<F::Dim>>::Output>,
    DefaultAllocator: Reallocator<F::Scalar, F::Dim, F::Dim, F::Dim, F::Dim>,
{
    const NAME: &'static str = "Trust-region";

    type Error = TrustRegionError;

    fn next<Sx, Sfx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        fx: &mut Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<F::Scalar, F::Dim>,
        Sfx: StorageMut<F::Scalar, F::Dim>,
    {
        let TrustRegionOptions {
            delta_min,
            delta_max,
            mu_min,
            shrink_thresh,
            expand_thresh,
            accept_thresh,
            rejections_thresh,
            allow_ascent,
            ..
        } = self.options;

        let Self {
            delta,
            mu,
            scale,
            jac,
            q_tr_fx_neg,
            newton,
            grad_neg,
            cauchy,
            jac_tr_jac,
            p,
            temp,
            iter,
            rejections_cnt,
            ..
        } = self;

        let scaled_newton: &mut OVector<F::Scalar, F::Dim>;
        let scale_inv2: &mut OVector<F::Scalar, F::Dim>;
        let cauchy_scaled: &mut OVector<F::Scalar, F::Dim>;

        #[derive(Clone, Copy, PartialEq)]
        enum StepType {
            FullNewton,
            ScaledNewton,
            LevenbergMarquardt,
            ScaledCauchy,
            Dogleg,
        }

        // Compute F(x) and F'(x).
        f.eval(x, fx)?;
        jac.compute(f, x, scale, fx)?;

        let fx_norm = fx.norm();

        let estimate_delta = *delta == F::Scalar::zero();
        if estimate_delta {
            // Zero delta signifies that the initial delta is to be set
            // automatically and it has not been done yet.
            //
            // The initial delta is estimated as follows. Let vector d be
            // defined as
            //
            //     d_j = || F'(x)_*j || or 1 if it would be 0
            //
            // Then delta = K * || diag(d) x || or K if || diag(d) x || = 0,
            // where K = 100. The approach is taken from GSL.
            for (j, col) in jac.column_iter().enumerate() {
                temp[j] = col.norm();
                if temp[j] == F::Scalar::zero() {
                    temp[j] = F::Scalar::one();
                }
            }
            temp.component_mul_assign(x);

            let factor = convert(100.0);
            *delta = temp.norm() * factor;

            if *delta == F::Scalar::zero() {
                *delta = factor;
            }
        }

        // Perform QR decomposition of F'(x).
        let (q, r) = jac.clone_owned().qr().unpack();

        // Compute -Q^T F(x).
        q.tr_mul_to(fx, q_tr_fx_neg);
        q_tr_fx_neg.neg_mut();

        // Find the Newton step by solving the system R newton = -Q^T F(x).
        newton.copy_from(q_tr_fx_neg);
        let is_newton_valid = r.solve_upper_triangular_mut(newton);

        if !is_newton_valid {
            // TODO: Moore-Penrose pseudoinverse?
            debug!(
                "Newton step is invalid for ill-defined Jacobian (zero columns: {:?})",
                jac.column_iter()
                    .enumerate()
                    .filter(|(_, col)| col.norm() == F::Scalar::zero())
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            );
        }

        // Compute the norm of scaled Newton step (use temp fo storage).
        scaled_newton = temp;
        scaled_newton.copy_from(newton);
        scaled_newton.component_mul_assign(scale);
        let newton_scaled_norm = scaled_newton.norm();

        let step_type = if is_newton_valid && newton_scaled_norm <= *delta {
            // Scaled Newton step is inside the trust region. We can safely take it.
            p.copy_from(newton);
            debug!("take full Newton: {:?}", p.as_slice());
            StepType::FullNewton
        } else {
            // Newton step is outside the trust region. We need to involve the
            // gradient.

            // Compute -grad F(x) = -F'(x)^T F(x) = -R^T Q^T F(x).
            r.tr_mul_to(q_tr_fx_neg, grad_neg);

            let grad_norm = grad_neg.norm();

            if grad_norm == F::Scalar::zero() {
                // Gradient is zero, it is useless to compute the dogleg step.
                // Instead, we take the Newton direction to the trust region
                // boundary.
                if is_newton_valid {
                    p.copy_from(newton);
                    *p *= *delta / newton_scaled_norm;
                    debug!(
                        "take scaled Newton to trust-region boundary: {:?}",
                        p.as_slice()
                    );
                    StepType::ScaledNewton
                } else {
                    return Err(TrustRegionError::NoValidStep);
                }
            } else {
                // Compute D^(-2) (use p for storage).
                scale_inv2 = p;
                scale_inv2.copy_from(scale);
                scale_inv2.apply(|s| F::Scalar::one() / (s * s));

                // Compute g = -D^(-2) grad F, the steepest descent direction in
                // scaled space (use cauchy for storage).
                cauchy.copy_from(scale_inv2);
                cauchy.component_mul_assign(grad_neg);
                let p = scale_inv2;

                // Compute tau = -(grad F)^T g / || F'(x) g ||^2.
                jac.mul_to(cauchy, temp);
                let jac_g_norm2 = temp.norm_squared();
                let grad_neg_g = grad_neg.dot(cauchy);
                let tau = grad_neg_g / jac_g_norm2;

                // Scale the steepest descent to the Cauchy point.
                *cauchy *= tau;

                // Compute ||D cauchy||.
                cauchy_scaled = temp;
                cauchy_scaled.copy_from(scale);
                cauchy_scaled.component_mul_assign(cauchy);
                let cauchy_scaled_norm = cauchy_scaled.norm();

                if cauchy_scaled_norm >= *delta {
                    // Cauchy point is outside the trust region. We take the
                    // steepest gradient descent to the trust region boundary.
                    p.copy_from(cauchy);
                    *p *= *delta / cauchy_scaled_norm;
                    debug!(
                        "take scaled Cauchy to trust region-boundary: {:?}",
                        p.as_slice()
                    );
                    StepType::ScaledCauchy
                } else if is_newton_valid {
                    let temp = cauchy_scaled;

                    // The trust region boundary is crossed by the dogleg path
                    // p(alpha) = cauchy + alpha (newton - cauchy). We need to
                    // find alpha such that || D p || = delta. It is found by
                    // solving the following quadratic equation:
                    //
                    //     || D p ||^2 - delta^2 = 0
                    //
                    // For equation a alpha^2 + 2b alpha + c = 0, we get:
                    //
                    //     a = || D (newton - cauchy) ||^2
                    //     b = cauchy^T D^2 (newton - cauchy)
                    //     c = || D cauchy ||^2 - delta^2
                    //
                    // This polynomial has one negative root and one root in [0,
                    // 1]. We seek for the latter. Due to our choice of the
                    // polynomial, we have
                    //
                    //     roots = (-b +- sqrt(b^2 - ac)) / a
                    //
                    // We can observe that, due to || D cauchy || < delta that
                    // we checked above, c is always negative. Further, a is
                    // always nonnegative. Therefore sqrt(b^2 - ac) >= b. That
                    // means that for whatever b, the root when using minus sign
                    // is negative, which is not what we seek for. Thus we can
                    // safely compute only one root.
                    //
                    // For slightly better numerical accuracy, we will avoid
                    // some subtractions (possible catastrophic cancellation) by
                    // computing -c and using Muller's formula for b > 0.

                    // Compute D (newton - cauchy) (use p for storage).
                    let diff_scaled = p;
                    newton.sub_to(cauchy, diff_scaled);
                    diff_scaled.component_mul_assign(scale);

                    // Compute a, b and -c.
                    let a = diff_scaled.norm_squared();

                    temp.copy_from(diff_scaled);
                    temp.component_mul_assign(scale);
                    let b = cauchy.dot(temp);

                    let c_neg = *delta * *delta - cauchy_scaled_norm * cauchy_scaled_norm;

                    #[allow(clippy::suspicious_operation_groupings)]
                    let d = (b * b + a * c_neg).sqrt();
                    let alpha = if b <= F::Scalar::zero() {
                        (-b + d) / a
                    } else {
                        c_neg / (b + d)
                    };
                    let p = diff_scaled;

                    // Finally, compute the dogleg step p = cauchy + alpha
                    // (newton - cauchy).
                    newton.sub_to(cauchy, p);
                    *p *= alpha;
                    *p += &*cauchy;
                    debug!("take dogleg (factor = {}): {:?}", alpha, p.as_slice());
                    StepType::Dogleg
                } else {
                    let temp = cauchy_scaled;

                    // Since F'(x) cannot be inverted so the Newton step is
                    // undefined, we need to fallback to Levenberg-Marquardt
                    // which overcomes this issue but at higher computational
                    // expense. We are looking for lambda such that
                    //
                    //     (B + lambda I) p = - grad F(x)
                    //
                    // such that p is the solution to min m(p) with || D p || <=
                    // delta.

                    // Determine lambda for Levenberg-Marquardt. A common choice
                    // proven to lead to quadratic convergence is
                    //
                    //     lambda = || F(x) ||^d,
                    //
                    // where d is from (0, 2]. An adaptive choice for d is:
                    //
                    //     d = 1 / || F(x) || if || F(x) || >= 1 and 1 + 1 / k otherwise,
                    //
                    // where k denotes the current iteration. Such choice
                    // ensures that lambda is not large when the point is far
                    // from the solution (i.e., for large || F(x) ||).

                    // Determine lambda.
                    let d = if fx_norm >= F::Scalar::one() {
                        F::Scalar::one() / fx_norm
                    } else {
                        F::Scalar::one() + F::Scalar::one() / convert(*iter as f64)
                    };

                    let lambda = *mu * fx_norm.powf(d);

                    // Compute B = F'(x)^T F'(x), which is a symmetric matrix.
                    jac.tr_mul_to(jac, jac_tr_jac);

                    // Compute B + lambda I.
                    let jac_tr_jac_lambda = jac_tr_jac;
                    for i in 0..f.dim().value() {
                        jac_tr_jac_lambda[(i, i)] += lambda;
                    }

                    // Solve p for (B + lambda I) p = - grad F(x).
                    p.copy_from(grad_neg);

                    let is_levenberg_marquardt_valid =
                        jac_tr_jac_lambda.clone_owned().qr().solve_mut(p);

                    if !is_levenberg_marquardt_valid {
                        debug!(
                            "Levenberg-Marquardt step is invalid for ill-defined matrix B (lambda = {})",
                            lambda
                        );
                    }

                    // Scale p to be in the trust region, i.e., || D p || <=
                    // delta.
                    let p_scaled = temp;
                    p_scaled.copy_from(scale);
                    p_scaled.component_mul_assign(p);

                    let p_scaled_norm = p_scaled.norm();

                    if p_scaled_norm > *delta {
                        // The original step was outside, scale it to the
                        // boundary.
                        *p *= *delta / p_scaled_norm;
                    }

                    debug!(
                        "take Levenberg-Marquardt (lambda = {}): {:?}",
                        lambda,
                        p.as_slice()
                    );
                    StepType::LevenbergMarquardt
                }
            }
        };

        // Vectors for Newton and Cauchy steps are no longed used, so we reuse
        // their allocations for another purpose.
        let x_trial = newton;
        let fx_trial = cauchy;

        // Get candidate x' for the next iterate.
        x.add_to(p, x_trial);

        let not_feasible = x_trial.project(dom);

        if not_feasible {
            debug!("new iterate is not feasible, performing the projection");

            // Compute the step after projection.
            x_trial.sub_to(x, p);
        }

        // Compute F(x').
        let is_trial_valid = f.eval(x_trial, fx_trial).is_ok();
        let fx_trial_norm = fx_trial.norm();

        let gain_ratio = if is_trial_valid {
            // Compute the gain ratio.
            jac.mul_to(p, temp);
            *temp += &*fx;
            let predicted = fx_norm - temp.norm();

            let deny = if allow_ascent {
                // If ascent is allowed, then check only for zero, which would
                // make the gain ratio calculation ill-defined.
                predicted == F::Scalar::zero()
            } else {
                // If ascent is not allowed, test positivity of the predicted
                // gain. Note that even if the actual reduction was positive,
                // the step would be rejected anyway because the gain ratio
                // would be negative.
                predicted <= F::Scalar::zero()
            };

            let gain_ratio = if deny {
                if allow_ascent {
                    debug!("predicted gain = 0");
                } else {
                    debug!("predicted gain <= 0");
                }
                F::Scalar::zero()
            } else {
                let actual = fx_norm - fx_trial_norm;
                let gain_ratio = actual / predicted;
                debug!("gain ratio = {} / {} = {}", actual, predicted, gain_ratio);

                gain_ratio
            };

            gain_ratio
        } else {
            debug!("trial step is invalid, gain ratio = 0");
            F::Scalar::zero()
        };

        // Decide if the step is accepted or not.
        if gain_ratio > accept_thresh {
            // Accept the trial step.
            x.copy_from(x_trial);
            fx.copy_from(fx_trial);
            debug!(
                "step accepted, || fx || = {}, x = {:?}",
                fx_trial_norm,
                x_trial.as_slice()
            );

            *rejections_cnt = 0;
        } else {
            debug!("step rejected, threshold for accepting = {}", accept_thresh);
            *rejections_cnt += 1;

            if *rejections_cnt == rejections_thresh {
                debug!(
                    "solving reached the rejections count limit ({})",
                    rejections_thresh
                );
                return Err(TrustRegionError::NoProgress);
            }
        }

        let p_scaled = p;
        p_scaled.component_mul_assign(scale);
        let p_scaled_norm = p_scaled.norm();

        // Potentially update the size of the trust region.
        let delta_old = *delta;
        if gain_ratio < shrink_thresh {
            *delta = (delta_old * convert(0.25))
                .min(p_scaled_norm * convert(0.25))
                .max(delta_min);
            debug!(
                "shrink delta from {} to {} (|| D p || = {})",
                delta_old, *delta, p_scaled_norm
            );
        } else if gain_ratio > expand_thresh {
            *delta = (delta_old * convert(2.0))
                .max(p_scaled_norm * convert(3.0))
                .min(delta_max);
            debug!(
                "expand delta from {} to {} (|| D p || = {})",
                delta_old, *delta, p_scaled_norm
            );
        }

        // Potentially update the mu parameter for LM method.
        if step_type == StepType::LevenbergMarquardt {
            let mu_old = *mu;

            // Note that shrinkage and expansion are reversed for mu compared to
            // delta. The less mu is, the more LM step exploits the information
            // from Jacobian, because it is less "deformed" by adding lambda I.
            // Thus, for successful steps, we want to decrease lambda by
            // decreasing mu and vice versa for bad steps.
            if gain_ratio < shrink_thresh {
                *mu = mu_old * convert(4.0);
                debug!("expand mu from {} to {}", mu_old, *mu);
            } else if gain_ratio > expand_thresh {
                *mu = (mu_old * convert(0.25)).max(mu_min);
                debug!("shrink mu from {} to {}", mu_old, *mu);
            }
        }

        *iter += 1;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::*;

    #[test]
    fn rosenbrock() {
        let n = 2;

        let f = ExtendedRosenbrock::new(n);
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let solver = TrustRegion::new(&f, &dom);
            f.is_root(&solve(&f, &dom, solver, x, 25, eps).unwrap(), eps);
        }
    }

    #[test]
    fn rosenbrock_scaling() {
        let n = 2;
        let alpha = 100.0;

        let f = ExtendedRosenbrock::with_scaling(n, alpha);
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let solver = TrustRegion::new(&f, &dom);
            f.is_root(&solve(&f, &dom, solver, x, 50, eps).unwrap(), eps);
        }
    }

    #[test]
    fn considering_domain() {
        let f = BullardBiegler::new();
        let dom = f.domain();

        let x = nalgebra::vector![1.0, 1.0];
        let solver = TrustRegion::new(&f, &dom);
        let eps = convert(1e-12);

        // From this initial point, dogleg converges to a point which is outside
        // the bounds.
        assert!(matches!(
            solve(&f, &dom, solver, x, 50, eps),
            Err(SolveError::Solver(TrustRegionError::NoProgress)) | Err(SolveError::Termination)
        ));
    }
}
