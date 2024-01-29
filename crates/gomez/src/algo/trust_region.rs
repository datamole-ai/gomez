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
    convert, storage::StorageMut, ComplexField, DimName, Dyn, IsContiguous, OMatrix, OVector,
    RealField, Vector, U1,
};
use thiserror::Error;

use crate::{
    core::{Domain, Function, Optimizer, Problem, RealField as _, Solver, System},
    derivatives::{Gradient, Hessian, Jacobian, StepRule},
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
pub struct TrustRegionOptions<P: Problem> {
    /// Minimum allowed trust region size. Default: `f64::EPSILON.sqrt()`.
    delta_min: P::Field,
    /// Maximum allowed trust region size. Default: `1e9`.
    delta_max: P::Field,
    /// Initial trust region size. Default: estimated (see [`DeltaInit`]).
    delta_init: DeltaInit<P::Field>,
    /// Minimum scaling factor for lambda in Levenberg-Marquardt step. Default:
    /// `1e-10`.
    mu_min: P::Field,
    /// Threshold for gain ratio to shrink trust region size if lower. Default:
    /// `0.25`.
    shrink_thresh: P::Field,
    /// Threshold for gain ratio to expand trust region size if higher. Default:
    /// `0.75`.
    expand_thresh: P::Field,
    /// Threshold for gain ratio that needs to be exceeded to accept the
    /// calculated step. Default: `0.0001`.
    accept_thresh: P::Field,
    /// Number of step rejections that are allowed to happen before returning
    /// [`TrustRegionError::NoProgress`] error. Default: `10`.
    rejections_thresh: usize,
    /// Determines whether steps that increase the error can be accepted.
    /// Default: `true`.
    allow_ascent: bool,
    /// Relative epsilon used in Jacobian and gradient computations.
    /// Default: `sqrt(EPSILON)`.
    eps_sqrt: P::Field,
    /// Relative epsilon used in Hessian computations.
    /// Default: `cbrt(EPSILON)`.
    eps_cbrt: P::Field,
    /// Step rule used in Jacobian, gradient and Hessian computations.
    step_rule: StepRule,
    #[getset(skip)]
    prefer_greater_magnitude_in_cauchy: bool,
}

impl<P: Problem> TrustRegionOptions<P> {
    // XXX: This is a hack. Setting this to true influences the solver in a way
    // that is not based on mathematical theory. However, when used, this makes
    // the solver to work substantially better for specific cases. I can't
    // explain why, that's why setting is hidden. If I manage to find the
    // explanation or the real reason why this helps, I will either remove this,
    // offer a proper setting or adjust the solver implementation accordingly.
    #[doc(hidden)]
    pub fn set_prefer_greater_magnitude_in_cauchy(
        &mut self,
        prefer_greater_magnitude_in_cauchy: bool,
    ) -> &mut Self {
        self.prefer_greater_magnitude_in_cauchy = prefer_greater_magnitude_in_cauchy;
        self
    }
}

impl<P: Problem> Default for TrustRegionOptions<P> {
    fn default() -> Self {
        Self {
            delta_min: P::Field::EPSILON_SQRT,
            delta_max: convert(1e9),
            delta_init: DeltaInit::Estimated,
            mu_min: convert(1e-10),
            shrink_thresh: convert(0.25),
            expand_thresh: convert(0.75),
            accept_thresh: convert(0.0001),
            rejections_thresh: 10,
            allow_ascent: true,
            prefer_greater_magnitude_in_cauchy: false,
            eps_sqrt: P::Field::EPSILON_SQRT,
            eps_cbrt: P::Field::EPSILON_CBRT,
            step_rule: StepRule::default(),
        }
    }
}

/// Trust region solver.
///
/// See [module](self) documentation for more details.
pub struct TrustRegion<P: Problem> {
    options: TrustRegionOptions<P>,
    delta: P::Field,
    mu: P::Field,
    scale: OVector<P::Field, Dyn>,
    jac: Jacobian<P>,
    grad: Gradient<P>,
    hes: Hessian<P>,
    q_tr_rx_neg: OVector<P::Field, Dyn>,
    newton: OVector<P::Field, Dyn>,
    grad_neg: OVector<P::Field, Dyn>,
    cauchy: OVector<P::Field, Dyn>,
    jac_tr_jac: OMatrix<P::Field, Dyn, Dyn>,
    p: OVector<P::Field, Dyn>,
    temp: OVector<P::Field, Dyn>,
    iter: usize,
    rejections_cnt: usize,
}

impl<P: Problem> TrustRegion<P> {
    /// Initializes trust region solver with default options.
    pub fn new(p: &P, dom: &Domain<P::Field>) -> Self {
        Self::with_options(p, dom, TrustRegionOptions::default())
    }

    /// Initializes trust region solver with given options.
    pub fn with_options(p: &P, dom: &Domain<P::Field>, options: TrustRegionOptions<P>) -> Self {
        let dim = Dyn(dom.dim());
        let delta_init = match options.delta_init {
            DeltaInit::Fixed(fixed) => fixed,
            // Zero is recognized in the function `next`.
            DeltaInit::Estimated => convert(0.0),
        };
        let scale = dom
            .scale()
            .map(|scale| OVector::from_iterator_generic(dim, U1::name(), scale.iter().copied()))
            .unwrap_or_else(|| OVector::from_element_generic(dim, U1::name(), convert(1.0)));

        Self {
            options,
            delta: delta_init,
            mu: convert(0.5),
            scale,
            jac: Jacobian::zeros(p),
            grad: Gradient::zeros(p),
            hes: Hessian::zeros(p),
            q_tr_rx_neg: OVector::zeros_generic(dim, U1::name()),
            newton: OVector::zeros_generic(dim, U1::name()),
            grad_neg: OVector::zeros_generic(dim, U1::name()),
            cauchy: OVector::zeros_generic(dim, U1::name()),
            jac_tr_jac: OMatrix::zeros_generic(dim, dim),
            p: OVector::zeros_generic(dim, U1::name()),
            temp: OVector::zeros_generic(dim, U1::name()),
            iter: 1,
            rejections_cnt: 0,
        }
    }

    /// Resets the internal state of the solver.
    pub fn reset(&mut self) {
        self.delta = match self.options.delta_init {
            DeltaInit::Fixed(fixed) => fixed,
            DeltaInit::Estimated => convert(0.0),
        };
        self.mu = convert(0.5);
        self.iter = 1;
        self.rejections_cnt = 0;
    }
}

/// Error returned from [`TrustRegion`] solver.
#[derive(Debug, Error)]
pub enum TrustRegionError {
    /// Could not take any valid step.
    #[error("neither newton nor steepest descent step can be taken from the point")]
    NoValidStep,
    /// Maximum number of step rejections exceeded.
    #[error("cannot make progress")]
    NoProgress,
}

impl<R: System> Solver<R> for TrustRegion<R> {
    const NAME: &'static str = "Trust-region";

    type Error = TrustRegionError;

    fn solve_next<Sx, Srx>(
        &mut self,
        r: &R,
        dom: &Domain<R::Field>,
        x: &mut Vector<R::Field, Dyn, Sx>,
        rx: &mut Vector<R::Field, Dyn, Srx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<R::Field, Dyn> + IsContiguous,
        Srx: StorageMut<R::Field, Dyn>,
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
            prefer_greater_magnitude_in_cauchy,
            eps_sqrt,
            step_rule,
            ..
        } = self.options;

        let Self {
            delta,
            mu,
            scale,
            jac,
            q_tr_rx_neg,
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

        let one = convert(1.0);
        let zero = convert(0.0);

        #[allow(clippy::needless_late_init)]
        let scaled_newton: &mut OVector<R::Field, Dyn>;
        let scale_inv2: &mut OVector<R::Field, Dyn>;
        let cauchy_scaled: &mut OVector<R::Field, Dyn>;

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum StepType {
            FullNewton,
            ScaledNewton,
            LevenbergMarquardt,
            ScaledCauchy,
            Dogleg,
        }

        // Compute r(x) and r'(x).
        r.eval(x, rx);
        jac.compute(r, x, scale, rx, eps_sqrt, step_rule);

        let rx_norm = rx.norm();

        let estimate_delta = *delta == zero;
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
                if temp[j] == zero {
                    temp[j] = one;
                }
            }
            temp.component_mul_assign(x);

            let factor = convert(100.0);
            *delta = temp.norm() * factor;

            if *delta == zero {
                *delta = factor;
            }
        }

        // Perform QR decomposition of F'(x).
        let (qr_q, qr_r) = jac.clone_owned().qr().unpack();

        // Compute -Q^T r(x).
        qr_q.tr_mul_to(rx, q_tr_rx_neg);
        q_tr_rx_neg.neg_mut();

        // Find the Newton step by solving the system R newton = -Q^T r(x).
        newton.copy_from(q_tr_rx_neg);
        let is_newton_valid = qr_r.solve_upper_triangular_mut(newton);

        if !is_newton_valid {
            // TODO: Moore-Penrose pseudoinverse?
            debug!(
                "Newton step is invalid for ill-defined Jacobian (zero columns: {:?})",
                jac.column_iter()
                    .enumerate()
                    .filter(|(_, col)| col.norm() == zero)
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

            // Compute -grad r(x) = -r'(x)^T r(x) = -R^T Q^T r(x).
            qr_r.tr_mul_to(q_tr_rx_neg, grad_neg);

            let grad_norm = grad_neg.norm();

            if grad_norm == zero {
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
                scale_inv2.apply(|s| *s = one / (*s * *s));

                // Compute g = -D^(-2) grad F, the steepest descent direction in
                // scaled space (use cauchy for storage).
                cauchy.copy_from(scale_inv2);
                cauchy.component_mul_assign(grad_neg);
                let p = scale_inv2;

                // Compute tau = -(grad r)^T g / || r'(x) g ||^2.
                jac.mul_to(cauchy, temp);
                let jac_g_norm2 = temp.norm_squared();
                let grad_neg_g = if !prefer_greater_magnitude_in_cauchy {
                    grad_neg.dot(cauchy)
                } else {
                    // XXX: By accident/misunderstanding, I discovered that
                    // using tau = g^T g / || F'(x) g ||^2 helps in some
                    // specific cases significantly. This is however not based
                    // on the theory. Very likely it just happens to cause some
                    // side-effect that is the actual reason why it works. This
                    // should be eventually explained or replaced by a proper
                    // solution.
                    //
                    // Note that by using g = -D^(-2) grad F instead of -grad F,
                    // the numerator amplifies the effect of variables that have
                    // greater magnitude (the inverse of scaling matrix), hence
                    // the name of the setting.
                    cauchy.dot(cauchy)
                };
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
                    // This polynomial has one negative root and one root in
                    // range (0, 1). We seek for the latter. Due to our choice
                    // of the polynomial, we have
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
                    let alpha = if b <= zero {
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
                    //     (B + lambda I) p = - grad r(x)
                    //
                    // such that p is the solution to
                    //
                    //     min 1/2 || r'(x) p + r(x) ||^2 s.t. || D p || <= delta.
                    //
                    // Determine lambda for Levenberg-Marquardt. A common choice
                    // proven to lead to quadratic convergence is
                    //
                    //     lambda = || r(x) ||^d,
                    //
                    // where d is from (0, 2]. An adaptive choice for d is:
                    //
                    //     d = 1 / || r(x) || if || r(x) || >= 1 and 1 + 1 / k otherwise,
                    //
                    // where k denotes the current iteration. Such choice
                    // ensures that lambda is not large when the point is far
                    // from the solution (i.e., for large || r(x) ||).

                    // Determine lambda.
                    let d = if rx_norm >= one {
                        one / rx_norm
                    } else {
                        one + one / convert(*iter as f64)
                    };

                    let lambda = *mu * rx_norm.powf(d);

                    // Compute B = F'(x)^T F'(x), which is a symmetric matrix.
                    jac.tr_mul_to(jac, jac_tr_jac);

                    // Compute B + lambda I.
                    let jac_tr_jac_lambda = jac_tr_jac;
                    for i in 0..dom.dim() {
                        jac_tr_jac_lambda[(i, i)] += lambda;
                    }

                    // Solve p for (B + lambda I) p = - grad r(x).
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
        let rx_trial = cauchy;

        // Get candidate x' for the next iterate.
        x.add_to(p, x_trial);

        let not_feasible = dom.project(x_trial);

        if not_feasible {
            debug!("new iterate is not feasible, performing the projection");

            // Compute the step after projection.
            x_trial.sub_to(x, p);
        }

        // Compute r(x').
        r.eval(x_trial, rx_trial);
        let is_trial_valid = rx_trial.iter().all(|rix| rix.is_finite());
        let rx_trial_norm = rx_trial.norm();

        let gain_ratio = if is_trial_valid {
            // Compute the gain ratio.
            jac.mul_to(p, temp);
            *temp += &*rx;
            let predicted = rx_norm - temp.norm();

            let deny = if allow_ascent {
                // If ascent is allowed, then check only for zero, which would
                // make the gain ratio calculation ill-defined.
                predicted == zero
            } else {
                // If ascent is not allowed, test positivity of the predicted
                // gain. Note that even if the actual reduction was positive,
                // the step would be rejected anyway because the gain ratio
                // would be negative.
                predicted <= zero
            };

            if deny {
                if allow_ascent {
                    debug!("predicted gain = 0");
                } else {
                    debug!("predicted gain <= 0");
                }
                zero
            } else {
                let actual = rx_norm - rx_trial_norm;
                let gain_ratio = actual / predicted;
                debug!("gain ratio = {} / {} = {}", actual, predicted, gain_ratio);

                gain_ratio
            }
        } else {
            debug!("trial step is invalid, gain ratio = 0");
            zero
        };

        // Decide if the step is accepted or not.
        if gain_ratio > accept_thresh {
            // Accept the trial step.
            x.copy_from(x_trial);
            rx.copy_from(rx_trial);
            debug!(
                "step accepted, || rx || = {}, x = {:?}",
                rx_trial_norm,
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

impl<F: Function> Optimizer<F> for TrustRegion<F> {
    const NAME: &'static str = "Trust-region";

    type Error = TrustRegionError;

    fn opt_next<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<<F>::Field>,
        x: &mut Vector<<F>::Field, Dyn, Sx>,
    ) -> Result<<F>::Field, Self::Error>
    where
        Sx: StorageMut<<F>::Field, Dyn> + IsContiguous,
    {
        let TrustRegionOptions {
            delta_min,
            delta_max,
            shrink_thresh,
            expand_thresh,
            accept_thresh,
            rejections_thresh,
            allow_ascent,
            eps_sqrt,
            eps_cbrt,
            step_rule,
            ..
        } = self.options;

        let Self {
            delta,
            scale,
            hes,
            grad,
            q_tr_rx_neg: q_tr_grad_neg,
            newton,
            grad_neg,
            cauchy,
            p,
            temp,
            rejections_cnt,
            ..
        } = self;

        let zero = convert(0.0);
        let one = convert(1.0);

        #[allow(clippy::needless_late_init)]
        let scaled_newton: &mut OVector<F::Field, Dyn>;
        let scale_inv2_grad: &mut OVector<F::Field, Dyn>;
        let cauchy_scaled: &mut OVector<F::Field, Dyn>;

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum StepType {
            FullNewton,
            ScaledNewton,
            ScaledCauchy,
            Dogleg,
        }

        // Compute f(x), grad f(x) and H(x).
        let mut fx = f.apply(x);
        grad.compute(f, x, scale, fx, eps_sqrt, step_rule);
        hes.compute(f, x, scale, fx, eps_cbrt, step_rule);

        let estimate_delta = *delta == zero;
        if estimate_delta {
            *delta = grad.norm() * convert(0.1);
        }

        // Perform QR decomposition of H(x).
        let (qr_q, qr_r) = hes.clone_owned().qr().unpack();

        // Compute -Q^T grad f(x).
        grad_neg.copy_from(grad);
        grad_neg.neg_mut();
        qr_q.tr_mul_to(grad_neg, q_tr_grad_neg);

        // Find the Newton step by solving the system R newton = -Q^T grad f(x).
        newton.copy_from(q_tr_grad_neg);
        let is_newton_valid = qr_r.solve_upper_triangular_mut(newton);

        if !is_newton_valid {
            // TODO: Moore-Penrose pseudoinverse?
            debug!(
                "Newton step is invalid for ill-defined Hessian (zero columns: {:?})",
                hes.column_iter()
                    .enumerate()
                    .filter(|(_, col)| col.norm() == zero)
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            );
        }

        // Compute the norm of scaled Newton step (use temp fo storage).
        scaled_newton = temp;
        scaled_newton.copy_from(newton);
        scaled_newton.component_mul_assign(scale);
        let newton_scaled_norm = scaled_newton.norm();

        if is_newton_valid && newton_scaled_norm <= *delta {
            // Scaled Newton step is inside the trust region. We can safely take it.
            p.copy_from(newton);
            debug!("take full Newton: {:?}", p.as_slice());
            StepType::FullNewton
        } else {
            // Newton step is outside the trust region. We need to involve the
            // gradient.

            let grad_norm = grad_neg.norm();

            if grad_norm == zero {
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
                // Compute || D^-1 grad || (use p for storage).
                scale_inv2_grad = p;
                scale_inv2_grad.copy_from(scale);
                scale_inv2_grad.apply(|s| *s = one / *s);
                scale_inv2_grad.component_mul_assign(grad_neg);
                let scale_inv_grad_norm = scale_inv2_grad.norm();

                // Compute D^(-2) grad.
                scale_inv2_grad.copy_from(scale);
                scale_inv2_grad.apply(|s| *s = one / (*s * *s));
                scale_inv2_grad.component_mul_assign(grad_neg);
                scale_inv2_grad.neg_mut();

                // Compute g = -delta / || D^-1 grad|| * -D^(-2) grad, the
                // steepest descent direction in
                // scaled space (use cauchy for storage).
                cauchy.copy_from(scale_inv2_grad);
                *cauchy *= -*delta / scale_inv_grad_norm;

                // Calculate grad^T D^-2 H D^-2 grad.
                hes.mul_to(scale_inv2_grad, temp);
                let quadratic_form = scale_inv2_grad.dot(temp);

                let tau = if quadratic_form <= zero {
                    one
                } else {
                    // tau = min(|| D^-1 grad|| / delta * grad^T D^-2 H D^-2 grad, 1).
                    (scale_inv_grad_norm.powi(3) / (*delta * quadratic_form)).min(one)
                };

                let p = scale_inv2_grad;

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
                    // This polynomial has one negative root and one root in
                    // range (0, 1). We seek for the latter. Due to our choice
                    // of the polynomial, we have
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
                    let alpha = if b <= zero {
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
                    return Err(TrustRegionError::NoValidStep);
                }
            }
        };

        // Vector for Newton step is no longed used, so we reuse its allocations
        // for another purpose.
        let x_trial = newton;

        // Get candidate x' for the next iterate.
        x.add_to(p, x_trial);

        let not_feasible = dom.project(x_trial);

        if not_feasible {
            debug!("new iterate is not feasible, performing the projection");

            // Compute the step after projection.
            x_trial.sub_to(x, p);
        }

        // Compute f(x').
        let fx_trial = f.apply(x_trial);
        let is_trial_valid = fx_trial.is_finite();

        let gain_ratio = if is_trial_valid {
            // Compute the gain ratio.
            hes.mul_to(p, temp);
            let p_hes_p = p.dot(temp);
            let grad_p = grad.dot(p);
            let predicted = -(grad_p + p_hes_p * convert(0.5));

            let deny = if allow_ascent {
                // If ascent is allowed, then check only for zero, which would
                // make the gain ratio calculation ill-defined.
                predicted == zero
            } else {
                // If ascent is not allowed, test positivity of the predicted
                // gain. Note that even if the actual reduction was positive,
                // the step would be rejected anyway because the gain ratio
                // would be negative.
                predicted <= zero
            };

            if deny {
                if allow_ascent {
                    debug!("predicted gain = 0");
                } else {
                    debug!("predicted gain <= 0");
                }
                zero
            } else {
                let actual = fx - fx_trial;
                let gain_ratio = actual / predicted;
                debug!("gain ratio = {} / {} = {}", actual, predicted, gain_ratio);

                gain_ratio
            }
        } else {
            debug!("trial step is invalid, gain ratio = 0");
            zero
        };

        // Decide if the step is accepted or not.
        if gain_ratio > accept_thresh {
            // Accept the trial step.
            x.copy_from(x_trial);
            fx = fx_trial;
            debug!(
                "step accepted, fx = {}, x = {:?}",
                fx_trial,
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

        Ok(fx)
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
            assert!(f.is_root(&solve(&f, &dom, solver, x, 25, eps).unwrap(), eps));
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
            assert!(f.is_root(&solve(&f, &dom, solver, x, 50, eps).unwrap(), eps));
        }
    }

    #[test]
    fn considering_domain() {
        let f = BullardBiegler::new();
        let dom = f.domain();

        let x = nalgebra::dvector![1.0, 1.0];
        let solver = TrustRegion::new(&f, &dom);
        let eps = convert(1e-12);

        // From this initial point, dogleg converges to a point which is outside
        // the bounds.
        assert!(matches!(
            solve(&f, &dom, solver, x, 50, eps),
            Err(TestingError::Inner(TrustRegionError::NoProgress)) | Err(TestingError::Termination)
        ));
    }

    #[test]
    fn infinite_solutions() {
        let f = InfiniteSolutions::default();
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let solver = TrustRegion::new(&f, &dom);
            assert!(match solve(&f, &dom, solver, x, 25, eps) {
                Ok(root) => f.is_root(&root, eps),
                Err(TestingError::Inner(TrustRegionError::NoValidStep)) => true,
                Err(error) => panic!("{:?}", error),
            });
        }
    }

    #[test]
    fn sphere_optimization() {
        let n = 2;

        let f = Sphere::new(n);
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let optimizer = TrustRegion::new(&f, &dom);
            optimize(&f, &dom, optimizer, x, convert(0.0), 25, eps).unwrap();
        }
    }

    #[test]
    fn rosenbrock_optimization() {
        let n = 2;

        let f = ExtendedRosenbrock::new(n);
        let dom = f.domain();
        let eps = convert(1e-3);

        for x in f.initials() {
            let optimizer = TrustRegion::new(&f, &dom);
            optimize(&f, &dom, optimizer, x, convert(0.0), 250, eps).unwrap();
        }
    }
}
