//! Nelder-mead (simplex) optimization method.
//!
//! [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
//! simplex-reflection method is a popular derivative-free optimization
//! algorithm. It keeps a [simplex](https://en.wikipedia.org/wiki/Simplex) of *n
//! + 1* points and the simplex is reflected, expanded or contracted based on
//! squared norm of residuals comparison.
//!
//! # References
//!
//! \[1\] [Numerical
//! Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5)
//!
//! \[2\] [Implementing the Nelder-Mead simplex algorithm with adaptive
//! parameters](https://link.springer.com/article/10.1007/s10589-010-9329-3)
//!
//! \[3\] [Less is more: Simplified Nelder-Mead method for large unconstrained
//! optimization](https://api.semanticscholar.org/CorpusID:59403095)

use getset::{CopyGetters, Setters};
use log::debug;
use nalgebra::{
    allocator::Allocator,
    convert,
    storage::{Storage, StorageMut},
    DefaultAllocator, Dim, DimName, IsContiguous, OVector, RealField, Vector, U1,
};
use num_traits::{One, Zero};
use thiserror::Error;

use crate::{
    core::{
        Domain, Function, FunctionResultExt, Optimizer, Problem, ProblemError, Solver, System,
        VectorDomainExt,
    },
    derivatives::EPSILON_SQRT,
};

/// Family of coefficients for reflection, expansion and contractions.
#[derive(Debug, Clone, Copy)]
pub enum CoefficientsFamily {
    /// Standard ("textbook") choice.
    Standard,
    /// The coefficients are adjusted compared to standard by taking system
    /// dimension into account to avoid diminishing of expansion and contraction
    /// steps in higher dimensions.
    Balanced,
    /// The coefficients are left unchanged so it is the responsibility of the
    /// user to set them through [`NelderMeadOptions`].
    Fixed,
}

/// Options for [`NelderMead`] solver.
#[derive(Debug, Clone, CopyGetters, Setters)]
#[getset(get_copy = "pub", set = "pub")]
pub struct NelderMeadOptions<F: Problem> {
    /// Family for coefficients adaptation or fixed coefficients. Default:
    /// balanced (see [`CoefficientsFamily`]).
    family: CoefficientsFamily,
    /// Coefficient for reflection operation. Default: `-1`.
    reflection_coeff: F::Scalar,
    /// Coefficient for expansion operation. Default: `-2`.
    expansion_coeff: F::Scalar,
    /// Coefficient for outer contraction operation. Default: `-0.5`.
    outer_contraction_coeff: F::Scalar,
    /// Coefficient for inner contraction operation. Default: `0.5`.
    inner_contraction_coeff: F::Scalar,
    /// Coefficient for shrinking operation. Default: `0.5`.
    shrink_coeff: F::Scalar,
}

impl<F: Problem> Default for NelderMeadOptions<F> {
    fn default() -> Self {
        Self {
            family: CoefficientsFamily::Balanced,
            reflection_coeff: convert(-1.0),
            expansion_coeff: convert(-2.0),
            outer_contraction_coeff: convert(-0.5),
            inner_contraction_coeff: convert(0.5),
            shrink_coeff: convert(0.5),
        }
    }
}

impl<F: Problem> NelderMeadOptions<F> {
    fn overwrite_coeffs(&mut self, f: &F) {
        let Self {
            family,
            reflection_coeff,
            expansion_coeff,
            outer_contraction_coeff,
            inner_contraction_coeff,
            shrink_coeff,
        } = self;

        match family {
            CoefficientsFamily::Standard => {
                *reflection_coeff = convert(-1.0);
                *expansion_coeff = convert(-2.0);
                *outer_contraction_coeff = convert(-0.5);
                *inner_contraction_coeff = -*outer_contraction_coeff;
                *shrink_coeff = convert(0.5);
            }
            CoefficientsFamily::Balanced => {
                let n: F::Scalar = convert(f.dim().value() as f64);
                let n_inv = F::Scalar::one() / n;

                *reflection_coeff = convert(-1.0);
                *expansion_coeff = -(n_inv * convert(2.0) + convert(1.0));
                *outer_contraction_coeff = -(F::Scalar::one() - n_inv);
                *inner_contraction_coeff = -*outer_contraction_coeff;
                *shrink_coeff = F::Scalar::one() - n_inv;
            }
            CoefficientsFamily::Fixed => {
                // Leave unchanged.
            }
        }
    }
}

/// Nelder-Mead solver. See [module](self) documentation for more details.
pub struct NelderMead<F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    options: NelderMeadOptions<F>,
    fx: OVector<F::Scalar, F::Dim>,
    fx_best: OVector<F::Scalar, F::Dim>,
    scale: OVector<F::Scalar, F::Dim>,
    centroid: OVector<F::Scalar, F::Dim>,
    reflection: OVector<F::Scalar, F::Dim>,
    expansion: OVector<F::Scalar, F::Dim>,
    contraction: OVector<F::Scalar, F::Dim>,
    simplex: Vec<OVector<F::Scalar, F::Dim>>,
    errors: Vec<F::Scalar>,
    sort_perm: Vec<usize>,
}

impl<F: Problem> NelderMead<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    /// Initializes Nelder-Mead solver with default options.
    pub fn new(f: &F, dom: &Domain<F::Scalar>) -> Self {
        Self::with_options(f, dom, NelderMeadOptions::default())
    }

    /// Initializes Nelder-Mead solver with given options.
    pub fn with_options(f: &F, dom: &Domain<F::Scalar>, mut options: NelderMeadOptions<F>) -> Self {
        options.overwrite_coeffs(f);

        let scale_iter = dom.vars().iter().map(|var| var.scale());

        Self {
            options,
            fx: OVector::zeros_generic(f.dim(), U1::name()),
            fx_best: OVector::zeros_generic(f.dim(), U1::name()),
            scale: OVector::from_iterator_generic(f.dim(), U1::name(), scale_iter),
            centroid: OVector::zeros_generic(f.dim(), U1::name()),
            reflection: OVector::zeros_generic(f.dim(), U1::name()),
            expansion: OVector::zeros_generic(f.dim(), U1::name()),
            contraction: OVector::zeros_generic(f.dim(), U1::name()),
            simplex: Vec::with_capacity(f.dim().value() + 1),
            errors: Vec::with_capacity(f.dim().value() + 1),
            sort_perm: Vec::with_capacity(f.dim().value() + 1),
        }
    }

    /// Resets the internal state of the solver.
    pub fn reset(&mut self) {
        // Causes simplex to be initialized again.
        self.simplex.clear();
        self.errors.clear();
        self.sort_perm.clear();
    }
}

/// Error returned from [`NelderMead`] solver.
#[derive(Debug, Error)]
pub enum NelderMeadError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    Problem(#[from] ProblemError),
    /// Simplex collapsed so it is impossible to make any progress.
    #[error("simplex collapsed")]
    SimplexCollapsed,
}

impl<F: Function> NelderMead<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    fn next_inner<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
    ) -> Result<F::Scalar, NelderMeadError>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
    {
        let NelderMeadOptions {
            reflection_coeff,
            expansion_coeff,
            outer_contraction_coeff,
            inner_contraction_coeff,
            shrink_coeff,
            ..
        } = self.options;

        let Self {
            fx,
            fx_best,
            scale,
            simplex,
            errors,
            sort_perm,
            centroid,
            reflection,
            expansion,
            contraction,
            ..
        } = self;

        let n = f.dim().value();
        let inf = convert(f64::INFINITY);

        if simplex.is_empty() {
            // Simplex initialization.

            // It is important to return early on error before the point is
            // added to the simplex.
            let mut error_best = f.apply(x)?;
            errors.push(error_best);
            simplex.push(x.clone_owned());

            for j in 0..n {
                let mut xi = x.clone_owned();
                xi[j] = dom.vars()[j].clamp(xi[j] + scale[j]);

                let error = match f.apply(&xi) {
                    Ok(error) => error,
                    // Do not fail when invalid value is encountered during
                    // building the simplex. Instead, treat it as infinity so
                    // that it is worse than (or "equal" to) any other error and
                    // hope that it gets replaced. The exception is the very
                    // point provided by the caller as it is expected to be
                    // valid and it is erroneous situation if it is not.
                    Err(ProblemError::InvalidValue) => inf,
                    Err(error) => {
                        // Clear the simplex so the solver is not in invalid
                        // state.
                        simplex.clear();
                        errors.clear();
                        return Err(error.into());
                    }
                };

                if error < error_best {
                    fx_best.copy_from(fx);
                    error_best = error;
                }

                errors.push(error);
                simplex.push(xi);
            }

            let inf_error_count = errors.iter().filter(|e| **e == inf).count();

            if inf_error_count >= n / 2 {
                // The simplex is too degenerate.
                simplex.clear();
                errors.clear();
                return Err(NelderMeadError::Problem(ProblemError::InvalidValue));
            }

            sort_perm.extend(0..=n);
            // Stable sort is important for sort_perm[0] being consistent with
            // fx_best.
            sort_perm.sort_by(|a, b| {
                errors[*a]
                    .partial_cmp(&errors[*b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Calculate the centroid.
        centroid.fill(F::Scalar::zero());
        (0..n)
            .map(|i| &simplex[sort_perm[i]])
            .for_each(|xi| *centroid += xi);
        *centroid /= convert(n as f64);

        #[derive(Debug, Clone, Copy, PartialEq)]
        enum Transformation {
            Reflection,
            Expansion,
            OuterContraction,
            InnerContraction,
            Shrinkage,
        }

        impl Transformation {
            fn as_str(&self) -> &str {
                match self {
                    Transformation::Reflection => "reflection",
                    Transformation::Expansion => "expansion",
                    Transformation::OuterContraction => "outer contraction",
                    Transformation::InnerContraction => "inner contraction",
                    Transformation::Shrinkage => "shrinkage",
                }
            }
        }

        // Perform one of possible simplex transformations.
        reflection.on_line2_mut(centroid, &simplex[sort_perm[n]], reflection_coeff);
        let reflection_not_feasible = reflection.project(dom);
        let reflection_error = f.apply_eval(reflection, fx).ignore_invalid_value(inf)?;

        #[allow(clippy::suspicious_else_formatting)]
        let (transformation, not_feasible) = if errors[sort_perm[0]] <= reflection_error
            && reflection_error < errors[sort_perm[n - 1]]
        {
            // Reflected point is neither best nor worst in the new simplex.
            // Just replace the worst point.
            simplex[sort_perm[n]].copy_from(reflection);
            errors[sort_perm[n]] = reflection_error;
            (Transformation::Reflection, reflection_not_feasible)
        } else if reflection_error < errors[sort_perm[0]] {
            fx_best.copy_from(fx);

            // Reflected point is better than the current best. Try to go
            // farther along this direction.
            expansion.on_line2_mut(centroid, &simplex[sort_perm[n]], expansion_coeff);
            let expansion_not_feasible = expansion.project(dom);
            let expansion_error = f.apply_eval(expansion, fx).ignore_invalid_value(inf)?;

            if expansion_error < reflection_error {
                fx_best.copy_from(fx);

                // Expansion indeed helped, replace the worst point.
                simplex[sort_perm[n]].copy_from(expansion);
                errors[sort_perm[n]] = expansion_error;
                (Transformation::Expansion, expansion_not_feasible)
            } else {
                // Expansion didn't help, replace the worst point with the
                // reflected point.
                simplex[sort_perm[n]].copy_from(reflection);
                errors[sort_perm[n]] = reflection_error;
                (Transformation::Reflection, reflection_not_feasible)
            }
        } else
        /* reflection_error >= errors[sort_perm[n - 1]] */
        {
            // Reflected point is still worse than the second to last point. Try
            // to do a contraction.
            let (transformation, not_feasible) = if errors[sort_perm[n - 1]] <= reflection_error
                && reflection_error < errors[sort_perm[n]]
            {
                // Try to perform outer contraction.
                contraction.on_line2_mut(centroid, &simplex[sort_perm[n]], outer_contraction_coeff);
                let contraction_not_feasible = contraction.project(dom);
                let contraction_error = f.apply_eval(contraction, fx).ignore_invalid_value(inf)?;

                if contraction_error <= reflection_error {
                    if contraction_error < errors[sort_perm[0]] {
                        fx_best.copy_from(fx);
                    }

                    // Use the contracted point instead of the reflected point
                    // because it's better.
                    simplex[sort_perm[n]].copy_from(contraction);
                    errors[sort_perm[n]] = contraction_error;
                    (
                        Some(Transformation::OuterContraction),
                        contraction_not_feasible,
                    )
                } else {
                    (None, false)
                }
            } else {
                // Try to perform inner contraction.
                contraction.on_line2_mut(centroid, &simplex[sort_perm[n]], inner_contraction_coeff);
                let contraction_not_feasible = contraction.project(dom);
                let contraction_error = f.apply_eval(contraction, fx).ignore_invalid_value(inf)?;

                if contraction_error <= errors[sort_perm[n]] {
                    if contraction_error < errors[sort_perm[0]] {
                        fx_best.copy_from(fx);
                    }

                    // The contracted point is better than the worst point.
                    simplex[sort_perm[n]].copy_from(contraction);
                    errors[sort_perm[n]] = contraction_error;
                    (
                        Some(Transformation::InnerContraction),
                        contraction_not_feasible,
                    )
                } else {
                    (None, false)
                }
            };

            match transformation {
                Some(transformation) => (transformation, not_feasible),
                None => {
                    // Neither outside nor inside contraction was acceptable.
                    // Shrink the simplex towards the best point.

                    contraction.copy_from(&simplex[sort_perm[0]]);
                    let mut error_best = errors[sort_perm[0]];

                    for i in 1..=n {
                        let xi = &mut simplex[sort_perm[i]];
                        xi.on_line_mut(contraction, shrink_coeff);
                        let error = f.apply_eval(xi, fx).ignore_invalid_value(inf)?;
                        errors[sort_perm[i]] = error;

                        if error < error_best {
                            fx_best.copy_from(fx);
                            error_best = error;
                        }
                    }

                    (Transformation::Shrinkage, false)
                }
            }
        };

        // Establish the ordering of simplex points. Stable sort is important
        // for sort_perm[0] being consistent with fx_best.
        sort_perm.sort_by(|a, b| {
            errors[*a]
                .partial_cmp(&errors[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            "performed {}{},\t|| fx || = {} - {}",
            transformation.as_str(),
            if not_feasible { " with projection" } else { "" },
            errors[sort_perm[0]],
            errors[sort_perm[n]]
        );

        // Return the best simplex point.
        x.copy_from(&simplex[sort_perm[0]]);
        // fx corresponding to x is stored in `self.fx_best`.

        if transformation == Transformation::Shrinkage || not_feasible {
            // Check whether the simplex collapsed or not. It can happen only
            // when shrinkage is performed or a new point was projected into the
            // feasible domain, because otherwise an error reduction was
            // achieved. This criterion is taken from "Less is more: Simplified
            // Nelder-Mead method for large unconstrained optimization".
            let eps: F::Scalar = convert(EPSILON_SQRT);

            let worst = errors[sort_perm[n]];
            let best = errors[sort_perm[0]];
            let numer = (worst - best) * convert(2.0);
            let denom = worst + best + eps;

            if numer / denom <= eps {
                return Err(NelderMeadError::SimplexCollapsed);
            }
        }

        Ok(errors[sort_perm[0]])
    }
}

impl<F: Function> Optimizer<F> for NelderMead<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    const NAME: &'static str = "Nelder-Mead";

    type Error = NelderMeadError;

    fn next<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
    ) -> Result<F::Scalar, Self::Error>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
    {
        self.next_inner(f, dom, x)
    }
}

impl<F: System + Function> Solver<F> for NelderMead<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
{
    const NAME: &'static str = "Nelder-Mead";

    type Error = NelderMeadError;

    fn next<Sx, Sfx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
        fx: &mut Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<(), Self::Error>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
        Sfx: StorageMut<F::Scalar, F::Dim>,
    {
        self.next_inner(f, dom, x)
            .map(|_| fx.copy_from(&self.fx_best))
    }
}

trait VectorNelderMeadExt<T: RealField, D: Dim> {
    fn on_line_mut<Sto>(&mut self, to: &Vector<T, D, Sto>, t: T)
    where
        Sto: Storage<T, D>;

    fn on_line2_mut<Sfrom, Sto>(
        &mut self,
        from: &Vector<T, D, Sfrom>,
        to: &Vector<T, D, Sto>,
        t: T,
    ) where
        Sfrom: Storage<T, D>,
        Sto: Storage<T, D>;
}

impl<T: RealField, D: Dim, S> VectorNelderMeadExt<T, D> for Vector<T, D, S>
where
    S: StorageMut<T, D>,
{
    fn on_line_mut<Sto>(&mut self, to: &Vector<T, D, Sto>, t: T)
    where
        Sto: Storage<T, D>,
    {
        *self += to;
        *self *= t;
    }

    fn on_line2_mut<Sfrom, Sto>(&mut self, from: &Vector<T, D, Sfrom>, to: &Vector<T, D, Sto>, t: T)
    where
        Sfrom: Storage<T, D>,
        Sto: Storage<T, D>,
    {
        to.sub_to(from, self);
        *self *= t;
        *self += from;
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
            let solver = NelderMead::new(&f, &dom);
            f.is_root(&solve(&f, &dom, solver, x, 250, eps).unwrap(), eps);
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
            let solver = NelderMead::new(&f, &dom);
            f.is_root(&solve(&f, &dom, solver, x, 500, eps).unwrap(), eps);
        }
    }

    #[test]
    fn considering_domain() {
        let f = BullardBiegler::new();
        let dom = f.domain();

        let x = nalgebra::vector![1.0, 1.0];
        let solver = NelderMead::new(&f, &dom);
        let eps = convert(1e-12);

        // From this initial point, Nelder-Mead converges to a point which is
        // outside the bounds.
        assert!(matches!(
            solve(&f, &dom, solver, x, 5, eps),
            Err(SolveError::Solver(NelderMeadError::SimplexCollapsed))
        ));
    }
}
