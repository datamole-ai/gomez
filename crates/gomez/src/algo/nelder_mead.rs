//! Nelder-mead (simplex) optimization method.
//!
//! [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
//! simplex-reflection method is a popular derivative-free optimization
//! algorithm. It keeps a [simplex](https://en.wikipedia.org/wiki/Simplex) of
//! _n + 1_ points and the simplex is reflected, expanded or contracted based on
//! the function values comparison.
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
//!
//! \[4\] [Gilding the Lily: A Variant of the Nelder-Mead Algorithm Based on
//! Golden-Section
//! Search](https://link.springer.com/article/10.1023/A:1014842520519)

use getset::{CopyGetters, Setters};
use log::debug;
use nalgebra::{
    convert,
    storage::{Storage, StorageMut},
    ComplexField, Dim, DimName, Dyn, IsContiguous, OVector, RealField, Vector, U1,
};
use thiserror::Error;

use crate::core::{Domain, Function, Optimizer, Problem, RealField as _, Solver, System};

/// Family of coefficients for reflection, expansion and contractions.
#[derive(Debug, Clone, Copy)]
pub enum CoefficientsFamily {
    /// Standard ("textbook") choice.
    Standard,
    /// The coefficients are adjusted compared to standard by taking system
    /// dimension into account to avoid diminishing of expansion and contraction
    /// steps in higher dimensions.
    Balanced,
    /// The coefficients are chosen such that the algorithm becomes a
    /// golden-section search.
    GoldenSection,
    /// The coefficients are left unchanged so it is the responsibility of the
    /// user to set them through [`NelderMeadOptions`].
    Fixed,
}

/// Options for [`NelderMead`] solver.
#[derive(Debug, Clone, CopyGetters, Setters)]
#[getset(get_copy = "pub", set = "pub")]
pub struct NelderMeadOptions<P: Problem> {
    /// Family for coefficients adaptation or fixed coefficients. Default:
    /// balanced (see [`CoefficientsFamily`]).
    family: CoefficientsFamily,
    /// Coefficient for reflection operation. Default: `-1`.
    reflection_coeff: P::Field,
    /// Coefficient for expansion operation. Default: `-2`.
    expansion_coeff: P::Field,
    /// Coefficient for outer contraction operation. Default: `-0.5`.
    outer_contraction_coeff: P::Field,
    /// Coefficient for inner contraction operation. Default: `0.5`.
    inner_contraction_coeff: P::Field,
    /// Coefficient for shrinking operation. Default: `0.5`.
    shrink_coeff: P::Field,
}

impl<P: Problem> Default for NelderMeadOptions<P> {
    fn default() -> Self {
        Self {
            family: CoefficientsFamily::Standard,
            reflection_coeff: convert(-1.0),
            expansion_coeff: convert(-2.0),
            outer_contraction_coeff: convert(-0.5),
            inner_contraction_coeff: convert(0.5),
            shrink_coeff: convert(0.5),
        }
    }
}

impl<P: Problem> NelderMeadOptions<P> {
    fn overwrite_coeffs(&mut self, dom: &Domain<P::Field>) {
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
                let n: P::Field = convert(dom.dim() as f64);
                let one: P::Field = convert(1.0);
                let n_inv = one / n;

                *reflection_coeff = convert(-1.0);
                *expansion_coeff = -(n_inv * convert(2.0) + one);
                *outer_contraction_coeff = -(one - n_inv);
                *inner_contraction_coeff = -*outer_contraction_coeff;
                *shrink_coeff = one - n_inv;
            }
            CoefficientsFamily::GoldenSection => {
                let alpha = 1.0 / (0.5 * (5f64.sqrt() + 1.0));
                *reflection_coeff = convert(-1.0);
                *expansion_coeff = convert(-1.0 / alpha);
                *outer_contraction_coeff = convert(-alpha);
                *inner_contraction_coeff = convert(alpha.powi(2));
                *shrink_coeff = convert(-alpha.powi(2));
            }
            CoefficientsFamily::Fixed => {
                // Leave unchanged.
            }
        }
    }
}

/// Nelder-Mead solver.
///
/// See [module](self) documentation for more details.
pub struct NelderMead<P: Problem> {
    options: NelderMeadOptions<P>,
    scale: OVector<P::Field, Dyn>,
    centroid: OVector<P::Field, Dyn>,
    reflection: OVector<P::Field, Dyn>,
    expansion: OVector<P::Field, Dyn>,
    contraction: OVector<P::Field, Dyn>,
    simplex: Vec<OVector<P::Field, Dyn>>,
    errors: Vec<P::Field>,
    sort_perm: Vec<usize>,
}

impl<P: Problem> NelderMead<P> {
    /// Initializes Nelder-Mead solver with default options.
    pub fn new(p: &P, dom: &Domain<P::Field>) -> Self {
        Self::with_options(p, dom, NelderMeadOptions::default())
    }

    /// Initializes Nelder-Mead solver with given options.
    pub fn with_options(_: &P, dom: &Domain<P::Field>, mut options: NelderMeadOptions<P>) -> Self {
        let dim = Dyn(dom.dim());

        options.overwrite_coeffs(dom);

        let scale = dom
            .scale()
            .map(|scale| OVector::from_iterator_generic(dim, U1::name(), scale.iter().copied()))
            .unwrap_or_else(|| OVector::from_element_generic(dim, U1::name(), convert(1.0)));

        Self {
            options,
            scale,
            centroid: OVector::zeros_generic(dim, U1::name()),
            reflection: OVector::zeros_generic(dim, U1::name()),
            expansion: OVector::zeros_generic(dim, U1::name()),
            contraction: OVector::zeros_generic(dim, U1::name()),
            simplex: Vec::with_capacity(dom.dim() + 1),
            errors: Vec::with_capacity(dom.dim() + 1),
            sort_perm: Vec::with_capacity(dom.dim() + 1),
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
    /// Simplex collapsed so it is impossible to make any progress.
    #[error("simplex collapsed")]
    SimplexCollapsed,
    /// Simplex contains too many invalid values (NaN, infinity).
    #[error("simplex contains too many invalid values")]
    SimplexInvalid,
}

impl<F: Function> NelderMead<F> {
    fn next_inner<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Field>,
        x: &mut Vector<F::Field, Dyn, Sx>,
    ) -> Result<F::Field, NelderMeadError>
    where
        Sx: StorageMut<F::Field, Dyn> + IsContiguous,
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

        let n = dom.dim();

        if simplex.is_empty() {
            // Simplex initialization.

            // It is important to return early on error before the point is
            // added to the simplex.
            let mut error_best = f.apply(x);
            errors.push(error_best);
            simplex.push(x.clone_owned());

            for j in 0..n {
                let mut xi = x.clone_owned();
                xi[j] += scale[j];
                dom.project_in(&mut xi, j);

                let error = f.apply(&xi);

                if error < error_best {
                    error_best = error;
                }

                errors.push(error);
                simplex.push(xi);
            }

            let error_count = errors.iter().filter(|e| !e.is_finite()).count();

            if error_count >= simplex.len() / 2 {
                // The simplex is too degenerate.
                debug!(
                    "{} out of {} points in simplex have invalid value, returning error",
                    error_count,
                    simplex.len()
                );
                simplex.clear();
                errors.clear();
                return Err(NelderMeadError::SimplexInvalid);
            }

            sort_perm.extend(0..=n);
            sort_perm.sort_by(|a, b| {
                errors[*a]
                    .partial_cmp(&errors[*b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Calculate the centroid.
        centroid.fill(convert(0.0));
        (0..n)
            .map(|i| &simplex[sort_perm[i]])
            .for_each(|xi| *centroid += xi);
        *centroid /= convert(n as f64);

        debug!("centroid of simplex: {:?}", centroid.as_slice());

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
        let reflection_not_feasible = dom.project(reflection);
        let reflection_error = f.apply(reflection).nan_to_inf();

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
            // Reflected point is better than the current best. Try to go
            // farther along this direction.
            expansion.on_line2_mut(centroid, &simplex[sort_perm[n]], expansion_coeff);
            let expansion_not_feasible = dom.project(expansion);
            let expansion_error = f.apply(expansion).nan_to_inf();

            if expansion_error < reflection_error {
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
                let contraction_not_feasible = dom.project(contraction);
                let contraction_error = f.apply(contraction).nan_to_inf();

                if contraction_error <= reflection_error {
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
                let contraction_not_feasible = dom.project(contraction);
                let contraction_error = f.apply(contraction).nan_to_inf();

                if contraction_error <= errors[sort_perm[n]] {
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
                        let error = f.apply(xi).nan_to_inf();
                        errors[sort_perm[i]] = error;

                        if error < error_best {
                            error_best = error;
                        }
                    }

                    (Transformation::Shrinkage, false)
                }
            }
        };

        // Establish the ordering of simplex points.
        sort_perm.sort_by(|a, b| {
            errors[*a]
                .partial_cmp(&errors[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            "performed {}{},\tfx = {} - {}",
            transformation.as_str(),
            if not_feasible { " with projection" } else { "" },
            errors[sort_perm[0]],
            errors[sort_perm[n]]
        );

        // Return the best simplex point.
        x.copy_from(&simplex[sort_perm[0]]);

        if transformation == Transformation::Shrinkage
            || transformation == Transformation::InnerContraction
            || not_feasible
        {
            // Check whether the simplex collapsed or not. It can happen only
            // when shrinkage or, when n = 1 inner contraction, is performed or
            // a new point was projected into the feasible domain, because
            // otherwise an error reduction was achieved. This criterion is
            // taken from "Less is more: Simplified Nelder-Mead method for large
            // unconstrained optimization".
            let eps = F::Field::EPSILON_SQRT;

            let worst = errors[sort_perm[n]];
            let best = errors[sort_perm[0]];
            let numer = (worst - best) * convert(2.0);
            let denom = worst + best + eps;

            if numer / denom <= eps {
                debug!("simplex collapsed: {} / {} <= {}", numer, denom, eps);
                return Err(NelderMeadError::SimplexCollapsed);
            }
        }

        Ok(errors[sort_perm[0]])
    }
}

impl<F: Function> Optimizer<F> for NelderMead<F> {
    const NAME: &'static str = "Nelder-Mead";

    type Error = NelderMeadError;

    fn opt_next<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Field>,
        x: &mut Vector<F::Field, Dyn, Sx>,
    ) -> Result<F::Field, Self::Error>
    where
        Sx: StorageMut<F::Field, Dyn> + IsContiguous,
    {
        self.next_inner(f, dom, x)
    }
}

impl<R: System> Solver<R> for NelderMead<R> {
    const NAME: &'static str = "Nelder-Mead";

    type Error = NelderMeadError;

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
        self.next_inner(r, dom, x)?;
        r.eval(x, rx);
        Ok(())
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

trait RealFieldNelderMeadExt {
    fn nan_to_inf(self) -> Self;
}

impl<T: RealField> RealFieldNelderMeadExt for T {
    fn nan_to_inf(self) -> Self {
        if self.is_finite() {
            self
        } else {
            // Not finite also covers NaN and negative infinity.
            T::from_subset(&f64::INFINITY)
        }
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
            assert!(f.is_root(&solve(&f, &dom, solver, x, 250, eps).unwrap(), eps));
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
            assert!(f.is_root(&solve(&f, &dom, solver, x, 500, eps).unwrap(), eps));
        }
    }

    #[test]
    fn considering_domain() {
        let f = BullardBiegler::new();
        let dom = f.domain();

        let x = nalgebra::dvector![1.0, 1.0];
        let solver = NelderMead::new(&f, &dom);
        let eps = convert(1e-12);

        // From this initial point, Nelder-Mead converges to a point which is
        // outside the bounds.
        assert!(matches!(
            solve(&f, &dom, solver, x, 5, eps),
            Err(TestingError::Inner(NelderMeadError::SimplexCollapsed))
        ));
    }

    #[test]
    fn univariate_optimization() {
        let f = Sphere::new(1);
        let dom = f.domain();
        let eps = convert(1e-3);

        for x in f.initials() {
            let optimizer = NelderMead::new(&f, &dom);
            optimize(&f, &dom, optimizer, x, convert(0.0), 10, eps).unwrap();
        }
    }
}
