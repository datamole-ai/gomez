//! Cuckoo search global optimization algorithm.
//!
//! [Cuckoo search](https://en.wikipedia.org/wiki/Cuckoo_search) is an
//! optimization algorithm inspired by brood parasitism of cuckoo species. It is
//! a combination of local search around current individuals and far-field
//! randomization to escape local optima. It is to some extent similar to
//! [differential
//! evolution](https://en.wikipedia.org/wiki/Differential_evolution).
//!
//! # References
//!
//! \[1\] [Engineering Optimisation by Cuckoo
//! Search](https://arxiv.org/abs/1005.2908)
//!
//! \[2\] [Nature-Inspired Metaheuristic
//! Algorithms](https://dl.acm.org/doi/10.5555/1628847)
//!
//! \[3\] [Applying Modified Cuckoo Search Algorithm for Solving Systems of
//! Nonlinear Equations](https://dl.acm.org/doi/10.1007/s00521-017-3088-3)

use getset::{CopyGetters, Getters, Setters};
use log::debug;
use nalgebra::{
    allocator::{Allocator, Reallocator},
    convert,
    storage::StorageMut,
    DefaultAllocator, DimMin, DimName, Dynamic, IsContiguous, OVector, Vector, U1,
};
use num_traits::{Float, Zero};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, StandardNormal, Uniform};
use thiserror::Error;

use crate::{
    core::{Domain, Function, Optimizer, Problem, ProblemError, Solver, System},
    population::{Population, PopulationInit, PopulationSize, UniformInit},
};

/// Direction when performing local search for an individual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalWalkDirection {
    /// Individuals are attracted to the best individual in the population.
    TowardsBest,
    /// The direction is influenced only by scales of the variables.
    Scaled,
}

/// Options for [`Cuckoo`] solver.
#[derive(Debug, Clone, CopyGetters, Getters, Setters)]
pub struct CuckooOptions<F: Problem, I: PopulationInit<F>> {
    /// Population size. Default: adaptive (see [`PopulationSize`]).
    #[getset(get_copy = "pub", set = "pub")]
    population_size: PopulationSize,
    /// Population initializer. Default: [`UniformInit`].
    #[getset(get = "pub")]
    population_init: I,
    /// Scale factor when doing local search. Default: `0.05`.
    #[getset(get_copy = "pub", set = "pub")]
    scale_factor: F::Scalar,
    /// Probability of abandoning a nest (i.e., doing far-field randomization).
    /// Default: `0.25`.
    #[getset(get_copy = "pub", set = "pub")]
    abandon_prob: f64,
    /// Fraction of the population that is immune to far-field randomization.
    /// Default: `0.15`.
    #[getset(get_copy = "pub", set = "pub")]
    elite_fraction: f64,
    /// Local search direction. Default: scaled (see [`LocalWalkDirection`]).
    #[getset(get_copy = "pub", set = "pub")]
    local_walk_dir: LocalWalkDirection,
}

impl<F: Problem, I: PopulationInit<F>> CuckooOptions<F, I> {
    /// Initializes the options with given population initializer.
    pub fn with_population_init(population_init: I) -> Self {
        Self {
            population_size: PopulationSize::Adaptive,
            population_init,
            scale_factor: convert(0.05),
            abandon_prob: 0.25,
            elite_fraction: 0.15,
            local_walk_dir: LocalWalkDirection::Scaled,
        }
    }
}

impl<F: Problem> Default for CuckooOptions<F, UniformInit<F>>
where
    F::Scalar: SampleUniform,
{
    fn default() -> Self {
        Self::with_population_init(UniformInit::default())
    }
}

/// Cuckoo search solver. See [module](self) documentation for more details.
pub struct Cuckoo<F: Problem, I: PopulationInit<F>, R: Rng>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    options: CuckooOptions<F, I>,
    population: Population<F>,
    next_gen: Population<F>,
    magnitude: OVector<F::Scalar, F::Dim>,
    rand_perm1: Vec<usize>,
    rand_perm2: Vec<usize>,
    best: OVector<F::Scalar, F::Dim>,
    temp: OVector<F::Scalar, F::Dim>,
    elite_size: usize,
    rng: R,
}

impl<F: Problem, R: Rng> Cuckoo<F, UniformInit<F>, R>
where
    DefaultAllocator: Allocator<F::Scalar, Dynamic>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, Dynamic, F::Dim>,
    F::Scalar: SampleUniform,
{
    /// Initializes cuckoo search solver with default options.
    pub fn new(f: &F, dom: &Domain<F::Scalar>, rng: R) -> Self
    where
        F: Function,
    {
        Self::with_options(f, dom, rng, CuckooOptions::default())
    }
}

impl<F: Problem, I: PopulationInit<F>, R: Rng> Cuckoo<F, I, R>
where
    DefaultAllocator: Allocator<F::Scalar, Dynamic>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, Dynamic, F::Dim>,
    F::Scalar: SampleUniform,
{
    /// Initializes cuckoo search solver with given options.
    pub fn with_options(
        f: &F,
        dom: &Domain<F::Scalar>,
        mut rng: R,
        options: CuckooOptions<F, I>,
    ) -> Self
    where
        F: Function,
    {
        let population_size = options.population_size.get(f);
        let elite_fraction = options.elite_fraction;

        let population = Population::new(
            f,
            dom,
            &mut rng,
            &options.population_init,
            options.population_size,
        );
        let next_gen = Population::zeros(f, options.population_size);
        let temp = OVector::zeros_generic(f.dim(), U1::name());

        let magnitude_iter = dom.vars().iter().map(|var| var.magnitude());

        Self {
            options,
            population,
            next_gen,
            magnitude: OVector::from_iterator_generic(f.dim(), U1::name(), magnitude_iter),
            rand_perm1: (0..population_size).collect(),
            rand_perm2: (0..population_size).collect(),
            best: OVector::zeros_generic(f.dim(), U1::name()),
            temp,
            elite_size: 1.max((elite_fraction * (population_size as f64)) as usize),
            rng,
        }
    }

    /// Get the current population.
    pub fn population(&self) -> &Population<F> {
        &self.population
    }

    /// Resets the internal state of the solver.
    pub fn reset(&mut self, f: &F, dom: &Domain<F::Scalar>)
    where
        F: Function,
    {
        self.population
            .reinit(f, dom, &mut self.rng, &self.options.population_init);
    }
}

/// Error returned from [`Cuckoo`] solver.
#[derive(Debug, Error)]
pub enum CuckooError {
    /// Error that occurred when evaluating the system.
    #[error("{0}")]
    Problem(#[from] ProblemError),
}

impl<F: Function, I: PopulationInit<F>, R: Rng> Cuckoo<F, I, R>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    F::Dim: DimMin<F::Dim, Output = F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, <F::Dim as DimMin<F::Dim>>::Output>,
    DefaultAllocator: Reallocator<F::Scalar, F::Dim, F::Dim, F::Dim, F::Dim>,
    StandardNormal: Distribution<F::Scalar>,
    F::Scalar: SampleUniform + Float,
{
    fn next_inner<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Scalar>,
        x: &mut Vector<F::Scalar, F::Dim, Sx>,
    ) -> Result<F::Scalar, CuckooError>
    where
        Sx: StorageMut<F::Scalar, F::Dim> + IsContiguous,
    {
        let CuckooOptions {
            scale_factor,
            abandon_prob,
            local_walk_dir,
            ..
        } = self.options;

        let elite_size = self.elite_size;

        let Self {
            population,
            next_gen,
            magnitude,
            rand_perm1,
            rand_perm2,
            best,
            temp,
            rng,
            ..
        } = self;

        best.copy_from(&*population.iter_sorted().next().unwrap());

        for (x, mut next) in population.iter().zip(next_gen.iter_mut()) {
            // Perform local random walk.
            temp.copy_from(&x);

            match local_walk_dir {
                LocalWalkDirection::TowardsBest => {
                    temp.sub_to(&*best, &mut *next);
                }
                LocalWalkDirection::Scaled => {
                    next.copy_from(magnitude);
                }
            }

            *next *= scale_factor;
            next.apply(|uj| *uj *= rng.sample(StandardNormal));
            *next += &*temp;

            // Make sure that the candidate is in domain.
            next.clamp(dom);

            // Evaluate and replace if better.
            match next.eval(f, temp) {
                Ok(error) if error < x.error() => {
                    // Accept the candidate, just update the error.
                    next.set_error(error);
                }
                _ => {
                    // Reject the candidate, copy the old individual.
                    next.copy_from(&x);
                    next.set_error(x.error());
                }
            }
        }

        rand_perm(rand_perm1, rng);
        rand_perm(rand_perm2, rng);

        // Sort the temporary population to be able to determine elite just
        // using the elite size.
        next_gen.sort();

        for (i, (x, mut next)) in next_gen
            .iter_sorted()
            .zip(population.iter_mut())
            .enumerate()
        {
            // Perform biased far-field random walk.
            next.copy_from(&next_gen.get(rand_perm1[i]).unwrap());
            temp.copy_from(&next_gen.get(rand_perm2[i]).unwrap());
            *next -= &*temp;
            next.apply(|uj| {
                *uj = if rng.gen_bool(abandon_prob) {
                    F::Scalar::zero()
                } else {
                    *uj * convert(rng.gen_range(0f64..=1.0))
                }
            });
            temp.copy_from(&x);
            *next += &*temp;

            // Make sure that the candidate is in domain.
            next.clamp(dom);

            // Evaluate and determine whether to abandon the nest.
            match next.eval(f, temp) {
                Ok(error)
                // Replace if it is better or it has been discovered while not
                // being in the elite.
                    if error < x.error()
                        || (rng.gen_bool(abandon_prob) && i >= elite_size) =>
                {
                    // Accept the candidate, just update the error.
                    next.set_error(error);
                }
                _ => {
                    // Reject the candidate, copy the old individual.
                    next.copy_from(&x);
                    next.set_error(x.error());
                },
            }
        }

        population.sort();

        // Assign te best individual.
        let best = population.iter_sorted().next().unwrap();
        x.copy_from(&best);

        let report = population.report();

        debug!(
            "best error = {}\taverage error = {}\tvalid/invalid ratio = {}:{}",
            report.best(),
            report.avg(),
            report.valid(),
            report.invalid(),
        );

        Ok(best.error())
    }
}

impl<F: Function, I: PopulationInit<F>, R: Rng> Optimizer<F> for Cuckoo<F, I, R>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    F::Dim: DimMin<F::Dim, Output = F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, <F::Dim as DimMin<F::Dim>>::Output>,
    DefaultAllocator: Reallocator<F::Scalar, F::Dim, F::Dim, F::Dim, F::Dim>,
    StandardNormal: Distribution<F::Scalar>,
    F::Scalar: SampleUniform + Float,
{
    const NAME: &'static str = "Cuckoo";
    type Error = CuckooError;

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

impl<F: System, I: PopulationInit<F>, R: Rng> Solver<F> for Cuckoo<F, I, R>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, F::Dim, F::Dim>,
    F::Dim: DimMin<F::Dim, Output = F::Dim>,
    DefaultAllocator: Allocator<F::Scalar, <F::Dim as DimMin<F::Dim>>::Output>,
    DefaultAllocator: Reallocator<F::Scalar, F::Dim, F::Dim, F::Dim, F::Dim>,
    StandardNormal: Distribution<F::Scalar>,
    F::Scalar: SampleUniform + Float,
{
    const NAME: &'static str = "Cuckoo";
    type Error = CuckooError;

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
            .and_then(|_| f.eval(x, fx).map_err(Into::into))
    }
}

fn rand_perm<R: Rng + ?Sized>(perm: &mut [usize], rng: &mut R) {
    // Based on https://en.wikipedia.org/wiki/Permutation#Algorithms_to_generate_permutations.
    for i in 0..perm.len() {
        let d = Uniform::new_inclusive(0, i).sample(rng);
        perm[i] = perm[d];
        perm[d] = i;
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    use crate::testing::*;

    #[test]
    fn rosenbrock() {
        let n = 4;

        let f = ExtendedRosenbrock::new(n);
        let dom = f.domain();

        for x in f.initials() {
            let mut errors = Vec::new();

            let solver = Cuckoo::new(&f, &dom, thread_rng());
            iter(&f, &dom, solver, x, 250, |_, _, error, _| {
                errors.push(error);
            })
            .unwrap();

            errors.dedup();

            assert!(errors.len() > 1, "no progress");
            assert!(
                errors.windows(2).all(|pair| pair[1] <= pair[0]),
                "error increase"
            );
        }
    }

    #[test]
    fn infinite_solutions() {
        let f = InfiniteSolutions::default();
        let dom = f.domain();
        let eps = convert(1e-12);

        for x in f.initials() {
            let solver = Cuckoo::new(&f, &dom, thread_rng());
            assert!(f.is_root(&solve(&f, &dom, solver, x, 25, eps).unwrap(), eps));
        }
    }
}
