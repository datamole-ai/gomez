//! Abstractions and types for population-based methods.
//!
//! The most important type is [`Population`].

use std::{
    cmp::Ordering,
    ops::{Deref, DerefMut},
};

use getset::CopyGetters;
use nalgebra::{
    allocator::Allocator, convert, storage::StorageMut, ComplexField, DefaultAllocator, Dim,
    DimName, OVector, Vector, U1,
};
use num_traits::Zero;
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Uniform};

use crate::core::{Domain, Function, Problem, ProblemError};

/// Population in a population-based solving algorithm.
///
/// There are two important invariants that the population must satisfy:
///
/// 1. Values of individuals and their corresponding errors must match. That is,
///    it must not happen that an individual is changed without updating its
///    error (see [`IndividualMut::set_error`](IndividualMut::set_error)).
/// 2. Before calling [`iter_sorted`](Population::iter_sorted) the population
///    must be sorted using [`sort`](Population::sort).
///
/// Violating any of these invariants results in panic in debug builds.
#[allow(clippy::len_without_is_empty)]
pub struct Population<F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    individuals: Vec<OVector<F::Scalar, F::Dim>>,
    errors: Vec<F::Scalar>,
    sorted: Vec<usize>,
    fx: OVector<F::Scalar, F::Dim>,
    sorted_dirty: bool,
}

impl<F: Problem> Population<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Creates new population with given initializer.
    pub fn new<R: Rng + ?Sized, I: PopulationInit<F>>(
        f: &F,
        dom: &Domain<F::Scalar>,
        rng: &mut R,
        initializer: &I,
        size: PopulationSize,
    ) -> Self
    where
        F: Function,
    {
        let size = size.get(f);

        let mut individuals = vec![OVector::zeros_generic(f.dim(), U1::name()); size];
        initializer.init_all(f, dom, rng, individuals.iter_mut());

        let errors = vec![F::Scalar::zero(); size];
        let sorted = (0..size).collect();
        let fx = OVector::zeros_generic(f.dim(), U1::name());

        let mut this = Self {
            individuals,
            errors,
            sorted,
            fx,
            sorted_dirty: true,
        };

        this.eval(f);
        this.sort();
        this
    }

    /// Creates new population initialized with zeros.
    ///
    /// This is usually useful for creating an additional population used for
    /// storing the next generation if the current one needs to be preserved.
    pub fn zeros(f: &F, size: PopulationSize) -> Self {
        let size = size.get(f);

        let individuals = vec![OVector::zeros_generic(f.dim(), U1::name()); size];
        let errors = vec![F::Scalar::zero(); size];
        let sorted = (0..size).collect();
        let fx = OVector::zeros_generic(f.dim(), U1::name());

        Self {
            individuals,
            errors,
            sorted,
            fx,
            sorted_dirty: false,
        }
    }

    /// Recreates the population with new individuals with given initializer.
    pub fn reinit<R: Rng + ?Sized, I: PopulationInit<F>>(
        &mut self,
        f: &F,
        dom: &Domain<F::Scalar>,
        rng: &mut R,
        initializer: &I,
    ) where
        F: Function,
    {
        initializer.init_all(f, dom, rng, self.individuals.iter_mut());
        self.eval(f);
        self.sort();
    }

    /// Get the size of the population.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Iterate over the population in order sorted by error from low to high..
    ///
    /// # Panics
    ///
    /// Panics if [`iter_mut`](Population::iter_mut) or
    /// [`get_mut`](Population::get_mut) was called without calling
    /// [`sort`](Population::sort) afterwards. This is the responsibility of the
    /// solving algorithm.
    pub fn iter_sorted(&self) -> IterSorted<'_, F> {
        debug_assert!(
            !self.sorted_dirty,
            "population is supposedly not sorted - this is a bug in the solving algorithm used"
        );
        IterSorted {
            individuals: &self.individuals,
            errors: &self.errors,
            sorted: self.sorted.iter(),
        }
    }

    /// Iterate over the population immutably.
    pub fn iter(&self) -> Iter<'_, F> {
        Iter {
            inner: self.individuals.iter().zip(self.errors.iter()),
        }
    }

    /// Iterate over the population mutably.
    ///
    /// **Important:** It is necessary to call [`sort`](Population::sort) before
    /// using (or allowing possibility of using)
    /// [`iter_sorted`](Population::iter_sorted).
    pub fn iter_mut(&mut self) -> IterMut<'_, F> {
        self.sorted_dirty = true;

        IterMut {
            inner: self.individuals.iter_mut().zip(self.errors.iter_mut()),
        }
    }

    /// Get an individual on specified index immutably.
    pub fn get(&self, index: usize) -> Option<Individual<'_, F>> {
        match (self.individuals.get(index), self.errors.get(index)) {
            (Some(x), Some(&error)) => Some(Individual { x, error }),
            _ => None,
        }
    }

    /// Get an individual on specified index mutably.
    ///
    /// **Important:** It is necessary to call [`sort`](Population::sort) before
    /// using (or allowing possibility of using)
    /// [`iter_sorted`](Population::iter_sorted).
    pub fn get_mut(&mut self, index: usize) -> Option<IndividualMut<'_, F>> {
        self.sorted_dirty = true;

        match (self.individuals.get_mut(index), self.errors.get_mut(index)) {
            (Some(x), Some(error)) => Some(IndividualMut {
                x,
                error,
                dirty: false,
            }),
            _ => None,
        }
    }

    /// Evaluate the whole population and store the errors.
    pub fn eval(&mut self, f: &F)
    where
        F: Function,
    {
        for (x, error) in self.individuals.iter().zip(self.errors.iter_mut()) {
            *error = f
                .apply_eval(x, &mut self.fx)
                .unwrap_or_else(|_| convert(f64::INFINITY));
        }
    }

    /// Sort the whole population ordered by errors of individuals from low to
    /// high.
    pub fn sort(&mut self) {
        let errors = &mut self.errors;
        self.sorted.sort_unstable_by(|lhs, rhs| {
            let lhs = errors[*lhs];
            let rhs = errors[*rhs];
            if lhs.is_finite() && rhs.is_finite() {
                lhs.partial_cmp(&rhs).unwrap()
            } else if lhs.is_finite() {
                Ordering::Less
            } else if rhs.is_finite() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });

        self.sorted_dirty = false;
    }

    /// Creates a simple report about the population in its current state.
    pub fn report(&self) -> PopulationReport<F> {
        PopulationReport::new(self)
    }
}

/// An individual from a population returned by [`get`](Population::get),
/// [`iter_sorted`](Population::iter_sorted) and [`iter`](Population::iter).
pub struct Individual<'a, F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    x: &'a OVector<F::Scalar, F::Dim>,
    error: F::Scalar,
}

impl<'a, F: Problem> Individual<'a, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Get the error of the individual.
    pub fn error(&self) -> F::Scalar {
        self.error
    }
}

impl<F: Problem> Deref for Individual<'_, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Target = OVector<F::Scalar, F::Dim>;

    fn deref(&self) -> &Self::Target {
        self.x
    }
}

/// Immutable iterator over a [population](`Population`).
///
/// For sorted version, see [`iter_sorted`](Population::iter_sorted).
pub struct Iter<'a, F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    inner: std::iter::Zip<
        std::slice::Iter<'a, OVector<F::Scalar, F::Dim>>,
        std::slice::Iter<'a, F::Scalar>,
    >,
}

impl<'a, F: Problem> Iterator for Iter<'a, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Item = Individual<'a, F>;

    fn next(&mut self) -> Option<Self::Item> {
        let (x, error) = self.inner.next()?;
        Some(Individual { x, error: *error })
    }
}

/// Immutable iterator over a [population](`Population`) sorted by error from
/// low to high.
///
/// For *un*sorted version, see [`iter`](Population::iter).
pub struct IterSorted<'a, F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    individuals: &'a [OVector<F::Scalar, F::Dim>],
    errors: &'a [F::Scalar],
    sorted: std::slice::Iter<'a, usize>,
}

impl<'a, F: Problem> Iterator for IterSorted<'a, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Item = Individual<'a, F>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = *self.sorted.next()?;
        Some(Individual {
            x: &self.individuals[index],
            error: self.errors[index],
        })
    }
}

/// An individual from a population returned by [`get_mut`](Population::get_mut)
/// and [`iter_mut`](Population::iter_mut).
///
/// **Important:** This type holds an information whether the individual is
/// *dirty* and supposedly needs reevaluation and correcting the error through
/// [`set_error`](IndividualMut::set_error). The individual is marked as dirty
/// whenever the underlying vector is accessed through [`DerefMut`]. It is the
/// responsibility of the solver algorithm to update the error before dropping
/// if any mutable dereference happened, otherwise it panics when dropped (in
/// debug builds).
///
/// ```rust
/// # use rand::{Rng, thread_rng};
/// # use rand_distr::StandardNormal;
/// # use gomez::{population::{Population, PopulationSize}, testing::{Sphere, TestSystem}};
/// # let f = Sphere::new(2);
/// # let mut fx = f.initials()[0].clone_owned();
/// # let mut rng = thread_rng();
/// #
/// # let mut population = Population::zeros(&f, PopulationSize::Adaptive);
/// #
/// for mut x in population.iter_mut() {
///     x.apply(|xi| *xi = rng.sample(StandardNormal));
///     let error = x.eval(&f, &mut fx).unwrap();
///     // Without the following line the code would panic!
///     // Because `x.apply` mutably borrows `x`.
///     x.set_error(error);
/// }
/// ```
pub struct IndividualMut<'a, F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    x: &'a mut OVector<F::Scalar, F::Dim>,
    error: &'a mut F::Scalar,
    dirty: bool,
}

impl<'a, F: Problem> IndividualMut<'a, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    /// Get the error of the individual.
    pub fn error(&self) -> F::Scalar {
        *self.error
    }

    /// Set the error of the individual.
    ///
    /// This unmarks the individual as dirty.
    ///
    /// **Important:** Dirty individuals cause panic when they are dropped.
    pub fn set_error(&mut self, error: F::Scalar) {
        *self.error = error;
        self.dirty = false;
    }

    /// Evaluate the individual.
    ///
    /// **Important:** This method does *not* unmark the individual as dirty. This
    /// needs to be done through [`set_error`](IndividualMut::set_error).
    pub fn eval<Sfx>(
        &self,
        f: &F,
        fx: &mut Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<F::Scalar, ProblemError>
    where
        Sfx: StorageMut<F::Scalar, F::Dim>,
        F: Function,
    {
        f.apply_eval(self.x, fx)
    }

    /// Clamp the individual to be with the bounds of given domain.
    ///
    /// **Important:** This marks the individual as dirty.
    pub fn clamp(&mut self, dom: &Domain<F::Scalar>) {
        self.x
            .iter_mut()
            .zip(dom.vars().iter())
            .for_each(|(xi, vi)| *xi = vi.clamp(*xi));

        self.dirty = true;
    }
}

impl<F: Problem> Deref for IndividualMut<'_, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Target = OVector<F::Scalar, F::Dim>;

    fn deref(&self) -> &Self::Target {
        self.x
    }
}

impl<F: Problem> DerefMut for IndividualMut<'_, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dirty = true;
        self.x
    }
}

impl<F: Problem> Drop for IndividualMut<'_, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    fn drop(&mut self) {
        debug_assert!(!self.dirty, "individual has supposedly obsolete error - this is a bug in the solving algorithm used");
    }
}

/// Mutable iterator over a [population](`Population`).
pub struct IterMut<'a, F: Problem>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    inner: std::iter::Zip<
        std::slice::IterMut<'a, OVector<F::Scalar, F::Dim>>,
        std::slice::IterMut<'a, F::Scalar>,
    >,
}

impl<'a, F: Problem> Iterator for IterMut<'a, F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    type Item = IndividualMut<'a, F>;

    fn next(&mut self) -> Option<Self::Item> {
        let (x, error) = self.inner.next()?;

        Some(IndividualMut {
            x,
            error,
            dirty: false,
        })
    }
}

/// A simple report about the population in its current state returned by
/// [`report`](Population::report) method.
#[derive(Debug, Clone, CopyGetters)]
#[get_copy = "pub"]
pub struct PopulationReport<F: Problem> {
    /// Error of the best individual in the population.
    best: F::Scalar,
    /// Average error of all individuals that have finite error.
    avg: F::Scalar,
    /// Number of individuals having a finite error.
    valid: usize,
    /// Number of individuals *not* having a finite error (i.e., the evaluation
    /// was not successful).
    invalid: usize,
}

impl<F: Problem> PopulationReport<F>
where
    DefaultAllocator: Allocator<F::Scalar, F::Dim>,
{
    fn new(population: &Population<F>) -> Self {
        let mut best = convert(f64::INFINITY);
        let mut sum = F::Scalar::zero();
        let mut valid = 0;
        let mut invalid = 0;

        for error in population.errors.iter().copied() {
            if error < best {
                best = error;
            }

            if error.is_finite() {
                sum += error;
                valid += 1;
            } else {
                invalid += 1;
            }
        }

        Self {
            best,
            avg: sum / convert(valid as f64),
            valid,
            invalid,
        }
    }
}

/// Trait defining an initialization of a population.
pub trait PopulationInit<F: Problem> {
    /// Initialize one individual in the population.
    fn init<R: Rng + ?Sized, S>(
        &self,
        f: &F,
        dom: &Domain<F::Scalar>,
        rng: &mut R,
        x: &mut Vector<F::Scalar, F::Dim, S>,
    ) where
        S: StorageMut<F::Scalar, F::Dim>;

    /// Initialize the whole population.
    fn init_all<'pop, R: Rng + ?Sized, I, S>(
        &self,
        f: &F,
        dom: &Domain<F::Scalar>,
        rng: &mut R,
        population: I,
    ) where
        I: Iterator<Item = &'pop mut Vector<F::Scalar, F::Dim, S>>,
        S: StorageMut<F::Scalar, F::Dim> + 'pop,
    {
        for x in population {
            self.init(f, dom, rng, x);
        }
    }
}

/// Initializes the population with uniform distribution within the bounds.
pub struct UniformInit<F: Problem> {
    factor: F::Scalar,
}

impl<F: Problem> UniformInit<F> {
    /// Creates the population initializer with given
    /// [factor](UniformInit::factor).
    pub fn with_factor(factor: F::Scalar) -> Self {
        Self { factor }
    }

    /// Creates the population initializer with default
    /// [factor](UniformInit::factor). Default: `10`.
    pub fn new() -> Self {
        Self::with_factor(convert(10.0))
    }

    /// Get the factor value that is used for estimating a bound when the actual
    /// bound is unconstrained.
    ///
    /// The estimated bound is computed as follows:
    /// * lower: `-magnitude * factor`
    /// * upper: `magnitude * factor`
    pub fn factor(&self) -> F::Scalar {
        self.factor
    }
}

impl<F: Problem> Default for UniformInit<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Problem> PopulationInit<F> for UniformInit<F>
where
    F::Scalar: SampleUniform,
{
    fn init<R: Rng + ?Sized, S>(
        &self,
        _f: &F,
        dom: &Domain<F::Scalar>,
        rng: &mut R,
        x: &mut Vector<F::Scalar, F::Dim, S>,
    ) where
        S: StorageMut<F::Scalar, F::Dim>,
    {
        x.iter_mut().zip(dom.vars().iter()).for_each(|(xi, vi)| {
            let lower = if vi.lower().is_finite() {
                vi.lower()
            } else {
                -vi.magnitude() * self.factor
            };
            let upper = if vi.upper().is_finite() {
                vi.upper()
            } else {
                vi.magnitude() * self.factor
            };
            *xi = Uniform::new_inclusive(lower, upper).sample(rng)
        });
    }
}

/// Setting for population size.
#[derive(Debug, Clone, Copy)]
pub enum PopulationSize {
    /// Fixed number of individuals.
    Fixed(usize),
    /// Number of individuals is based on the dimension of the system. The
    /// concrete heuristic is unspecified, but it is some nonlinear function
    /// with decreasing speed of growth.
    Adaptive,
}

impl PopulationSize {
    /// Get the determined number of individuals in the population, potentially
    /// influenced by the system dimension.
    ///
    /// It is guaranteed that the population is of size at least 2.
    pub fn get<F: Problem>(&self, f: &F) -> usize {
        let size = match self {
            PopulationSize::Fixed(size) => *size,
            PopulationSize::Adaptive => {
                // A nonlinearly increasing function with a reasonable minimum.
                let size = 10.0 + 5.0 * (f.dim().value() as f64).sqrt();
                // Round the size towards infinity to a multiplier of 5.
                let size = (size / 5.0).ceil() * 5.0;
                size as usize
            }
        };

        // The population should be always at least two individuals.
        size.max(2)
    }
}
