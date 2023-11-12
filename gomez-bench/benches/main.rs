fn main() {
    divan::main();
}

use gomez::{
    nalgebra as na,
    nalgebra::Dynamic,
    prelude::*,
    solver::{NelderMead, TrustRegion},
    testing::*,
};
use gsl_wrapper::{
    function::Function as GslFunction,
    multiroot::{Solver as GslSolver, SolverType::HybridScaled},
    prelude::*,
};
use na::IsContiguous;

const MAX_ITERS: usize = 1_000_000;
const TOLERANCE: f64 = 1e-12;

#[divan::bench_group]
mod rosenbrock1 {
    use gomez::testing::ExtendedRosenbrock;

    use super::*;

    fn with_system() -> (ExtendedRosenbrock, usize) {
        (ExtendedRosenbrock::new(2), 0)
    }

    #[divan::bench]
    fn trust_region(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_trust_region);
    }

    #[divan::bench]
    fn nelder_mead(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_nelder_mead);
    }

    #[divan::bench]
    fn gsl_hybrids(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_gsl_hybrids);
    }
}

#[divan::bench_group]
mod rosenbrock2 {
    use gomez::testing::ExtendedRosenbrock;

    use super::*;

    fn with_system() -> (ExtendedRosenbrock, usize) {
        (ExtendedRosenbrock::new(2), 1)
    }

    #[divan::bench]
    fn trust_region(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_trust_region);
    }

    #[divan::bench]
    fn nelder_mead(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_nelder_mead);
    }

    #[divan::bench]
    fn gsl_hybrids(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_gsl_hybrids);
    }
}

#[divan::bench_group]
mod rosenbrock_scaled {
    use gomez::testing::ExtendedRosenbrock;

    use super::*;

    fn with_system() -> (ExtendedRosenbrock, usize) {
        (ExtendedRosenbrock::with_scaling(2, 100.0), 1)
    }

    #[divan::bench]
    fn trust_region(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_trust_region);
    }

    #[divan::bench]
    fn nelder_mead(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_nelder_mead);
    }

    #[divan::bench]
    fn gsl_hybrids(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_gsl_hybrids);
    }
}

#[divan::bench_group]
mod rosenbrock_large {
    use gomez::testing::ExtendedRosenbrock;

    use super::*;

    fn with_system() -> (ExtendedRosenbrock, usize) {
        (ExtendedRosenbrock::new(200), 0)
    }

    #[divan::bench]
    fn trust_region(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_trust_region);
    }

    #[divan::bench]
    fn gsl_hybrids(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_gsl_hybrids);
    }
}

#[divan::bench_group]
mod powell {
    use gomez::testing::ExtendedPowell;

    use super::*;

    fn with_system() -> (ExtendedPowell, usize) {
        (ExtendedPowell::new(4), 0)
    }

    #[divan::bench]
    fn trust_region(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_trust_region);
    }

    #[divan::bench]
    fn nelder_mead(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_nelder_mead);
    }

    #[divan::bench]
    fn gsl_hybrids(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_gsl_hybrids);
    }
}

#[divan::bench_group]
mod bullard_biegler {
    use gomez::testing::BullardBiegler;

    use super::*;

    fn with_system() -> (BullardBiegler, usize) {
        (BullardBiegler::new(), 0)
    }

    #[divan::bench]
    fn trust_region(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_trust_region);
    }

    #[divan::bench]
    fn gsl_hybrids(bencher: divan::Bencher) {
        bench_solve(bencher, with_system, with_gsl_hybrids);
    }
}

fn bench_solve<F, S, GF, GS>(bencher: divan::Bencher, with_system: GF, with_solver: GS)
where
    GF: Fn() -> (F, usize),
    GS: Fn(&F, &Domain<F::Scalar>, &na::OVector<F::Scalar, Dynamic>) -> S,
    F: TestSystem,
    S: Solver<F>,
{
    bencher
        .with_inputs(move || {
            let (f, initial) = with_system();
            let dom = f.domain();
            let x = f.initials()[initial].clone_owned();
            let solver = with_solver(&f, &dom, &x);
            (f, dom, x, solver)
        })
        .bench_local_values(|(f, dom, mut x, mut solver)| {
            let mut fx = x.clone_owned();
            let mut iter = 0;
            loop {
                if solver.solve_next(&f, &dom, &mut x, &mut fx).is_err() {
                    panic!("solver error");
                }

                if fx.norm() < na::convert(TOLERANCE) {
                    return true;
                }

                if iter == MAX_ITERS {
                    panic!("did not converge");
                } else {
                    iter += 1;
                }
            }
        });
}

fn with_trust_region<F>(
    f: &F,
    dom: &Domain<F::Scalar>,
    _: &na::OVector<F::Scalar, Dynamic>,
) -> TrustRegion<F>
where
    F: Problem,
{
    TrustRegion::new(f, dom)
}

fn with_nelder_mead<F>(
    f: &F,
    dom: &Domain<F::Scalar>,
    _: &na::OVector<F::Scalar, Dynamic>,
) -> NelderMead<F>
where
    F: Problem,
{
    NelderMead::new(f, dom)
}

fn with_gsl_hybrids<F>(
    f: &F,
    _: &Domain<F::Scalar>,
    x: &na::OVector<F::Scalar, Dynamic>,
) -> GslSolverWrapper<GslFunctionWrapper<F>>
where
    F: TestSystem<Scalar = f64> + Clone,
{
    GslSolverWrapper::new(GslFunctionWrapper::new(
        f.clone(),
        GslVec::from(x.as_slice()),
    ))
}

pub struct GslFunctionWrapper<F> {
    f: F,
    init: GslVec,
}

impl<F> GslFunctionWrapper<F> {
    pub fn new(f: F, init: GslVec) -> Self {
        Self { f, init }
    }
}

impl<F: TestSystem<Scalar = f64>> GslFunction for GslFunctionWrapper<F> {
    fn eval(&self, x: &GslVec, f: &mut GslVec) -> GslStatus {
        use na::DimName;
        let dim = Dynamic::new(x.len());

        let x = na::MatrixSlice::<f64, Dynamic, na::U1>::from_slice_generic(
            x.as_slice(),
            dim,
            na::U1::name(),
        );
        let mut fx = na::MatrixSliceMut::<f64, Dynamic, na::U1>::from_slice_generic(
            f.as_mut_slice(),
            dim,
            na::U1::name(),
        );

        self.f.eval(&x, &mut fx);
        GslStatus::ok()
    }

    fn init(&self) -> GslVec {
        self.init.clone()
    }
}

pub struct GslSolverWrapper<F> {
    solver: GslSolver<F>,
}

impl<F: GslFunction> GslSolverWrapper<F> {
    pub fn new(f: F) -> Self {
        Self {
            solver: GslSolver::new(f, HybridScaled),
        }
    }
}

impl<F: TestSystem<Scalar = f64>> Solver<F> for GslSolverWrapper<GslFunctionWrapper<F>> {
    const NAME: &'static str = "GSL hybrids";

    type Error = String;

    fn solve_next<Sx, Sfx>(
        &mut self,
        _f: &F,
        _dom: &Domain<F::Scalar>,
        x: &mut na::Vector<F::Scalar, Dynamic, Sx>,
        fx: &mut na::Vector<F::Scalar, Dynamic, Sfx>,
    ) -> Result<(), Self::Error>
    where
        Sx: na::storage::StorageMut<F::Scalar, Dynamic> + IsContiguous,
        Sfx: na::storage::StorageMut<F::Scalar, Dynamic>,
    {
        let result = self.solver.step().to_result();
        x.copy_from_slice(self.solver.root());
        fx.copy_from_slice(self.solver.residuals());

        result
    }
}
