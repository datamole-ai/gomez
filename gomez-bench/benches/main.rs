fn main() {
    divan::main();
}

use gomez::{
    algo::{NelderMead, TrustRegion},
    nalgebra as na,
    nalgebra::Dyn,
    testing::*,
    Domain, Problem, Solver,
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

fn bench_solve<R, A, GR, GA>(bencher: divan::Bencher, with_system: GR, with_solver: GA)
where
    GR: Fn() -> (R, usize),
    GA: Fn(&R, &Domain<R::Field>, &na::OVector<R::Field, Dyn>) -> A,
    R: TestSystem,
    A: Solver<R>,
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
            let mut rx = x.clone_owned();
            let mut iter = 0;
            loop {
                if solver.solve_next(&f, &dom, &mut x, &mut rx).is_err() {
                    panic!("solver error");
                }

                if rx.norm() < na::convert(TOLERANCE) {
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

fn with_trust_region<R>(
    r: &R,
    dom: &Domain<R::Field>,
    _: &na::OVector<R::Field, Dyn>,
) -> TrustRegion<R>
where
    R: Problem,
{
    TrustRegion::new(r, dom)
}

fn with_nelder_mead<R>(
    r: &R,
    dom: &Domain<R::Field>,
    _: &na::OVector<R::Field, Dyn>,
) -> NelderMead<R>
where
    R: Problem,
{
    NelderMead::new(r, dom)
}

fn with_gsl_hybrids<R>(
    r: &R,
    _: &Domain<R::Field>,
    x: &na::OVector<R::Field, Dyn>,
) -> GslSolverWrapper<GslFunctionWrapper<R>>
where
    R: TestSystem<Field = f64> + Clone,
{
    GslSolverWrapper::new(GslFunctionWrapper::new(
        r.clone(),
        GslVec::from(x.as_slice()),
    ))
}

pub struct GslFunctionWrapper<R> {
    r: R,
    init: GslVec,
}

impl<R> GslFunctionWrapper<R> {
    pub fn new(r: R, init: GslVec) -> Self {
        Self { r, init }
    }
}

impl<R: TestSystem<Field = f64>> GslFunction for GslFunctionWrapper<R> {
    fn eval(&self, x: &GslVec, rx: &mut GslVec) -> GslStatus {
        use na::DimName;
        let dim = Dyn(x.len());

        let x = na::MatrixView::<f64, Dyn, na::U1>::from_slice_generic(
            x.as_slice(),
            dim,
            na::U1::name(),
        );
        let mut rx = na::MatrixViewMut::<f64, Dyn, na::U1>::from_slice_generic(
            rx.as_mut_slice(),
            dim,
            na::U1::name(),
        );

        self.r.eval(&x, &mut rx);
        GslStatus::ok()
    }

    fn init(&self) -> GslVec {
        self.init.clone()
    }
}

pub struct GslSolverWrapper<R> {
    solver: GslSolver<R>,
}

impl<R: GslFunction> GslSolverWrapper<R> {
    pub fn new(r: R) -> Self {
        Self {
            solver: GslSolver::new(r, HybridScaled),
        }
    }
}

impl<R: TestSystem<Field = f64>> Solver<R> for GslSolverWrapper<GslFunctionWrapper<R>> {
    const NAME: &'static str = "GSL hybrids";

    type Error = String;

    fn solve_next<Sx, Srx>(
        &mut self,
        _r: &R,
        _dom: &Domain<R::Field>,
        x: &mut na::Vector<R::Field, Dyn, Sx>,
        rx: &mut na::Vector<R::Field, Dyn, Srx>,
    ) -> Result<(), Self::Error>
    where
        Sx: na::storage::StorageMut<R::Field, Dyn> + IsContiguous,
        Srx: na::storage::StorageMut<R::Field, Dyn>,
    {
        let result = self.solver.step().to_result();
        x.copy_from_slice(self.solver.root());
        rx.copy_from_slice(self.solver.residuals());

        result
    }
}
