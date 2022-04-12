use criterion::{criterion_group, criterion_main, Criterion};
use gomez::{
    nalgebra as na,
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

fn solve<F, S>(
    f: &F,
    dom: &Domain<F::Scalar>,
    mut solver: S,
    mut x: na::OVector<F::Scalar, F::Dim>,
) -> bool
where
    F: TestSystem,
    S: Solver<F>,
    na::DefaultAllocator: na::allocator::Allocator<F::Scalar, F::Dim>,
    na::DefaultAllocator: na::allocator::Allocator<F::Scalar, F::Dim, F::Dim>,
    F::Dim: na::DimMin<F::Dim, Output = F::Dim>,
    na::DefaultAllocator:
        na::allocator::Allocator<F::Scalar, <F::Dim as na::DimMin<F::Dim>>::Output>,
    na::DefaultAllocator: na::allocator::Reallocator<F::Scalar, F::Dim, F::Dim, F::Dim, F::Dim>,
{
    let mut fx = x.clone_owned();
    let mut iter = 0;
    loop {
        if solver.next(f, dom, &mut x, &mut fx).is_err() {
            return false;
        }

        if fx.norm() < na::convert(TOLERANCE) {
            return true;
        }

        if iter == MAX_ITERS {
            return false;
        } else {
            iter += 1;
        }
    }
}

fn rosenbrock1(c: &mut Criterion) {
    let f = ExtendedRosenbrock::new(2);
    let dom = f.domain();
    let x = &f.initials()[0];

    c.bench_function("trust region rosenbrock 1", |b| {
        b.iter(|| assert!(solve(&f, &dom, TrustRegion::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("Nelder-Mead rosenbrock 1", |b| {
        b.iter(|| assert!(solve(&f, &dom, NelderMead::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("gsl hybrids rosenbrock 1", |b| {
        b.iter(|| {
            assert!(solve(
                &f,
                &dom,
                GslSolverWrapper::new(GslFunctionWrapper::new(f, GslVec::from(x.as_slice()))),
                x.clone_owned()
            ))
        })
    });
}

fn rosenbrock2(c: &mut Criterion) {
    let f = ExtendedRosenbrock::new(2);
    let dom = f.domain();
    let x = &f.initials()[1];

    c.bench_function("trust region rosenbrock 2", |b| {
        b.iter(|| assert!(solve(&f, &dom, TrustRegion::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("Nelder-Mead rosenbrock 2", |b| {
        b.iter(|| assert!(solve(&f, &dom, NelderMead::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("gsl hybrids rosenbrock 2", |b| {
        b.iter(|| {
            assert!(solve(
                &f,
                &dom,
                GslSolverWrapper::new(GslFunctionWrapper::new(f, GslVec::from(x.as_slice()))),
                x.clone_owned()
            ))
        })
    });
}

fn rosenbrock_scaled(c: &mut Criterion) {
    let f = ExtendedRosenbrock::with_scaling(2, 100.0);
    let dom = f.domain();
    let x = &f.initials()[1];

    c.bench_function("trust region rosenbrock scaled", |b| {
        b.iter(|| assert!(solve(&f, &dom, TrustRegion::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("Nelder-Mead rosenbrock scaled", |b| {
        b.iter(|| assert!(solve(&f, &dom, NelderMead::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("gsl hybrids rosenbrock scaled", |b| {
        b.iter(|| {
            assert!(solve(
                &f,
                &dom,
                GslSolverWrapper::new(GslFunctionWrapper::new(f, GslVec::from(x.as_slice()))),
                x.clone_owned()
            ))
        })
    });
}

fn rosenbrock_large(c: &mut Criterion) {
    let f = ExtendedRosenbrock::new(200);
    let dom = f.domain();
    let x = &f.initials()[0];

    c.bench_function("trust region rosenbrock large", |b| {
        b.iter(|| assert!(solve(&f, &dom, TrustRegion::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("gsl hybrids rosenbrock large", |b| {
        b.iter(|| {
            assert!(solve(
                &f,
                &dom,
                GslSolverWrapper::new(GslFunctionWrapper::new(f, GslVec::from(x.as_slice()))),
                x.clone_owned()
            ))
        })
    });
}

fn powell(c: &mut Criterion) {
    let f = ExtendedPowell::new(4);
    let dom = f.domain();
    let x = &f.initials()[0];

    c.bench_function("trust region powell", |b| {
        b.iter(|| assert!(solve(&f, &dom, TrustRegion::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("Nelder-Mead powell", |b| {
        b.iter(|| assert!(solve(&f, &dom, NelderMead::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("gsl hybrids powell", |b| {
        b.iter(|| {
            assert!(solve(
                &f,
                &dom,
                GslSolverWrapper::new(GslFunctionWrapper::new(f, GslVec::from(x.as_slice()))),
                x.clone_owned()
            ))
        })
    });
}

fn bullard_biegler(c: &mut Criterion) {
    let f = BullardBiegler::new();
    let dom = f.domain();
    let x = &f.initials()[0];

    c.bench_function("trust region bullard-biegler", |b| {
        b.iter(|| assert!(solve(&f, &dom, TrustRegion::new(&f, &dom), x.clone_owned())))
    });

    c.bench_function("gsl hybrids bullard-biegler", |b| {
        b.iter(|| {
            assert!(solve(
                &f,
                &dom,
                GslSolverWrapper::new(GslFunctionWrapper::new(f, GslVec::from(x.as_slice()))),
                x.clone_owned()
            ))
        })
    });
}

criterion_group!(
    solvers,
    rosenbrock1,
    rosenbrock2,
    rosenbrock_scaled,
    rosenbrock_large,
    powell,
    bullard_biegler
);
criterion_main!(solvers);

pub struct GslFunctionWrapper<F> {
    f: F,
    init: GslVec,
}

impl<F> GslFunctionWrapper<F> {
    pub fn new(f: F, init: GslVec) -> Self {
        Self { f, init }
    }
}

impl<F: TestSystem<Scalar = f64>> GslFunction for GslFunctionWrapper<F>
where
    na::DefaultAllocator: na::allocator::Allocator<F::Scalar, F::Dim>,
{
    fn eval(&self, x: &GslVec, f: &mut GslVec) -> GslStatus {
        use na::DimName;

        let x = na::MatrixSlice::<f64, F::Dim, na::U1>::from_slice_generic(
            x.as_slice(),
            self.f.dim(),
            na::U1::name(),
        );
        let mut fx = na::MatrixSliceMut::<f64, F::Dim, na::U1>::from_slice_generic(
            f.as_mut_slice(),
            self.f.dim(),
            na::U1::name(),
        );

        match self.f.eval(&x, &mut fx) {
            Ok(_) => GslStatus::ok(),
            Err(_) => GslStatus::err(GslError::BadFunc),
        }
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

impl<F: TestSystem<Scalar = f64>> Solver<F> for GslSolverWrapper<GslFunctionWrapper<F>>
where
    na::DefaultAllocator: na::allocator::Allocator<F::Scalar, F::Dim>,
    na::DefaultAllocator: na::allocator::Allocator<F::Scalar, F::Dim, F::Dim>,
    F::Dim: na::DimMin<F::Dim, Output = F::Dim>,
    na::DefaultAllocator:
        na::allocator::Allocator<F::Scalar, <F::Dim as na::DimMin<F::Dim>>::Output>,
    na::DefaultAllocator: na::allocator::Reallocator<F::Scalar, F::Dim, F::Dim, F::Dim, F::Dim>,
{
    const NAME: &'static str = "GSL hybrids";

    type Error = String;

    fn next<Sx, Sfx>(
        &mut self,
        _f: &F,
        _dom: &Domain<F::Scalar>,
        x: &mut na::Vector<F::Scalar, F::Dim, Sx>,
        fx: &mut na::Vector<F::Scalar, F::Dim, Sfx>,
    ) -> Result<(), Self::Error>
    where
        Sx: na::storage::StorageMut<F::Scalar, F::Dim> + IsContiguous,
        Sfx: na::storage::StorageMut<F::Scalar, F::Dim>,
    {
        let result = self.solver.step().to_result();
        x.copy_from_slice(self.solver.root());
        fx.copy_from_slice(self.solver.residuals());

        result
    }
}
