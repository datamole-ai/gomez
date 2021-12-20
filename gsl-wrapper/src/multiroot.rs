use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::{c_int, c_void};
use std::ptr::NonNull;

use crate::function::Function;
use crate::status::GslStatus;
use crate::vec::{as_slice, GslVec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    HybridScaled,
    Hybrid,
    DiscreteNewton,
    Broyden,
}

impl Default for SolverType {
    fn default() -> Self {
        SolverType::HybridScaled
    }
}

impl SolverType {
    fn as_raw(&self) -> *const gsl_sys::gsl_multiroot_fsolver_type {
        unsafe {
            match self {
                SolverType::HybridScaled => gsl_sys::gsl_multiroot_fsolver_hybrids,
                SolverType::Hybrid => gsl_sys::gsl_multiroot_fsolver_hybrid,
                SolverType::DiscreteNewton => gsl_sys::gsl_multiroot_fsolver_dnewton,
                SolverType::Broyden => gsl_sys::gsl_multiroot_fsolver_broyden,
            }
        }
    }
}

pub struct Solver<F> {
    solver: NonNull<gsl_sys::gsl_multiroot_fsolver>,
    func: Box<gsl_sys::gsl_multiroot_function>,
    ty: PhantomData<F>,
}

impl<F> Solver<F>
where
    F: Function,
{
    pub fn new(func: F, solver_type: SolverType) -> Self {
        let init = func.init();
        let size = init.len();

        // Put the function on the heap so the address does not change.
        let func = Box::new(func);

        let solver = unsafe { gsl_sys::gsl_multiroot_fsolver_alloc(solver_type.as_raw(), size) };
        let solver = NonNull::new(solver).expect("out of memory");

        unsafe fn inner<F>(
            x: *const gsl_sys::gsl_vector,
            params: *mut c_void,
            f: *mut gsl_sys::gsl_vector,
        ) -> c_int
        where
            F: Function,
        {
            // Wrap the raw pointers.
            let x = GslVec::from_raw(x as *mut _);
            let mut f = GslVec::from_raw(f);

            let func = &mut *(params as *mut F);
            let status = func.eval(&x, &mut f);

            // Leak the raw pointers again.
            GslVec::into_raw(x);
            GslVec::into_raw(f);

            status.to_raw()
        }

        let mut func = Box::new(gsl_sys::gsl_multiroot_function {
            f: Some(unsafe {
                mem::transmute::<_, unsafe extern "C" fn(_, _, _) -> _>(inner::<F> as *const ())
            }),
            n: size,
            params: Box::into_raw(func) as *mut c_void,
        });

        unsafe {
            gsl_sys::gsl_multiroot_fsolver_set(
                solver.as_ptr(),
                &mut *func as *mut _,
                init.as_raw() as *const _,
            );
        }

        Self {
            solver,
            func,
            ty: PhantomData,
        }
    }

    pub fn step(&mut self) -> GslStatus {
        let status = unsafe { gsl_sys::gsl_multiroot_fsolver_iterate(self.solver.as_ptr()) };
        GslStatus::from_raw(status)
    }

    pub fn test_residual(&self, eps: f64) -> GslStatus {
        assert!(eps >= 0.0);
        let status = unsafe {
            let f = gsl_sys::gsl_multiroot_fsolver_f(self.solver.as_ptr());
            gsl_sys::gsl_multiroot_test_residual(f, eps)
        };
        GslStatus::from_raw(status)
    }

    pub fn root(&self) -> &[f64] {
        unsafe { as_slice(gsl_sys::gsl_multiroot_fsolver_root(self.solver.as_ptr())) }
    }

    pub fn residuals(&self) -> &[f64] {
        unsafe { as_slice(gsl_sys::gsl_multiroot_fsolver_f(self.solver.as_ptr())) }
    }

    pub fn name(&self) -> &str {
        // SAFETY: The pointer returned from `gsl_multiroot_fsolver_name` points
        // to a static string.
        let cstr =
            unsafe { CStr::from_ptr(gsl_sys::gsl_multiroot_fsolver_name(self.solver.as_ptr())) };
        cstr.to_str().unwrap()
    }
}

impl<F> Drop for Solver<F> {
    fn drop(&mut self) {
        unsafe {
            Box::from_raw(self.func.params as *mut F);
            gsl_sys::gsl_multiroot_fsolver_free(self.solver.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::status::GslError;

    #[test]
    fn rosenbrock() {
        struct Rosenbrock {
            a: f64,
            b: f64,
        }

        impl Function for Rosenbrock {
            fn eval(&self, x: &GslVec, f: &mut GslVec) -> GslStatus {
                f[0] = self.a * (1.0 - x[0]);
                f[1] = self.b * (x[1] - x[0] * x[0]);
                GslStatus::ok()
            }

            fn init(&self) -> GslVec {
                GslVec::from(&[10.0, 5.0][..])
            }
        }

        let mut solver = Solver::new(Rosenbrock { a: 1.0, b: 10.0 }, SolverType::default());

        let mut status = GslStatus::err(GslError::Continue);

        for _ in 0..1000 {
            status = solver.step();

            if status.is_err() {
                break;
            }

            status = solver.test_residual(1e-7);

            if status != GslError::Continue {
                break;
            }
        }

        assert_eq!(status, GslStatus::ok());
        assert!(solver
            .root()
            .iter()
            .zip(GslVec::from(&[1.0, 1.0][..]).iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() <= 1e-7));
    }
}
