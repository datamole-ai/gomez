use gomez::nalgebra as na;
use gomez::{Domain, Problem, SolverDriver, System};
use na::{Dyn, IsContiguous};

// https://en.wikipedia.org/wiki/Rosenbrock_function
struct Rosenbrock {
    a: f64,
    b: f64,
}

impl Problem for Rosenbrock {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        Domain::unconstrained(2)
    }
}

impl System for Rosenbrock {
    fn eval<Sx, Srx>(
        &self,
        x: &na::Vector<Self::Field, Dyn, Sx>,
        rx: &mut na::Vector<Self::Field, Dyn, Srx>,
    ) where
        Sx: na::storage::Storage<Self::Field, Dyn> + IsContiguous,
        Srx: na::storage::StorageMut<Self::Field, Dyn>,
    {
        rx[0] = (self.a - x[0]).powi(2);
        rx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
    }
}

fn main() -> Result<(), String> {
    let r = Rosenbrock { a: 1.0, b: 1.0 };
    let mut solver = SolverDriver::builder(&r)
        .with_initial(vec![10.0, -5.0])
        .build();

    let tolerance = 1e-6;

    let (_, norm) = solver
        .find(|state| {
            println!(
                "iter = {}\t|| r(x) || = {}\tx = {:?}",
                state.iter(),
                state.norm(),
                state.x()
            );
            state.norm() <= tolerance || state.iter() >= 100
        })
        .map_err(|error| format!("{error}"))?;

    if norm <= tolerance {
        Ok(())
    } else {
        Err("did not converge".to_string())
    }
}
