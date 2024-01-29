use gomez::nalgebra as na;
use gomez::{Domain, Function, OptimizerDriver, Problem};
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

impl Function for Rosenbrock {
    fn apply<Sx>(&self, x: &na::Vector<Self::Field, Dyn, Sx>) -> Self::Field
    where
        Sx: na::Storage<Self::Field, Dyn> + IsContiguous,
    {
        (self.a - x[0]).powi(2) + self.b * (x[1] - x[0].powi(2)).powi(2)
    }
}

fn main() -> Result<(), String> {
    let f = Rosenbrock { a: 1.0, b: 1.0 };
    let mut optimizer = OptimizerDriver::builder(&f)
        .with_initial(vec![10.0, -5.0])
        .build();

    let tolerance = 1e-6;

    let (_, value) = optimizer
        .find(|state| {
            println!(
                "iter = {}\tf(x) = {}\tx = {:?}",
                state.iter(),
                state.fx(),
                state.x()
            );
            state.fx() <= tolerance || state.iter() >= 100
        })
        .map_err(|error| format!("{error}"))?;

    if value <= tolerance {
        Ok(())
    } else {
        Err("did not converge".to_string())
    }
}
