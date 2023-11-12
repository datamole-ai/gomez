use gomez::nalgebra as na;
use gomez::prelude::*;
use gomez::solver::TrustRegion;
use na::{Dynamic, IsContiguous};

// https://en.wikipedia.org/wiki/Rosenbrock_function
struct Rosenbrock {
    a: f64,
    b: f64,
}

impl Problem for Rosenbrock {
    type Scalar = f64;

    fn domain(&self) -> Domain<Self::Scalar> {
        Domain::unconstrained(2)
    }
}

impl System for Rosenbrock {
    fn eval<Sx, Sfx>(
        &self,
        x: &na::Vector<Self::Scalar, Dynamic, Sx>,
        fx: &mut na::Vector<Self::Scalar, Dynamic, Sfx>,
    ) where
        Sx: na::storage::Storage<Self::Scalar, Dynamic> + IsContiguous,
        Sfx: na::storage::StorageMut<Self::Scalar, Dynamic>,
    {
        fx[0] = (self.a - x[0]).powi(2);
        fx[1] = self.b * (x[1] - x[0].powi(2)).powi(2);
    }
}

fn main() -> Result<(), String> {
    let f = Rosenbrock { a: 1.0, b: 1.0 };
    let dom = f.domain();
    let mut solver = TrustRegion::new(&f, &dom);

    // Initial guess.
    let mut x = na::dvector![-10.0, -5.0];

    let mut fx = na::dvector![0.0, 0.0];

    for i in 1..=100 {
        solver
            .next(&f, &dom, &mut x, &mut fx)
            .map_err(|err| format!("{}", err))?;

        println!(
            "iter = {}\t|| fx || = {}\tx = {:?}",
            i,
            fx.norm(),
            x.as_slice()
        );

        if fx.norm() < 1e-6 {
            println!("solved");
            return Ok(());
        }
    }

    Err("did not converge".to_string())
}
