use fastrand::Rng;
use gomez::nalgebra as na;
use gomez::{Domain, Function, Optimizer, OptimizerDriver, Problem, Sample};
use na::{storage::StorageMut, Dyn, IsContiguous, Vector};

struct Random {
    rng: Rng,
}

impl Random {
    fn new(rng: Rng) -> Self {
        Self { rng }
    }
}

impl<F: Function> Optimizer<F> for Random
where
    F::Field: Sample,
{
    const NAME: &'static str = "Random";
    type Error = std::convert::Infallible;

    fn opt_next<Sx>(
        &mut self,
        f: &F,
        dom: &Domain<F::Field>,
        x: &mut Vector<F::Field, Dyn, Sx>,
    ) -> Result<F::Field, Self::Error>
    where
        Sx: StorageMut<F::Field, Dyn> + IsContiguous,
    {
        // Randomly sample in the domain.
        dom.sample(x, &mut self.rng);

        // We must compute the value.
        Ok(f.apply(x))
    }
}

// https://en.wikipedia.org/wiki/Rosenbrock_function
struct Rosenbrock {
    a: f64,
    b: f64,
}

impl Problem for Rosenbrock {
    type Field = f64;

    fn domain(&self) -> Domain<Self::Field> {
        Domain::rect(vec![-10.0, -10.0], vec![10.0, 10.0])
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

fn main() {
    let f = Rosenbrock { a: 1.0, b: 1.0 };
    let mut optimizer = OptimizerDriver::builder(&f)
        .with_algo(|_, _| Random::new(Rng::new()))
        .build();

    optimizer
        .find(|state| {
            println!("f(x) = {}\tx = {:?}", state.fx(), state.x());
            state.iter() >= 100
        })
        .unwrap();
}
