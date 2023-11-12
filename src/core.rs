//! Core abstractions and types for Gomez.
//!
//! *Users* are mainly interested in implementing the [`System`] and [`Problem`]
//! traits, optionally specifying the [domain](Domain).
//!
//! Algorithms *developers* are interested in implementing the [`Solver`] trait
//! (or possibly the [`Optimizer`] trait too) and using extension traits (e.g.,
//! [`VectorDomainExt`]) as well as tools in [derivatives](crate::derivatives).

mod base;
mod domain;
mod function;
mod optimizer;
mod solver;
mod system;

pub use base::*;
pub use domain::*;
pub use function::*;
pub use optimizer::*;
pub use solver::*;
pub use system::*;
