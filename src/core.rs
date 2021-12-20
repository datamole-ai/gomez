//! Core abstractions and types for Gomez.
//!
//! *Users* are mainly interested in implementing the [`System`] trait,
//! optionally specifying the [domain](Domain).
//!
//! Algorithms *developers* are interested in implementing the [`Solver`] trait
//! and using extension traits [`SystemExt`] and [`VectorDomainExt`] as well as
//! tools in [derivatives](crate::derivatives) or
//! [population](crate::population) modules.

mod domain;
mod solver;
mod system;

pub use domain::*;
pub use solver::*;
pub use system::*;
