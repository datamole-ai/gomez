//! The collection of implemented solvers.

pub mod cuckoo;
pub mod nelder_mead;
pub mod trust_region;

pub use cuckoo::Cuckoo;
pub use nelder_mead::NelderMead;
pub use trust_region::TrustRegion;
