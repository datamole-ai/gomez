//! The collection of implemented solvers.

pub mod cuckoo;
pub mod lipo;
pub mod nelder_mead;
pub mod steffensen;
pub mod trust_region;

pub use cuckoo::Cuckoo;
pub use lipo::Lipo;
pub use nelder_mead::NelderMead;
pub use steffensen::Steffensen;
pub use trust_region::TrustRegion;
