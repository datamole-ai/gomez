//! The collection of implemented algorithms.

pub mod lipo;
pub mod nelder_mead;
pub mod steffensen;
pub mod trust_region;

pub use lipo::Lipo;
pub use nelder_mead::NelderMead;
pub use steffensen::Steffensen;
pub use trust_region::TrustRegion;
