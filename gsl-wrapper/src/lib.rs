pub mod function;
pub mod multiroot;
pub mod status;
pub mod vec;

pub mod prelude {
    pub use super::function::Function;
    pub use super::status::{GslError, GslStatus};
    pub use super::vec::GslVec;
}
