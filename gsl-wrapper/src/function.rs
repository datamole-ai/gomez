use crate::status::GslStatus;
use crate::vec::GslVec;

pub trait Function {
    fn eval(&self, x: &GslVec, f: &mut GslVec) -> GslStatus;
    fn init(&self) -> GslVec;
}
